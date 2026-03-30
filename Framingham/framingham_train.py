# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions import Normal

from framingham_utils import kde_entropy
from framingham_config import *
from framingham_models import Z_given_WA, W_given_ZA, A_given_W, NN_W, NN_TW, h_net, w_net, t_net, g_net, NN_ZW, NN_M

# ------------------------
# High-level orchestrators
# ------------------------

def train_outcome_model(cfg, W_t_train, Z_t_train, A_t_train, Y_t_train,
        W_all_train, Z_all_train,
        W_t_val, Z_t_val, A_t_val, Y_t_val,
        W_all_val, Z_all_val, seed=42, USE_AE=False, USE_ENTROPY=False, ckpt_dir="checkpoints"):
    # best_W_ZA = outcome_generator_training(cfg, W_t, Z_t, A_t, W_all, Z_all, seed, ckpt_dir)
    best_outcome_models, best_val_outcome = outcome_bridge_training(cfg, W_t_train, Z_t_train, A_t_train, Y_t_train,
        W_all_train, Z_all_train,
        W_t_val, Z_t_val, A_t_val, Y_t_val,
        W_all_val, Z_all_val, seed, ckpt_dir, USE_AE=USE_AE, USE_ENTROPY=USE_ENTROPY)
    return best_outcome_models, best_val_outcome

def train_treatment_model(cfg, W_t_train, Z_t_train, A_t_train, Y_t_train,
        W_all_train, Z_all_train,
        W_t_val, Z_t_val, A_t_val, Y_t_val,
        W_all_val, Z_all_val, seed=42, USE_AE=False, USE_ENTROPY=False, ckpt_dir="checkpoints"):
    # best_Z_WA, best_W_A  = treatment_generator_training(cfg, W_t, Z_t, A_t, W_all, Z_all, seed, ckpt_dir)
    best_W_A  = treatment_generator_training(cfg, W_t_train, Z_t_train, A_t_train, W_t_val, Z_t_val, A_t_val, seed, ckpt_dir)
    best_treatment_models, best_val_treatment = treatment_bridge_training(cfg, W_t_train, Z_t_train, A_t_train, Y_t_train,
        W_all_train, Z_all_train,
        W_t_val, Z_t_val, A_t_val, Y_t_val,
        W_all_val, Z_all_val, seed, ckpt_dir, best_W_A, USE_AE=USE_AE, USE_ENTROPY=USE_ENTROPY)
    return best_treatment_models, best_val_treatment

# ------------------------
# Generators
# ------------------------

def outcome_generator_training(cfg, W_t, Z_t, A_t, seed=42, ckpt_dir="checkpoints"):
    """ Train p(W | Z, A) """
    dataset = TensorDataset(torch.cat([Z_t, A_t], dim=1), W_t)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.generator_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.generator_batch_size, shuffle=False)

    model = W_given_ZA().to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.generator_lr)
    best_val = float("inf")
    best_W_ZA = None

    for epoch in range(1, cfg.num_epochs + 101):
        model.train()
        train_loss = 0
        for cond, w_true in train_loader:
            optimizer.zero_grad()
            mu, sd = model(cond)
            dist = Normal(mu, sd)
            loss = -dist.log_prob(w_true).mean(dim=-1).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for cond, w_true in val_loader:
                mu, sd = model(cond)
                dist = Normal(mu, sd)
                val_loss += (-dist.log_prob(w_true).mean(dim=-1).mean()).item()
        avg_val = val_loss / len(val_loader)

        if avg_val < best_val:
            best_val = avg_val
            best_W_ZA = model
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"W_given_ZA_{cfg.run_name}.pth"))
            if cfg.logging:
                print(f"[Outcome Generator] Epoch {epoch:03d}, Val Loss={best_val:.4f} (saved)")
        
    return best_W_ZA


def treatment_generator_training(cfg, W_t_train, Z_t_train, A_t_train, W_t_val, Z_t_val, A_t_val, seed=42, ckpt_dir="checkpoints"):
    """ Train p(Z | W, A) and p(A | W) """

    # --- Train p(A | W) ---
    train_ds = TensorDataset(W_t_train, A_t_train)
    val_ds = TensorDataset(W_t_val, A_t_val)

    # dataset_A = TensorDataset(W_t, A_t)
    # n_train = int(0.8 * len(dataset_A))
    # n_val = len(dataset_A) - n_train
    # train_ds, val_ds = random_split(dataset_A, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.generator_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.generator_batch_size, shuffle=False)

    model_A = A_given_W().to(cfg.device)
    opt_A = optim.Adam(model_A.parameters(), lr=cfg.generator_lr)
    bce = nn.BCELoss()
    best_val = float("inf")

    for epoch in range(1, cfg.num_epochs):
        model_A.train()
        train_loss = 0
        for w_batch, a_true in train_loader:
            opt_A.zero_grad()
            p = model_A(w_batch)
            loss = bce(p, a_true)
            loss.backward()
            opt_A.step()
            train_loss += loss.item()

        val_loss = 0
        with torch.no_grad():
            model_A.eval()
            for w_batch, a_true in val_loader:
                p = model_A(w_batch)
                val_loss += bce(p, a_true).item()
        avg_val = val_loss / len(val_loader)

        if avg_val < best_val:
            best_val = avg_val
            best_W_A = model_A
            torch.save(model_A.state_dict(), os.path.join(ckpt_dir, f"A_given_W_{cfg.run_name}.pth"))
            if cfg.logging:
                print(f"[Treatment Generator A|W] Epoch {epoch:03d}, Val Loss={best_val:.4f} (saved)")

    return best_W_A

# ------------------------
# Bridges
# ------------------------

def outcome_bridge_training(cfg, W_t_train, Z_t_train, A_t_train, Y_t_train,
        W_all_train, Z_all_train,
        W_t_val, Z_t_val, A_t_val, Y_t_val,
        W_all_val, Z_all_val, seed=42, ckpt_dir="checkpoints", W_TZ=None,
                            USE_AE = False, USE_ENTROPY = False):
    """ Train outcome bridge networks (NN_W, NN_TW for A=0,1) """
    U_DIM = cfg.u_dim
    nn_w0, nn_w1 = NN_W(u_dim=U_DIM).to(cfg.device), NN_W(u_dim=U_DIM).to(cfg.device)
    nn_tw0, nn_tw1 = NN_TW(u_dim=U_DIM).to(cfg.device), NN_TW(u_dim=U_DIM).to(cfg.device)

    nn_zw = NN_ZW(u_dim=U_DIM).to(cfg.device)
    nn_m = NN_M(u_dim=U_DIM).to(cfg.device)

    J = cfg.J
    M = cfg.M
    EPS_DIM = cfg.eps_dim
    EPS_LAMBDA = cfg.eps_lambda
    DEVICE = cfg.device
    RUN_NAME = cfg.run_name
    AE_T_LAMBDA_OUTCOME = cfg.ae_t_lambda_outcome
    AE_Z_LAMBDA_OUTCOME = cfg.ae_z_lambda_outcome
    ENTROPY_LAMBDA_OUTCOME = cfg.entropy_lambda_outcome

    params = list(nn_w0.parameters()) + list(nn_w1.parameters()) + list(nn_tw0.parameters()) + list(nn_tw1.parameters()) + list(nn_zw.parameters()) + list(nn_m.parameters())
    optimizer = optim.Adam(params, lr=cfg.outcome_lr)
    mse_loss = nn.MSELoss()
    best_val = float("inf")

    bce = nn.BCELoss()

    W_all_train = W_all_train.permute(1, 0, 2).contiguous()
    W_all_val = W_all_val.permute(1, 0, 2).contiguous()

    train_ds = TensorDataset(Y_t_train, A_t_train, Z_t_train, W_t_train, W_all_train)
    val_ds = TensorDataset(Y_t_val, A_t_val, Z_t_val, W_t_val, W_all_val)

    # dataset = TensorDataset(Y_t, A_t, Z_t, W_t, W_all)
    # n_train = int(0.8 * len(dataset))
    # n_val = len(dataset) - n_train
    # train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    best_models = None

    for epoch in range(1, cfg.num_epochs + 1):
        # ——— Train ———
        nn_w0.train();  nn_w1.train()
        nn_tw0.train(); nn_tw1.train()
        nn_zw.train(); nn_m.train()
        train_loss = 0.0

        for (y_batch, a_batch, z_batch, w_batch, w_sample_batch) in train_loader:

            ######################outcome bridge######################

            total_loss = 0.0
            outcome_loss = 0.0

            # ——— Process T = 0 group ———
            mask0 = (a_batch.squeeze() == 0)
            if mask0.any():
                y0 = y_batch[mask0]   # [B0,1]
                z0 = z_batch[mask0]   # [B0,1]
                w0 = w_batch[mask0]
                B0 = y0.size(0)
                A0 = torch.zeros((B0,1), device=cfg.device)  # all zeros

                # # 5.1) Sample W ∼ p(W|Z,A=0,X)
                # conds0 = torch.cat([z0, A0], dim=1)         # [B0, 1+1+x_dim]
                # mean0, std0 = W_TZ(conds0)                       # each [B0,1]
                # w_dist0 = Normal(mean0, std0)
                # w_samps0 = w_dist0.sample((J,)).permute(1, 0, 2).detach()
                # # w_samps0 = w_samps0.reshape(B0 * J, 1)
                # # w_samps0 = w_samps0.repeat(1, M).view(-1, 1)
                # w_samps0 = w_samps0.reshape(B0*J, 16)       # was (B0*J,1)
                # w_samps0 = w_samps0.repeat(1, M).view(-1, 16)

                w_samps0 = w_sample_batch[mask0, :, :]        # still [J, B0, 16]
                # w_samps0 = w_samps0.permute(1, 0, 2)          # [B0, J, 16]

                # Expand across M (eps draws)
                w_samps0 = w_samps0.unsqueeze(2).repeat(1, 1, M, 1)   # [B0, J, M, 16]
                w_samps0 = w_samps0.reshape(B0 * J * M, 16)
                
                eps0 = torch.randn(B0 * J * M, EPS_DIM, device=cfg.device) * EPS_LAMBDA  # [(B0·n·m), ε_dim]
                U0_outcome = nn_w0(w_samps0, eps0)  # [(B0·n·m), u_dim]
                y0_hat = nn_tw0(U0_outcome, w_samps0)  # [(B0·n·m), 1]

                y0_hat = y0_hat.view(B0 * J, M, 1)
                y0_hat = y0_hat.mean(dim=1)               # [B0·n, 1]
                y0_hat = y0_hat.view(B0, J).mean(dim=1, keepdim=True)  # [B0,1]

                # 5.7) MSE vs true y0
                loss0 = bce(y0_hat, y0)
                total_loss += loss0
                outcome_loss += loss0


                ## Bridge for t ###
                if USE_AE:
                    z = z0.repeat(1, J * M).view(-1, 16)

                    bridge_x = nn_zw(z, U0_outcome)
                    bridge_x = bridge_x.view(B0 * J, M, -1).mean(dim=1)
                    bridge_x = bridge_x.view(B0, J).mean(dim = 1).unsqueeze(1)
                    l2 = bce(bridge_x, A0) * AE_T_LAMBDA_OUTCOME
                    
                    ## Bridge for z ###
                    bridge_z = nn_m(U0_outcome)
                    bridge_z = bridge_z.view(B0 * J, M, 16).mean(dim=1)
                    bridge_z = bridge_z.view(B0, J, 16).mean(dim = 1)
                    l3 = mse_loss(bridge_z, z0) * AE_Z_LAMBDA_OUTCOME

                    total_loss += (l2 + l3)
                    outcome_loss += (l2 + l3)

                if USE_ENTROPY:
                    entropy_reg = kde_entropy(U0_outcome, sigma=cfg.entropy_sigma)
                    total_loss -= entropy_reg * ENTROPY_LAMBDA_OUTCOME
                    outcome_loss -= entropy_reg * ENTROPY_LAMBDA_OUTCOME

            # ——— Process T = 1 group ———
            mask1 = (a_batch.squeeze() == 1)
            if mask1.any():
                y1 = y_batch[mask1]  # [B1,1]
                z1 = z_batch[mask1]  # [B1,1]
                w1 = w_batch[mask1]
                B1 = y1.size(0)
                A1 = torch.ones((B1,1), device=DEVICE)  # all ones

                # # 5.1) Sample W ∼ p(W|Z,A=1,X)
                # conds1 = torch.cat([z1, A1], dim=1)       # [B1,1+1+x_dim]
                # mean1, std1 = W_TZ(conds1)                     # each [B1,1]
                # w_dist1 = Normal(mean1, std1)
                # w_samps1 = w_dist1.sample((J,)).permute(1, 0, 2).detach()

                # w_samps1 = w_samps1.reshape(B1 * J, 16)
                # w_samps1 = w_samps1.repeat(1, M).view(-1, 16)

                w_samps1 = w_sample_batch[mask1, :, :]        # still [J, B0, 16]
                # w_samps1 = w_samps1.permute(1, 0, 2)          # [B0, J, 16]

                # Expand across M (eps draws)
                w_samps1 = w_samps1.unsqueeze(2).repeat(1, 1, M, 1)   # [B0, J, M, 16]
                w_samps1 = w_samps1.reshape(B1 * J * M, 16)

                # 5.3) ε for each copy
                eps1 = torch.randn(B1 * J * M, EPS_DIM, device=DEVICE) * EPS_LAMBDA

                # 5.4) U1 = nn_w1(W1, ε1, X1)
                U1_outcome = nn_w1(w_samps1, eps1)  # [(B1·n·m), u_dim]

                # 5.5) y1_hat = nn_tw1(U1, W1, X1)
                y1_hat = nn_tw1(U1_outcome, w_samps1)  # [(B1·n·m), 1]

                # 5.6) Reshape & average
                y1_hat = y1_hat.view(B1 * J, M, 1).mean(dim=1)  # [B1·n,1]
                y1_hat = y1_hat.view(B1, J).mean(dim=1, keepdim=True)                # [B1,1]

                # 5.7) MSE vs y1
                loss1 = bce(y1_hat, y1)
                total_loss += loss1
                outcome_loss += loss1

                ## Bridge for t ###
                if USE_AE:
                    z = z1.repeat(1, J * M).view(-1, 16)

                    bridge_x = nn_zw(z, U1_outcome)
                    bridge_x = bridge_x.view(B1 * J, M, -1).mean(dim=1)
                    bridge_x = bridge_x.view(B1, J).mean(dim = 1).unsqueeze(1)
                    l2 = bce(bridge_x, A1) * AE_T_LAMBDA_OUTCOME
                    
                    ## Bridge for z ###
                    bridge_z = nn_m(U1_outcome)
                    bridge_z = bridge_z.view(B1 * J, M, 16).mean(dim=1)
                    bridge_z = bridge_z.view(B1, J, 16).mean(dim = 1)
                    l3 = mse_loss(bridge_z, z1) * AE_Z_LAMBDA_OUTCOME

                    total_loss += (l2 + l3)
                    outcome_loss += (l2 + l3)

                if USE_ENTROPY:
                    entropy_reg = kde_entropy(U1_outcome, sigma=cfg.entropy_sigma)
                    total_loss -= entropy_reg * ENTROPY_LAMBDA_OUTCOME
                    outcome_loss -= entropy_reg * ENTROPY_LAMBDA_OUTCOME

            optimizer.zero_grad()
            outcome_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        avg_train = train_loss / len(train_loader)

        # ====================== VALIDATION LOOP ======================
        nn_w0.eval(); nn_w1.eval()
        nn_tw0.eval(); nn_tw1.eval()
        nn_zw.eval(); nn_m.eval()
        val_loss = 0.0
        
        with torch.no_grad():  # Disable gradient calculation
            for (y_batch, a_batch, z_batch, w_batch, w_sample_batch) in val_loader:
                # Transfer batch to device
                total_loss = 0.0

                # Process A=0 group
                mask0 = (a_batch.squeeze() == 0)
                if mask0.any():
                    y0 = y_batch[mask0]
                    z0 = z_batch[mask0]
                    B0 = y0.size(0)
                    A0 = torch.zeros((B0, 1), device=DEVICE)
                    
                    # Outcome bridge for T=0
                    # conds0 = torch.cat([z0, A0], dim=1)
                    # mean0, std0 = W_TZ(conds0)
                    # w_samps0 = Normal(mean0, std0).sample((J,)).permute(1, 0, 2)
                    # w_samps0 = w_samps0.reshape(B0 * J, 16).repeat(1, M).view(-1, 16)

                    w_samps0 = w_sample_batch[mask0, :, :]        # still [J, B0, 16]
                    # w_samps0 = w_samps0.permute(1, 0, 2)          # [B0, J, 16]

                    # Expand across M (eps draws)
                    w_samps0 = w_samps0.unsqueeze(2).repeat(1, 1, M, 1)   # [B0, J, M, 16]
                    w_samps0 = w_samps0.reshape(B0 * J * M, 16)

                    eps0 = torch.randn(B0 * J * M, EPS_DIM, device=DEVICE) * EPS_LAMBDA
                    
                    U0_outcome = nn_w0(w_samps0, eps0)
                    y0_hat = nn_tw0(U0_outcome, w_samps0)
                    y0_hat = y0_hat.view(B0 * J, M, 1).mean(1).view(B0, J).mean(1, keepdim=True)
                    
                    # loss0 = mse_loss(y0_hat, y0)
                    loss0 = bce(y0_hat, y0)
                    total_loss += loss0

                # Process A=1 group
                mask1 = (a_batch.squeeze() == 1)
                if mask1.any():
                    y1 = y_batch[mask1]
                    z1 = z_batch[mask1]
                    B1 = y1.size(0)
                    A1 = torch.ones((B1, 1), device=DEVICE)
                    
                    # Outcome bridge for T=1
                    # conds1 = torch.cat([z1, A1], dim=1)
                    # mean1, std1 = W_TZ(conds1)
                    # w_samps1 = Normal(mean1, std1).sample((J,)).permute(1, 0, 2)
                    # w_samps1 = w_samps1.reshape(B1 * J, 16).repeat(1, M).view(-1, 16)
                    w_samps1 = w_sample_batch[mask1, :, :]        # still [J, B0, 16]
                    # w_samps1 = w_samps1.permute(1, 0, 2)          # [B0, J, 16]

                    # Expand across M (eps draws)
                    w_samps1 = w_samps1.unsqueeze(2).repeat(1, 1, M, 1)   # [B0, J, M, 16]
                    w_samps1 = w_samps1.reshape(B1 * J * M, 16)

                    eps1 = torch.randn(B1 * J * M, EPS_DIM, device=DEVICE) * EPS_LAMBDA
                    
                    U1_outcome = nn_w1(w_samps1, eps1)
                    y1_hat = nn_tw1(U1_outcome, w_samps1)
                    y1_hat = y1_hat.view(B1 * J, M, 1).mean(1).view(B1, J).mean(1, keepdim=True)
                    
                    # loss1 = mse_loss(y1_hat, y1)
                    loss1 = bce(y1_hat, y1)
                    total_loss += loss1
                
                val_loss += total_loss.item()

        avg_val = val_loss / len(val_loader)
        # ====================== END VALIDATION ======================
        if cfg.logging:
            print(f"Epoch {epoch:03d} — Train Loss: {avg_train:.6f}, Val Loss: {avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            best_models = {"nn_w0": nn_w0, "nn_w1": nn_w1, "nn_tw0": nn_tw0, "nn_tw1": nn_tw1}
            torch.save(nn_w0.state_dict(), os.path.join(ckpt_dir, f"best_nn_w0_{RUN_NAME}.pth"))
            torch.save(nn_w1.state_dict(), os.path.join(ckpt_dir, f"best_nn_w1_{RUN_NAME}.pth"))
            torch.save(nn_tw0.state_dict(), os.path.join(ckpt_dir, f"best_nn_tw0_{RUN_NAME}.pth"))
            torch.save(nn_tw1.state_dict(), os.path.join(ckpt_dir, f"best_nn_tw1_{RUN_NAME}.pth"))
            if cfg.logging:
                print(f"  → Saved best outcome bridge (Val Loss={avg_val:.6f})")

    return best_models, best_val


def treatment_bridge_training(cfg, W_t_train, Z_t_train, A_t_train, Y_t_train,
        W_all_train, Z_all_train,
        W_t_val, Z_t_val, A_t_val, Y_t_val,
        W_all_val, Z_all_val, seed=42, ckpt_dir="checkpoints", W_A=None, 
                              USE_AE = False, USE_ENTROPY = False):
    """ Train treatment bridge networks (h_net, g_net, q_net, etc.) """
    best_treatment_loss_1 = float("inf")

    J = cfg.J
    M = cfg.M
    EPS_DIM = cfg.eps_dim
    EPS_LAMBDA = cfg.eps_lambda
    DEVICE = cfg.device
    RUN_NAME = cfg.run_name
    U_DIM = cfg.u_dim
    AE_T_LAMBDA_TREATMENT = cfg.ae_t_lambda_treatment
    AE_W_LAMBDA_TREATMENT = cfg.ae_w_lambda_treatment
    ENTROPY_LAMBDA_TREATMENT = cfg.entropy_lambda_treatment

    # Instantiate:
    h0 = h_net(eps_dim=EPS_DIM, u_dim=U_DIM).to(DEVICE)  # for A=0
    h1 = h_net(eps_dim=EPS_DIM, u_dim=U_DIM).to(DEVICE)  # for A=1
    w_model = w_net(u_dim=U_DIM).to(DEVICE)
    t_model = t_net(u_dim=U_DIM).to(DEVICE)
    g0 = g_net(u_dim=U_DIM).to(DEVICE)
    g1 = g_net(u_dim=U_DIM).to(DEVICE)

    best_models = None
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    optimizer_treatment_1 = torch.optim.AdamW(
        list(h0.parameters()) +
        list(h1.parameters()) +
        list(w_model.parameters()) +
        list(t_model.parameters()) +
        list(g0.parameters()) +
        list(g1.parameters()), lr=cfg.treatment_lr)
    
    Z_all_train = Z_all_train.permute(1, 0, 2).contiguous()
    Z_all_val = Z_all_val.permute(1, 0, 2).contiguous()

    train_ds = TensorDataset(Y_t_train, A_t_train, Z_t_train, W_t_train, Z_all_train)
    val_ds = TensorDataset(Y_t_val, A_t_val, Z_t_val, W_t_val, Z_all_val)

    # dataset = TensorDataset(Y_t, A_t, Z_t, W_t, Z_all)
    # n_train = int(0.8 * len(dataset))
    # n_val = len(dataset) - n_train
    # train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    for epoch in range(1, cfg.num_epochs + 1):
        # ——— Train ———
        h0.train(); h1.train(); g0.train(); g1.train(); w_model.train(); t_model.train()
        # q0_net.train(); q1_net.train()
        train_loss = 0.0

        for (y_batch, a_batch, z_batch, w_batch, z_all_batch) in train_loader:

            ######################outcome bridge######################

            B = y_batch.size(0)

            total_loss = 0.0

            # 4.1) inv_prop_W = 1 / p(A | W, X)
            pW = W_A(torch.cat([w_batch], dim=1))          # [B,1]
            pW_a = a_batch * pW + (1 - a_batch) * (1 - pW)               # p(A=Ab | W, X)
            invW = 1.0 / pW_a                         # stabilize

            # 4.2) Sample J draws Z_j ~ p(Z | Wb, Ab, Xb)
            # ZA_conds = torch.cat([w_batch, a_batch], dim=1)          # [B, 1+1+x_dim]
            # muZ, sdZ = Z_WA(ZA_conds)                       # each [B,1]
            # # Draw (J, B, 1) then permute → [B, J, 1]
            # Zs = Normal(muZ, sdZ).rsample((J,)).permute(1, 0, 2).detach()

            Zs = z_all_batch.to(DEVICE)

            invU_avg_full_1 = torch.zeros(B, 1, device=DEVICE)  # to store inv_pU_avg for each i
            invU_avg_full_2 = torch.zeros(B, 1, device=DEVICE)

            # ——— Process T = 0 group ———
            mask0 = (a_batch.squeeze() == 0)
            if mask0.any():
                y0 = y_batch[mask0]   # [B0,1]
                z0 = z_batch[mask0]   # [B0,1]
                w0 = w_batch[mask0]
                B0 = y0.size(0)
                A0 = torch.zeros((B0,1), device=DEVICE)  # all zeros

                idx0 = mask0.nonzero(as_tuple=False).squeeze(1)   # indices where A=0
                B0 = idx0.size(0)

                # Select W0, X0, Zs0:
                W0 = w_batch[idx0]                                    # [B0,1]
                Zs0 = Zs[idx0]                                   # [B0, J, 1]

                # Tile Zs0 & X0 & A0=0 over M eps
                Zs0_rep = Zs0.unsqueeze(2).repeat(1, 1, M, 1).view(-1, 16)   # [B0*J*M, 1]
                # Since A=0 always for this group:
                eps0    = torch.randn(B0 * J * M, EPS_DIM, device=DEVICE) * EPS_LAMBDA

                # Compute U0 samples:
                U0_treatment = h0(Zs0_rep, eps0)                # [B0*J*M, u_dim]

                if USE_AE:
                    W_pred = w_model(U0_treatment)
                    W_pred = W_pred.view(B0, J * M, 16).mean(dim=1)
                    total_loss += mse_loss(W_pred, W0) * AE_W_LAMBDA_TREATMENT

                    t_pred = t_model(U0_treatment, Zs0_rep)
                    t_pred = t_pred.view(B0, J * M, 1).mean(dim=1)
                    #treatment should use log probability instead of mse
                    total_loss += bce_loss(t_pred, A0) * AE_T_LAMBDA_TREATMENT

                if USE_ENTROPY:
                    entropy_reg = kde_entropy(U0_treatment, sigma=cfg.entropy_sigma)
                    total_loss -= entropy_reg * ENTROPY_LAMBDA_TREATMENT

                p1_U0 = g0(U0_treatment)
                # Average over J*M to get inv_pU_avg for each i in group0
                invU0_avg = p1_U0.view(B0, J * M, 1).mean(dim=1)  # [B0,1]
                # Scatter into the full batch tensor
                invU_avg_full_1[idx0] = invU0_avg

                # invU0_all = q0_net(Zs0)
                # # Average over J*M to get inv_pU_avg for each i in group0
                # invU0_avg = invU0_all.view(B0, J, 1).mean(dim=1)  # [B0,1]
                # # Scatter into the full batch tensor
                # invU_avg_full_2[idx0] = invU0_avg

            # ——— Process T = 1 group ———
            mask1 = (a_batch.squeeze() == 1)
            if mask1.any():
                y1 = y_batch[mask1]  # [B1,1]
                z1 = z_batch[mask1]  # [B1,1]
                w1 = w_batch[mask1]
                B1 = y1.size(0)
                A1 = torch.ones((B1,1), device=DEVICE)  # all ones

                idx1 = mask1.nonzero(as_tuple=False).squeeze(1)
                B1 = idx1.size(0)
                W1 = w_batch[idx1]
                Zs1 = Zs[idx1]  # [B1, J, 1]

                Zs1_rep = Zs1.unsqueeze(2).repeat(1, 1, M, 1).view(-1, 16)   # [B1*J*M, 1]
                eps1    = torch.randn(B1 * J * M, EPS_DIM, device=DEVICE) * EPS_LAMBDA

                U1_treatment = h1(Zs1_rep, eps1)                 # [B1*J*M, u_dim]

                if USE_AE:
                    W_pred = w_model(U1_treatment)
                    W_pred = W_pred.view(B1, J * M, 16).mean(dim=1)
                    total_loss += mse_loss(W_pred, W1) * AE_W_LAMBDA_TREATMENT

                    t_pred = t_model(U1_treatment, Zs1_rep)
                    t_pred = t_pred.view(B1, J * M, 1).mean(dim=1)
                    total_loss += bce_loss(t_pred, A1) * AE_T_LAMBDA_TREATMENT

                if USE_ENTROPY:
                    entropy_reg = kde_entropy(U1_treatment, sigma=cfg.entropy_sigma)
                    total_loss -= entropy_reg * ENTROPY_LAMBDA_TREATMENT

                p1_U1 = g1(U1_treatment)

                invU1_avg = p1_U1.view(B1, J * M, 1).mean(dim=1)  # [B1,1]
                invU_avg_full_1[idx1] = invU1_avg

                # invU1_all = q1_net(Zs1)
                # invU1_avg = invU1_all.view(B1, J, 1).mean(dim=1)  # [B1,1]
                # invU_avg_full_2[idx1] = invU1_avg


            loss1 = ((torch.log(invW) - torch.log(invU_avg_full_1)) ** 2).mean()
            # loss2 = ((invW - invU_avg_full_2) ** 2).mean()
            # print(invW)
            
            total_loss += loss1

            optimizer_treatment_1.zero_grad()

            total_loss.backward()

            optimizer_treatment_1.step()

            train_loss += total_loss.item()

        avg_train = train_loss / len(train_loader)

        # ====================== VALIDATION LOOP ======================
        h0.eval(); h1.eval(); g0.eval(); g1.eval(); w_model.eval(); t_model.eval()
        # q0_net.train(); q1_net.train()
        val_loss = 0.0
        
        with torch.no_grad():  # Disable gradient calculation
            for (y_batch, a_batch, z_batch, w_batch, z_all_batch) in val_loader:
                # Transfer batch to device
                
                B = y_batch.size(0)
                total_loss = 0.0

                # 1) Inverse propensity weighting for W
                pW = W_A(torch.cat([w_batch], dim=1))
                pW_a = a_batch * pW + (1 - a_batch) * (1 - pW)
                invW = 1.0 / (pW_a)

                # 2) Sample Z ~ p(Z|W,A,X)
                # ZA_conds = torch.cat([w_batch, a_batch], dim=1)
                # muZ, sdZ = Z_WA(ZA_conds)
                # Zs = Normal(muZ, sdZ).rsample((J,)).permute(1, 0, 2)
                Zs = z_all_batch.to(DEVICE)
                # Zs = z_all_batch.permute(1, 0, 2)   # [B, J, 16]
                
                invU_avg_full_1 = torch.zeros(B, 1, device=DEVICE)
                invU_avg_full_2 = torch.zeros(B, 1, device=DEVICE)

                # Process A=0 group
                mask0 = (a_batch.squeeze() == 0)
                if mask0.any():
                    y0 = y_batch[mask0]
                    z0 = z_batch[mask0]
                    B0 = y0.size(0)
                    A0 = torch.zeros((B0, 1), device=DEVICE)
                    
                    # Treatment bridge for T=0
                    idx0 = mask0.nonzero(as_tuple=False).squeeze(1)
                    Zs0 = Zs[idx0]
                    
                    Zs0_rep = Zs0.unsqueeze(2).repeat(1, 1, M, 1).view(-1, 16)
                    eps0 = torch.randn(B0 * J * M, EPS_DIM, device=DEVICE) * EPS_LAMBDA
                    
                    U0_treatment = h0(Zs0_rep, eps0)
                    p1_U0 = g0(U0_treatment)
                    invU0_avg = p1_U0.view(B0, J * M, 1).mean(dim=1)
                    invU_avg_full_1[idx0] = invU0_avg

                    # invU0_all = q0_net(Zs0)
                    # invU0_avg = invU0_all.view(B0, J, 1).mean(dim=1)  # [B0,1]
                    # invU_avg_full_2[idx0] = invU0_avg


                # Process A=1 group
                mask1 = (a_batch.squeeze() == 1)
                if mask1.any():
                    y1 = y_batch[mask1]
                    z1 = z_batch[mask1]
                    B1 = y1.size(0)
                    A1 = torch.ones((B1, 1), device=DEVICE)
                    
                    # Treatment bridge for T=1
                    idx1 = mask1.nonzero(as_tuple=False).squeeze(1)
                    Zs1 = Zs[idx1]
                    
                    Zs1_rep = Zs1.unsqueeze(2).repeat(1, 1, M, 1).view(-1, 16)
                    eps1 = torch.randn(B1 * J * M, EPS_DIM, device=DEVICE) * EPS_LAMBDA
                    
                    U1_treatment = h1(Zs1_rep, eps1)
                    p1_U1 = g1(U1_treatment)
                    invU1_avg = p1_U1.view(B1, J * M, 1).mean(1)
                    invU_avg_full_1[idx1] = invU1_avg

                    # invU1_all = q1_net(Zs1)
                    # invU1_avg = invU1_all.view(B1, J, 1).mean(dim=1)  # [B1,1]
                    # invU_avg_full_2[idx1] = invU1_avg

                loss1 = ((torch.log(invW) - torch.log(invU_avg_full_1)) ** 2).mean()
                # loss2 = ((invW - invU_avg_full_2) ** 2).mean()
                total_loss += loss1
                # print(invW)

                val_treatment_loss_1 = loss1.item()
                # val_treatment_loss_2 = loss2.item()

                val_loss += total_loss.item()

        avg_val = val_loss / len(val_loader)
        # ====================== END VALIDATION ======================

        if cfg.logging:
            print(f"Epoch {epoch:03d} — Train Loss: {avg_train:.6f}, Val Loss: {avg_val:.6f}")

        # if val_treatment_loss_2 < best_treatment_loss_2:
        #     best_treatment_loss_2 = val_treatment_loss_2
        #     torch.save(q0_net.state_dict(), "q0_sem.pth")
        #     torch.save(q1_net.state_dict(), "q1_sem.pth")
        #     print(f"  → Saved best treatment bridge (Val Loss={val_treatment_loss_2:.6f})")

        if avg_val < best_treatment_loss_1:
            best_treatment_loss_1 = avg_val
            best_models = {"h0": h0, "h1": h1, "g0": g0, "g1": g1}
            torch.save(h0.state_dict(), os.path.join(ckpt_dir, f"h0_sem_{RUN_NAME}.pth"))
            torch.save(h1.state_dict(), os.path.join(ckpt_dir, f"h1_sem_{RUN_NAME}.pth"))
            torch.save(g0.state_dict(), os.path.join(ckpt_dir, f"g0_sem_{RUN_NAME}.pth"))
            torch.save(g1.state_dict(), os.path.join(ckpt_dir, f"g1_sem_{RUN_NAME}.pth"))
            # torch.save(t_model.state_dict(), f"t_sem_{RUN_NAME}.pth")
            if cfg.logging:
                print(f"  → Saved best treatment bridge (Val Loss={avg_val:.6f})")

    return best_models, best_treatment_loss_1
