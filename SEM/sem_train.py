# sem_train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions import Normal

from sem_utils import kde_entropy
from sem_config import *
from sem_models import (
    Z_given_WAX, W_given_ZAX, A_given_WX,
    NN_WX, NN_TWX,
    h_net, w_net, t_net, g_net, NN_ZW, NN_M
)

# ------------------------
# High-level orchestrators
# ------------------------

def train_outcome_model(cfg, W_t, Z_t, A_t, Y_t, X_t,
                        seed=42, USE_AE=False, USE_ENTROPY=False,
                        ckpt_dir="checkpoints"):
    best_W_ZAX = outcome_generator_training(cfg, W_t, Z_t, A_t, X_t, seed, ckpt_dir)
    best_outcome_models = outcome_bridge_training(
        cfg, W_t, Z_t, A_t, Y_t, X_t, seed, ckpt_dir,
        best_W_ZAX, USE_AE=USE_AE, USE_ENTROPY=USE_ENTROPY
    )
    return best_outcome_models

def train_treatment_model(cfg, W_t, Z_t, A_t, Y_t, X_t,
                          seed=42, USE_AE=False, USE_ENTROPY=False,
                          ckpt_dir="checkpoints"):
    best_Z_WAX, best_W_X  = treatment_generator_training(cfg, W_t, Z_t, A_t, X_t, seed, ckpt_dir)
    best_treatment_models = treatment_bridge_training(
        cfg, W_t, Z_t, A_t, Y_t, X_t, seed, ckpt_dir,
        best_W_X, best_Z_WAX, USE_AE=USE_AE, USE_ENTROPY=USE_ENTROPY
    )
    return best_treatment_models

# ------------------------
# Generators
# ------------------------

def outcome_generator_training(cfg, W_t, Z_t, A_t, X_t,
                               seed=42, ckpt_dir="checkpoints"):
    """ Train p(W | Z, A, X) """
    dataset = TensorDataset(Z_t, A_t, X_t, W_t)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.generator_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.generator_batch_size, shuffle=False)

    model = W_given_ZAX(x_dim=cfg.x_dim).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.generator_lr)
    best_val = float("inf")
    best_model = None

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        for z, a, x, w_true in train_loader:
            optimizer.zero_grad()
            mu, sd = model(z, a, x)
            dist = Normal(mu, sd)
            loss = -dist.log_prob(w_true).mean()
            loss.backward()
            optimizer.step()

        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for z, a, x, w_true in val_loader:
                mu, sd = model(z, a, x)
                dist = Normal(mu, sd)
                val_loss += (-dist.log_prob(w_true).mean()).item()
        avg_val = val_loss / len(val_loader)

        if avg_val < best_val:
            best_val = avg_val
            best_model = model
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"W_given_ZAX_{cfg.run_name}.pth"))
            print(f"[Outcome Generator] Epoch {epoch:03d}, Val Loss={best_val:.4f} (saved)")

    return best_model


def treatment_generator_training(cfg, W_t, Z_t, A_t, X_t,
                                 seed=42, ckpt_dir="checkpoints"):
    """ Train p(Z | W, A, X) and p(A | W, X) """

    # --- Train p(Z | W, A, X) ---
    dataset = TensorDataset(W_t, A_t, X_t, Z_t)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.generator_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.generator_batch_size, shuffle=False)

    model_Z = Z_given_WAX(x_dim=cfg.x_dim).to(cfg.device)
    opt_Z = optim.Adam(model_Z.parameters(), lr=cfg.generator_lr)
    best_val = float("inf")
    best_model_Z = None

    for epoch in range(1, cfg.num_epochs + 1):
        model_Z.train()
        for w, a, x, z_true in train_loader:
            opt_Z.zero_grad()
            mu, sd = model_Z(w, a, x)
            dist = Normal(mu, sd)
            loss = -dist.log_prob(z_true).mean()
            loss.backward()
            opt_Z.step()

        # Validation
        val_loss = 0
        model_Z.eval()
        with torch.no_grad():
            for w, a, x, z_true in val_loader:
                mu, sd = model_Z(w, a, x)
                dist = Normal(mu, sd)
                val_loss += (-dist.log_prob(z_true).mean()).item()
        avg_val = val_loss / len(val_loader)

        if avg_val < best_val:
            best_val = avg_val
            best_model_Z = model_Z
            torch.save(model_Z.state_dict(), os.path.join(ckpt_dir, f"Z_given_WAX_{cfg.run_name}.pth"))
            print(f"[Treatment Generator Z|WAX] Epoch {epoch:03d}, Val Loss={best_val:.4f} (saved)")

    # --- Train p(A | W, X) ---
    dataset_A = TensorDataset(W_t, X_t, A_t)
    n_train = int(0.8 * len(dataset_A))
    n_val = len(dataset_A) - n_train
    train_ds, val_ds = random_split(dataset_A, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.generator_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.generator_batch_size, shuffle=False)

    model_A = A_given_WX(x_dim=cfg.x_dim).to(cfg.device)
    opt_A = optim.Adam(model_A.parameters(), lr=cfg.generator_lr)
    bce = nn.BCELoss()
    best_val = float("inf")
    best_model_A = None

    for epoch in range(1, cfg.num_epochs + 1):
        model_A.train()
        for w, x, a_true in train_loader:
            opt_A.zero_grad()
            p = model_A(w, x)
            loss = bce(p, a_true)
            loss.backward()
            opt_A.step()

        val_loss = 0
        model_A.eval()
        with torch.no_grad():
            for w, x, a_true in val_loader:
                p = model_A(w, x)
                val_loss += bce(p, a_true).item()
        avg_val = val_loss / len(val_loader)

        if avg_val < best_val:
            best_val = avg_val
            best_model_A = model_A
            torch.save(model_A.state_dict(), os.path.join(ckpt_dir, f"A_given_WX_{cfg.run_name}.pth"))
            print(f"[Treatment Generator A|WX] Epoch {epoch:03d}, Val Loss={best_val:.4f} (saved)")

    return best_model_Z, best_model_A

# ------------------------
# Bridges
# ------------------------

def outcome_bridge_training(cfg, W_t, Z_t, A_t, Y_t, X_t,
                            seed=42, ckpt_dir="checkpoints", W_TZX=None,
                            USE_AE=False, USE_ENTROPY=False):
    """ Train outcome bridge networks (NN_WX, NN_TWX for A=0,1) with SEM data """

    nn_w0, nn_w1 = NN_WX(x_dim=cfg.x_dim).to(cfg.device), NN_WX(x_dim=cfg.x_dim).to(cfg.device)
    nn_tw0, nn_tw1 = NN_TWX(x_dim=cfg.x_dim).to(cfg.device), NN_TWX(x_dim=cfg.x_dim).to(cfg.device)
    nn_zw = NN_ZW().to(cfg.device)
    nn_m  = NN_M().to(cfg.device)

    J, M = cfg.J, cfg.M
    EPS_DIM, EPS_LAMBDA = cfg.eps_dim, cfg.eps_lambda
    DEVICE, RUN_NAME = cfg.device, cfg.run_name

    params = (list(nn_w0.parameters()) + list(nn_w1.parameters()) +
              list(nn_tw0.parameters()) + list(nn_tw1.parameters()) +
              list(nn_zw.parameters()) + list(nn_m.parameters()))
    optimizer = optim.Adam(params, lr=cfg.outcome_lr)
    mse_loss, bce = nn.MSELoss(), nn.BCELoss()
    best_val, best_models = float("inf"), None

    dataset = TensorDataset(Y_t, A_t, Z_t, W_t, X_t)
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    for epoch in range(1, cfg.num_epochs + 1):
        nn_w0.train(); nn_w1.train(); nn_tw0.train(); nn_tw1.train()
        train_loss = 0.0

        for (y, a, z, w, x) in train_loader:
            total_loss = 0.0

            # ----- A=0 group -----
            mask0 = (a.squeeze() == 0)
            if mask0.any():
                y0, z0, x0 = y[mask0], z[mask0], x[mask0]
                B0 = y0.size(0)
                A0 = torch.zeros((B0, 1), device=DEVICE)

                mu, sd = W_TZX(z0, A0, x0)
                w_samps0 = Normal(mu, sd).sample((J,)).permute(1, 0, 2).detach()  # (B0,J,1)
                w_samps0 = w_samps0.reshape(B0*J, 1).repeat(1, M).view(-1, 1)      # (B0*J*M,1)

                # align x with (J,M) repeats
                x0_rep = x0.unsqueeze(1).unsqueeze(2).expand(B0, J, M, cfg.x_dim).reshape(B0*J*M, cfg.x_dim)

                eps0 = torch.randn(B0*J*M, EPS_DIM, device=DEVICE) * EPS_LAMBDA
                U0 = nn_w0(w_samps0, x0_rep, eps0)

                y0_hat = nn_tw0(U0, w_samps0, x0_rep).view(B0*J, M, 1).mean(1).view(B0, J).mean(1, keepdim=True)

                loss0 = mse_loss(y0_hat, y0)
                total_loss += loss0

                if USE_AE:
                    z0_rep = z0.unsqueeze(1).unsqueeze(2).expand(B0, J, M, z0.size(1)).reshape(B0*J*M, z0.size(1))
                    bridge_x = nn_zw(z0_rep, x0_rep, U0)
                    bridge_x = bridge_x.view(B0*J, M, -1).mean(1).view(B0, J).mean(1, keepdim=True)
                    l2 = bce(bridge_x, A0)

                    bridge_z = nn_m(x0_rep, U0).view(B0*J, M, -1).mean(1).view(B0, J, -1).mean(1)
                    l3 = mse_loss(bridge_z, z0)

                    total_loss += (l2 + l3) * cfg.ae_lambda_outcome

                if USE_ENTROPY:
                    entropy_reg = kde_entropy(U0, sigma=cfg.entropy_sigma)
                    total_loss -= entropy_reg * cfg.entropy_lambda_outcome

            # ----- A=1 group -----
            mask1 = (a.squeeze() == 1)
            if mask1.any():
                y1, z1, x1 = y[mask1], z[mask1], x[mask1]
                B1 = y1.size(0)
                A1 = torch.ones((B1, 1), device=DEVICE)

                mu, sd = W_TZX(z1, A1, x1)
                w_samps1 = Normal(mu, sd).sample((J,)).permute(1, 0, 2).detach()
                w_samps1 = w_samps1.reshape(B1*J, 1).repeat(1, M).view(-1, 1)

                x1_rep = x1.unsqueeze(1).unsqueeze(2).expand(B1, J, M, cfg.x_dim).reshape(B1*J*M, cfg.x_dim)

                eps1 = torch.randn(B1*J*M, EPS_DIM, device=DEVICE) * EPS_LAMBDA
                U1 = nn_w1(w_samps1, x1_rep, eps1)

                y1_hat = nn_tw1(U1, w_samps1, x1_rep).view(B1*J, M, 1).mean(1).view(B1, J).mean(1, keepdim=True)

                loss1 = mse_loss(y1_hat, y1)
                total_loss += loss1

                if USE_AE:
                    z1_rep = z1.unsqueeze(1).unsqueeze(2).expand(B1, J, M, z1.size(1)).reshape(B1*J*M, z1.size(1))
                    bridge_x = nn_zw(z1_rep, x1_rep, U1)
                    bridge_x = bridge_x.view(B1*J, M, -1).mean(1).view(B1, J).mean(1, keepdim=True)
                    l2 = bce(bridge_x, A1)

                    bridge_z = nn_m(x1_rep, U1).view(B1*J, M, -1).mean(1).view(B1, J, -1).mean(1)
                    l3 = mse_loss(bridge_z, z1)

                    total_loss += (l2 + l3) * cfg.ae_lambda_outcome

                if USE_ENTROPY:
                    entropy_reg = kde_entropy(U1, sigma=cfg.entropy_sigma)
                    total_loss -= entropy_reg * cfg.entropy_lambda_outcome

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        # Validation (same repeat logic for x + w)
        avg_val = 0.0
        nn_w0.eval(); nn_w1.eval(); nn_tw0.eval(); nn_tw1.eval()
        with torch.no_grad():
            for (y, a, z, w, x) in val_loader:
                val_loss = 0.0
                mask0 = (a.squeeze() == 0)
                if mask0.any():
                    y0, z0, x0 = y[mask0], z[mask0], x[mask0]
                    B0 = y0.size(0); A0 = torch.zeros((B0,1), device=DEVICE)
                    mu, sd = W_TZX(z0, A0, x0)
                    w_samps0 = Normal(mu, sd).sample((J,)).permute(1,0,2)
                    w_samps0 = w_samps0.reshape(B0*J,1).repeat(1,M).view(-1,1)
                    x0_rep = x0.unsqueeze(1).unsqueeze(2).expand(B0,J,M,cfg.x_dim).reshape(B0*J*M,cfg.x_dim)
                    eps0 = torch.randn(B0*J*M, EPS_DIM, device=DEVICE)*EPS_LAMBDA
                    U0 = nn_w0(w_samps0, x0_rep, eps0)
                    y0_hat = nn_tw0(U0, w_samps0, x0_rep).view(B0*J,M,1).mean(1).view(B0,J).mean(1,keepdim=True)
                    val_loss += mse_loss(y0_hat, y0)

                mask1 = (a.squeeze() == 1)
                if mask1.any():
                    y1, z1, x1 = y[mask1], z[mask1], x[mask1]
                    B1 = y1.size(0); A1 = torch.ones((B1,1), device=DEVICE)
                    mu, sd = W_TZX(z1, A1, x1)
                    w_samps1 = Normal(mu, sd).sample((J,)).permute(1,0,2)
                    w_samps1 = w_samps1.reshape(B1*J,1).repeat(1,M).view(-1,1)
                    x1_rep = x1.unsqueeze(1).unsqueeze(2).expand(B1,J,M,cfg.x_dim).reshape(B1*J*M,cfg.x_dim)
                    eps1 = torch.randn(B1*J*M, EPS_DIM, device=DEVICE)*EPS_LAMBDA
                    U1 = nn_w1(w_samps1, x1_rep, eps1)
                    y1_hat = nn_tw1(U1, w_samps1, x1_rep).view(B1*J,M,1).mean(1).view(B1,J).mean(1,keepdim=True)
                    val_loss += mse_loss(y1_hat, y1)

                avg_val += val_loss.item()
        avg_val /= len(val_loader)

        print(f"[Outcome Bridge] Epoch {epoch:03d}, TrainLoss={train_loss/len(train_loader):.4f}, ValLoss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            best_models = {"nn_w0": nn_w0, "nn_w1": nn_w1, "nn_tw0": nn_tw0, "nn_tw1": nn_tw1}
            os.makedirs(ckpt_dir, exist_ok=True)
            for name, model in best_models.items():
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{name}_{RUN_NAME}.pth"))
            print(f"  → Saved best outcome bridge (Val Loss={avg_val:.4f})")

    return best_models


def treatment_bridge_training(cfg, W_t, Z_t, A_t, Y_t, X_t,
                              seed=42, ckpt_dir="checkpoints", W_X=None, Z_WAX=None,
                              USE_AE=False, USE_ENTROPY=False):
    """ Train treatment bridge networks (h_net, g_net, etc.) with SEM data """
    J, M = cfg.J, cfg.M
    EPS_DIM, EPS_LAMBDA = cfg.eps_dim, cfg.eps_lambda
    DEVICE, RUN_NAME = cfg.device, cfg.run_name

    h0 = h_net(x_dim=cfg.x_dim, eps_dim=cfg.eps_dim, u_dim=cfg.u_dim).to(DEVICE)
    h1 = h_net(x_dim=cfg.x_dim, eps_dim=cfg.eps_dim, u_dim=cfg.u_dim).to(DEVICE)
    g0 = g_net(u_dim=cfg.u_dim, x_dim=cfg.x_dim).to(DEVICE)
    g1 = g_net(u_dim=cfg.u_dim, x_dim=cfg.x_dim).to(DEVICE)

    w_model = w_net(u_dim=cfg.u_dim, x_dim=cfg.x_dim).to(DEVICE)
    t_model = t_net(x_dim=cfg.x_dim).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(h0.parameters()) + list(h1.parameters()) +
        list(g0.parameters()) + list(g1.parameters()) +
        list(w_model.parameters()) + list(t_model.parameters()),
        lr=cfg.treatment_lr
    )

    dataset = TensorDataset(Y_t, A_t, Z_t, W_t, X_t)
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    best_val, best_models = float("inf"), None
    mse_loss, bce_loss = nn.MSELoss(), nn.BCELoss()

    for epoch in range(1, cfg.num_epochs + 1):
        h0.train(); h1.train(); g0.train(); g1.train(); w_model.train(); t_model.train()
        train_loss = 0.0

        for y, a, z, w, x in train_loader:
            B = y.size(0)
            total_loss = 0.0

            # 1) Inverse propensity 1/p(A|W,X)
            pA = W_X(w, x)
            pA_true = a * pA + (1 - a) * (1 - pA)
            invW = 1.0 / (pA_true + 1e-8)

            # 2) Sample Z ~ p(Z|W,A,X)
            muZ, sdZ = Z_WAX(w, a, x)
            Zs = Normal(muZ, sdZ).sample((J,)).permute(1, 0, 2).detach()  # (B,J,1)

            invU_avg = torch.zeros(B, 1, device=DEVICE)

            # -------- A=0 group --------
            mask0 = (a.squeeze() == 0)
            if mask0.any():
                X0, Zs0, W0 = x[mask0], Zs[mask0], w[mask0]
                B0 = X0.size(0)
                A0 = torch.zeros((B0,1), device=DEVICE)

                Zs0_rep = Zs0.reshape(B0*J,1).repeat(1, M).view(-1,1)
                X0_rep  = X0.unsqueeze(1).unsqueeze(2).expand(B0,J,M,cfg.x_dim).reshape(B0*J*M, cfg.x_dim)
                eps0    = torch.randn(B0*J*M, EPS_DIM, device=DEVICE) * EPS_LAMBDA

                U0 = h0(Zs0_rep, X0_rep, eps0)

                if USE_AE:
                    W_pred = w_model(X0_rep, U0).view(B0,J*M,1).mean(1)
                    total_loss += mse_loss(W_pred, W0) * 1e-3

                    t_pred = t_model(X0_rep, U0, Zs0_rep).view(B0,J*M,1).mean(1)
                    total_loss += bce_loss(t_pred, A0) * cfg.ae_lambda_treatment

                if USE_ENTROPY:
                    entropy_reg = kde_entropy(U0, sigma=cfg.entropy_sigma)
                    total_loss -= entropy_reg * cfg.entropy_lambda_treatment

                p1_U0 = g0(U0, X0_rep)
                invU0_avg = p1_U0.view(B0,J*M,1).mean(1)
                invU_avg[mask0] = invU0_avg

            # -------- A=1 group --------
            mask1 = (a.squeeze() == 1)
            if mask1.any():
                X1, Zs1, W1 = x[mask1], Zs[mask1], w[mask1]
                B1 = X1.size(0)
                A1 = torch.ones((B1,1), device=DEVICE)

                Zs1_rep = Zs1.reshape(B1*J,1).repeat(1, M).view(-1,1)
                X1_rep  = X1.unsqueeze(1).unsqueeze(2).expand(B1,J,M,cfg.x_dim).reshape(B1*J*M, cfg.x_dim)
                eps1    = torch.randn(B1*J*M, EPS_DIM, device=DEVICE) * EPS_LAMBDA

                U1 = h1(Zs1_rep, X1_rep, eps1)

                if USE_AE:
                    W_pred = w_model(X1_rep, U1).view(B1,J*M,1).mean(1)
                    total_loss += mse_loss(W_pred, W1) * 1e-3

                    t_pred = t_model(X1_rep, U1, Zs1_rep).view(B1,J*M,1).mean(1)
                    total_loss += bce_loss(t_pred, A1) * cfg.ae_lambda_treatment

                if USE_ENTROPY:
                    entropy_reg = kde_entropy(U1, sigma=cfg.entropy_sigma)
                    total_loss -= entropy_reg * cfg.entropy_lambda_treatment

                p1_U1 = g1(U1, X1_rep)
                invU1_avg = p1_U1.view(B1,J*M,1).mean(1)
                invU_avg[mask1] = invU1_avg

            # fit loss
            loss = ((torch.log(invW+1e-8) - torch.log(invU_avg+1e-8))**2).mean()
            total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        # ---------------- Validation ----------------
        h0.eval(); h1.eval(); g0.eval(); g1.eval()
        avg_val = 0.0
        with torch.no_grad():
            for y, a, z, w, x in val_loader:
                B = y.size(0)
                val_loss = 0.0

                pA = W_X(w, x)
                pA_true = a * pA + (1 - a) * (1 - pA)
                invW = 1.0 / (pA_true + 1e-8)

                muZ, sdZ = Z_WAX(w, a, x)
                Zs = Normal(muZ, sdZ).sample((J,)).permute(1, 0, 2)

                invU_avg = torch.zeros(B, 1, device=DEVICE)

                mask0 = (a.squeeze()==0)
                if mask0.any():
                    X0, Zs0 = x[mask0], Zs[mask0]
                    B0 = X0.size(0)
                    Zs0_rep = Zs0.reshape(B0*J,1).repeat(1,M).view(-1,1)
                    X0_rep  = X0.unsqueeze(1).unsqueeze(2).expand(B0,J,M,cfg.x_dim).reshape(B0*J*M,cfg.x_dim)
                    eps0    = torch.randn(B0*J*M, EPS_DIM, device=DEVICE)*EPS_LAMBDA
                    U0 = h0(Zs0_rep, X0_rep, eps0)
                    p1_U0 = g0(U0, X0_rep)
                    invU0_avg = p1_U0.view(B0,J*M,1).mean(1)
                    invU_avg[mask0] = invU0_avg

                mask1 = (a.squeeze()==1)
                if mask1.any():
                    X1, Zs1 = x[mask1], Zs[mask1]
                    B1 = X1.size(0)
                    Zs1_rep = Zs1.reshape(B1*J,1).repeat(1,M).view(-1,1)
                    X1_rep  = X1.unsqueeze(1).unsqueeze(2).expand(B1,J,M,cfg.x_dim).reshape(B1*J*M,cfg.x_dim)
                    eps1    = torch.randn(B1*J*M, EPS_DIM, device=DEVICE)*EPS_LAMBDA
                    U1 = h1(Zs1_rep, X1_rep, eps1)
                    p1_U1 = g1(U1, X1_rep)
                    invU1_avg = p1_U1.view(B1,J*M,1).mean(1)
                    invU_avg[mask1] = invU1_avg

                val_loss = ((torch.log(invW+1e-8) - torch.log(invU_avg+1e-8))**2).mean()
                avg_val += val_loss.item()
        avg_val /= len(val_loader)

        print(f"[Treatment Bridge] Epoch {epoch:03d}, TrainLoss={train_loss/len(train_loader):.4f}, ValLoss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            best_models = {"h0":h0,"h1":h1,"g0":g0,"g1":g1}
            os.makedirs(ckpt_dir, exist_ok=True)
            for name, model in best_models.items():
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{name}_{RUN_NAME}.pth"))
            print(f"  → Saved best treatment bridge (Val Loss={avg_val:.4f})")

    return best_models

