import os
import torch
import numpy as np
from torch.distributions import Normal
from sem_config import *
from sem_models import NN_WX, NN_TWX, h_net, g_net

# -------------------------------------------------------
# For SEM data the true ATE is known from the DGP
# -------------------------------------------------------
TRUE_ATE = 2.0

# -------------------------------------------------------
# Evaluation function
# -------------------------------------------------------
def evaluate_models(seed, cfg, W_t, Z_t, A_t, Y_t, X_t,
                    outcome_models, treatment_models,
                    results_dir="results"):

    EPS_DIM = cfg.eps_dim
    EPS_LAMBDA = cfg.eps_lambda
    DEVICE = cfg.device

    # --------------------
    # Unpack trained models
    # --------------------
    nn_w0, nn_w1, nn_tw0, nn_tw1 = outcome_models["nn_w0"], outcome_models["nn_w1"], outcome_models["nn_tw0"], outcome_models["nn_tw1"]
    h0, h1, g0, g1 = treatment_models["h0"], treatment_models["h1"], treatment_models["g0"], treatment_models["g1"]

    for m in [nn_w0, nn_w1, nn_tw0, nn_tw1, h0, h1, g0, g1]:
        m.eval()

    N = W_t.shape[0]

    # --------------------
    # Naive ATE
    # --------------------
    EY0_naive = Y_t[A_t.squeeze() == 0].mean().item()
    EY1_naive = Y_t[A_t.squeeze() == 1].mean().item()
    ATE_naive = EY1_naive - EY0_naive

    # --------------------
    # Outcome bridge ATE
    # --------------------
    M = 500
    eps_all = torch.randn(N * M, EPS_DIM, device=DEVICE) * EPS_LAMBDA
    W_rep = W_t.unsqueeze(1).repeat(1, M, 1).view(-1, 1)
    X_rep = X_t.unsqueeze(1).repeat(1, M, 1).view(-1, cfg.x_dim)

    with torch.no_grad():
        U0 = nn_w0(W_rep, X_rep, eps_all)
        Y0_hat = nn_tw0(U0, W_rep, X_rep).view(N, M, 1).mean(1)

        U1 = nn_w1(W_rep, X_rep, eps_all)
        Y1_hat = nn_tw1(U1, W_rep, X_rep).view(N, M, 1).mean(1)

        E_Y_do0 = Y0_hat.mean().item()
        E_Y_do1 = Y1_hat.mean().item()
        ATE_outcome = E_Y_do1 - E_Y_do0

    # --------------------
    # Treatment bridge ATE
    # --------------------
    def compute_q(Z_grp, X_grp, model_h, model_g):
        n = Z_grp.size(0)
        Z_rep = Z_grp.unsqueeze(1).repeat(1, M, 1).view(-1, 1)
        X_rep = X_grp.unsqueeze(1).repeat(1, M, 1).view(-1, cfg.x_dim)
        eps = torch.randn(n * M, EPS_DIM, device=DEVICE) * EPS_LAMBDA
        U = model_h(Z_rep, X_rep, eps)
        pA = model_g(U, X_rep)
        return pA.view(n, M, 1).mean(1)

    mask0 = (A_t.squeeze() == 0)
    mask1 = (A_t.squeeze() == 1)
    Y0, Z0, X0 = Y_t[mask0], Z_t[mask0], X_t[mask0]
    Y1, Z1, X1 = Y_t[mask1], Z_t[mask1], X_t[mask1]

    q0 = compute_q(Z0, X0, h0, g0)
    q1 = compute_q(Z1, X1, h1, g1)

    EY_do0_tr = (Y0 * q0).sum() / N
    EY_do1_tr = (Y1 * q1).sum() / N
    ATE_treatment = (EY_do1_tr - EY_do0_tr).item()

    # --------------------
    # Double robust ATE
    # --------------------
    with torch.no_grad():
        h_W0 = Y0_hat
        h_W1 = Y1_hat
        h_diff = h_W1 - h_W0

        q_full = torch.zeros(N, 1, device=DEVICE)
        q_full[mask1] = q1
        q_full[mask0] = q0

        h_WA = torch.zeros(N, 1, device=DEVICE)
        h_WA[mask1] = h_W1[mask1]
        h_WA[mask0] = h_W0[mask0]

        sign = (-1.0) ** (1 - A_t.squeeze())
        sign = sign.view(N, 1)

        tau_i = sign * q_full * (Y_t - h_WA) + h_diff
        ATE_DR = tau_i.mean().item()

    # --------------------
    # Save results
    # --------------------
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"results.txt"), "a") as f:
        f.write(f"Naive ATE: {ATE_naive:.6f}\n")
        f.write(f"Outcome Bridge ATE: {ATE_outcome:.6f}\n")
        f.write(f"Treatment Bridge ATE: {ATE_treatment:.6f}\n")
        f.write(f"Double Robust ATE: {ATE_DR:.6f}\n")
        f.write(f"True ATE: {TRUE_ATE:.6f}\n\n")

    print(f"[Evaluation] Naive={ATE_naive:.4f}, Outcome={ATE_outcome:.4f}, "
          f"Treatment={ATE_treatment:.4f}, DR={ATE_DR:.4f}, True={TRUE_ATE:.4f}")
