# framingham_evaluate.py
import os
import torch
import numpy as np
from torch.distributions import Normal
from framingham_config import *
from framingham_models import NN_W, NN_TW, h_net, g_net

# -------------------------------------------------------
# Ground truth ratio (data-specific, not simulated)
# -------------------------------------------------------
TRUE_RATIO = 0.75   # approx, from domain knowledge
TRUE_ATE   = None   # unknown

# -------------------------------------------------------
# Evaluation function
# -------------------------------------------------------
def evaluate_models(seed, cfg, W_t, Z_t, A_t, Y_t,
                    outcome_models, treatment_models,
                    results_dir="results"):

    EPS_DIM = cfg.eps_dim
    EPS_LAMBDA = cfg.eps_lambda
    DEVICE = cfg.device

    # --------------------
    # Unpack trained models
    # --------------------
    nn_w0, nn_w1, nn_tw0, nn_tw1 = (
        outcome_models["nn_w0"],
        outcome_models["nn_w1"],
        outcome_models["nn_tw0"],
        outcome_models["nn_tw1"],
    )
    h0, h1, g0, g1 = (
        treatment_models["h0"],
        treatment_models["h1"],
        treatment_models["g0"],
        treatment_models["g1"],
    )

    for m in [nn_w0, nn_w1, nn_tw0, nn_tw1, h0, h1, g0, g1]:
        m.eval()

    N = W_t.shape[0]

    # --------------------
    # Naive
    # --------------------
    EY0_naive = Y_t[A_t.squeeze() == 0].mean().item()
    EY1_naive = Y_t[A_t.squeeze() == 1].mean().item()
    Ratio_naive = EY1_naive / (EY0_naive + 1e-8)
    ATE_naive = EY1_naive - EY0_naive

    # --------------------
    # Outcome bridge
    # --------------------
    M = 500
    eps_all = torch.randn(N * M, EPS_DIM, device=DEVICE) * EPS_LAMBDA
    W_rep = W_t.unsqueeze(1).repeat(1, M, 1).view(-1, 16)

    with torch.no_grad():
        U0 = nn_w0(W_rep, eps_all)
        Y0_hat = nn_tw0(U0, W_rep).view(N, M, 1).mean(1)

        U1 = nn_w1(W_rep, eps_all)
        Y1_hat = nn_tw1(U1, W_rep).view(N, M, 1).mean(1)

        E_Y_do0 = Y0_hat.mean().item()
        E_Y_do1 = Y1_hat.mean().item()
        Ratio_outcome = E_Y_do1 / (E_Y_do0 + 1e-8)
        ATE_outcome   = E_Y_do1 - E_Y_do0

    # --------------------
    # Treatment bridge
    # --------------------
    def compute_q(Z_grp, A_val, model_h, model_g):
        n = Z_grp.size(0)
        Z_rep = Z_grp.unsqueeze(1).repeat(1, M, 1).view(-1, 16)
        eps = torch.randn(n * M, EPS_DIM, device=DEVICE) * EPS_LAMBDA
        U = model_h(Z_rep, eps)
        pA = model_g(U)
        q = pA.view(n, M, 1).mean(1)
        return q

    mask0 = (A_t.squeeze() == 0)
    mask1 = (A_t.squeeze() == 1)
    Y0, Z0 = Y_t[mask0], Z_t[mask0]
    Y1, Z1 = Y_t[mask1], Z_t[mask1]

    q0 = compute_q(Z0, 0, h0, g0)
    q1 = compute_q(Z1, 1, h1, g1)

    EY_do0_tr = (Y0 * q0).sum() / N
    EY_do1_tr = (Y1 * q1).sum() / N
    Ratio_treatment = (EY_do1_tr / (EY_do0_tr + 1e-8)).item()
    ATE_treatment   = (EY_do1_tr - EY_do0_tr).item()

    # --------------------
    # Double robust
    # --------------------
    with torch.no_grad():
        q_full = torch.zeros(N, 1, device=DEVICE)
        q_full[mask1] = q1
        q_full[mask0] = q0

        h_W0 = Y0_hat
        h_W1 = Y1_hat

        EY0_DR = ( ((Y_t - h_W0) * q_full * (A_t == 0).float()).sum() / N
                 + h_W0.mean() )
        EY1_DR = ( ((Y_t - h_W1) * q_full * (A_t == 1).float()).sum() / N
                 + h_W1.mean() )

        Ratio_DR = (EY1_DR / (EY0_DR + 1e-8)).item()
        ATE_DR   = (EY1_DR - EY0_DR).item()

    # --------------------
    # Save results
    # --------------------
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"results.txt"), "a") as f:
        f.write(f"Naive Ratio: {Ratio_naive:.6f}\n")
        f.write(f"Outcome Bridge Ratio: {Ratio_outcome:.6f}\n")
        f.write(f"Treatment Bridge Ratio: {Ratio_treatment:.6f}\n")
        f.write(f"Double Robust Ratio: {Ratio_DR:.6f}\n")
        f.write(f"True Ratio: {TRUE_RATIO:.6f}\n")
        f.write(f"Naive ATE: {ATE_naive:.6f}\n")
        f.write(f"Outcome Bridge ATE: {ATE_outcome:.6f}\n")
        f.write(f"Treatment Bridge ATE: {ATE_treatment:.6f}\n")
        f.write(f"Double Robust ATE: {ATE_DR:.6f}\n")
        f.write(f"True ATE: \n\n")  # unknown

    print(f"[Evaluation] Ratio → Naive={Ratio_naive:.4f}, Outcome={Ratio_outcome:.4f}, "
          f"Treatment={Ratio_treatment:.4f}, DR={Ratio_DR:.4f}, True={TRUE_RATIO:.4f}")
    print(f"[Evaluation] ATE   → Naive={ATE_naive:.4f}, Outcome={ATE_outcome:.4f}, "
          f"Treatment={ATE_treatment:.4f}, DR={ATE_DR:.4f}, True=?")
