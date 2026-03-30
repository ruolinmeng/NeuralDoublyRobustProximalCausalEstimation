import numpy as np
import pandas as pd
import torch

# Fixed parameters (from your notebook)
Gamma_x = np.array([0.25, 0.25])              # shape (2,)
sigma_x = 0.25
Sigma_x = np.diag([sigma_x**2, sigma_x**2])   # 2×2 covariance

# Propensity: Pr(A=1 | X) = sigmoid( (0.125, 0.125) · X )
alpha_logistic = np.array([0.125, 0.125])     # shape (2,)

# Conditional means
alpha_0  = 0.25;  alpha_a  = 0.25;  alpha_x  = np.array([0.25, 0.25])
mu_0     = 0.25;  mu_a     = 0.125; mu_x     = np.array([0.25, 0.25])
kappa_0  = 0.25;  kappa_a  = 0.25;  kappa_x  = np.array([0.25, 0.25])

# Covariance Σ_{ZWU}
Sigma_ZWU = np.array([
    [1.00, 0.25, 0.50],
    [0.25, 1.00, 0.50],
    [0.50, 0.50, 1.00]
])

# Outcome coefficients
b_0     = 2.0
b_a     = 2.0
b_x     = np.array([0.25, 0.25])
b_w     = 4.0
omega   = 2.0
sigma_y = 0.25   # std of ε_Y

# Extract covariance pieces
sigma_WU = Sigma_ZWU[1, 2]
sigma_U2 = Sigma_ZWU[2, 2]


def generate_sem_data(n):
    """
    Generate SEM-style data with covariates X, treatment A, 
    mediators (Z, W, U), and outcome Y.
    Returns a pandas DataFrame with all variables.
    """
    # Step 1: X ~ N(Gamma_x, Sigma_x)
    X = np.random.multivariate_normal(mean=Gamma_x, cov=Sigma_x, size=n)  # (n,2)

    # Step 2: A ~ Bernoulli(sigmoid(alpha_logistic^T X))
    logits = X.dot(alpha_logistic)          # (n,)
    pA     = 1.0 / (1.0 + np.exp(-logits))
    A      = np.random.binomial(1, pA)      # (n,)

    # Step 3: Generate (Z,W,U) ~ MVN(mean_ZWU, Sigma_ZWU)
    mean_Z = alpha_0 + alpha_a * A + X.dot(alpha_x)
    mean_W = mu_0 + mu_a * A + X.dot(mu_x)
    mean_U = kappa_0 + kappa_a * A + X.dot(kappa_x)
    mean_ZWU = np.stack([mean_Z, mean_W, mean_U], axis=1)

    ZWU = np.array([
        np.random.multivariate_normal(mean=mean_ZWU[i], cov=Sigma_ZWU)
        for i in range(n)
    ])
    Z, W, U = ZWU[:, 0], ZWU[:, 1], ZWU[:, 2]

    # Step 4: E[W | U,X]
    base_W_mean = mu_0 + X.dot(mu_x)
    correction  = (sigma_WU / sigma_U2) * (U - (kappa_0 + X.dot(kappa_x)))
    EW_given_UX = base_W_mean + correction

    # Step 5: Y
    noise_Y = np.random.normal(loc=0.0, scale=sigma_y, size=n)
    Y = (
        b_0
        + b_a * A
        + X.dot(b_x)
        + b_w * EW_given_UX
        + omega * (W - EW_given_UX)
        + noise_Y
    )

    # Build DataFrame
    df = pd.DataFrame({
        "X0": X[:, 0],
        "X1": X[:, 1],
        "A": A,
        "Z": Z,
        "W": W,
        "U": U,   # latent (useful if you want to inspect but not fed to learners)
        "Y": Y,
    })
    return df


def make_tensors(df, device):
    """
    Convert SEM DataFrame into torch tensors.
    Returns: W, Z, A, Y, X
    """
    W_t = torch.tensor(df["W"].to_numpy(), dtype=torch.float32, device=device).unsqueeze(1)
    Z_t = torch.tensor(df["Z"].to_numpy(), dtype=torch.float32, device=device).unsqueeze(1)
    A_t = torch.tensor(df["A"].to_numpy(), dtype=torch.float32, device=device).unsqueeze(1)
    Y_t = torch.tensor(df["Y"].to_numpy(), dtype=torch.float32, device=device).unsqueeze(1)
    X_t = torch.tensor(df[["X0","X1"]].to_numpy(), dtype=torch.float32, device=device)

    return W_t, Z_t, A_t, Y_t, X_t
