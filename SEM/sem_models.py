import torch
import torch.nn as nn

# --------------------------
# Generators
# --------------------------

class Z_given_WAX(nn.Module):
    """Model p(Z | W, A, X)"""
    def __init__(self, x_dim=2):
        super().__init__()
        in_dim = 1 + 1 + x_dim  # W (1) + A (1) + X (x_dim)
        self.base = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.mu_net  = nn.Linear(64, 1)
        self.std_net = nn.Sequential(nn.Linear(64, 1), nn.Softplus())

    def forward(self, w, a, x):
        cond = torch.cat([w, a, x], dim=1)
        h = self.base(cond)
        return self.mu_net(h), self.std_net(h)


class W_given_ZAX(nn.Module):
    """Model p(W | Z, A, X)"""
    def __init__(self, x_dim=2):
        super().__init__()
        in_dim = 1 + 1 + x_dim  # Z (1) + A (1) + X (x_dim)
        self.base = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.mean = nn.Linear(64, 1)
        self.std  = nn.Sequential(nn.Linear(64, 1), nn.Softplus())

    def forward(self, z, a, x):
        cond = torch.cat([z, a, x], dim=1)
        h = self.base(cond)
        return self.mean(h), self.std(h)


class A_given_WX(nn.Module):
    """Propensity model p(A | W, X)"""
    def __init__(self, x_dim=2):
        super().__init__()
        in_dim = 1 + x_dim
        self.base = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.sig = nn.Sigmoid()

    def forward(self, w, x):
        cond = torch.cat([w, x], dim=1)
        return self.sig(self.base(cond))

# --------------------------
# Outcome bridge
# --------------------------

class NN_WX(nn.Module):
    """Bridge NN for U = f(W, X, eps)"""
    def __init__(self, x_dim=2):
        super().__init__()
        in_dim = 1 + x_dim + 1  # W (1) + X (x_dim) + eps (1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, w, x, eps):
        return self.fc(torch.cat([w, x, eps], dim=1))


class NN_TWX(nn.Module):
    """Bridge NN for Y_hat = f(U, W, X)"""
    def __init__(self, x_dim=2):
        super().__init__()
        in_dim = 1 + 1 + x_dim  # U (1) + W (1) + X (x_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, u, w, x):
        return self.fc(torch.cat([u, w, x], dim=1))

# --------------------------
# Treatment bridge helpers
# --------------------------

class h_net(nn.Module):
    """Treatment bridge encoder U = f(Z, X, eps)"""
    def __init__(self, x_dim=2, eps_dim=1, u_dim=1):
        super().__init__()
        in_dim = 1 + x_dim + eps_dim  # Z (1) + X (x_dim) + eps
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, u_dim)
        )

    def forward(self, z, x, eps):
        return self.fc(torch.cat([z, x, eps], dim=1))


class g_net(nn.Module):
    """Bridge weighting function g(U, X) > 0"""
    def __init__(self, u_dim=1, x_dim=2):
        super().__init__()
        in_dim = u_dim + x_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Softplus()
        )

    def forward(self, u, x):
        return self.fc(torch.cat([u, x], dim=1))


class w_net(nn.Module):
    """Decoder W_hat = f(U, X)"""
    def __init__(self, u_dim=1, x_dim=2):
        super().__init__()
        in_dim = u_dim + x_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, u, x):
        return self.fc(torch.cat([u, x], dim=1))


class t_net(nn.Module):
    """Predict treatment A_hat from W and X"""
    def __init__(self, x_dim=2):
        super().__init__()
        in_dim = 1 + x_dim + 1
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, w, x, z):
        return self.fc(torch.cat([w, x, z], dim=1))
    

class NN_ZW(nn.Module):
    def __init__(self):
        super(NN_ZW, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z, w, x):
        x = torch.cat((z, w, x), dim=1)
        return self.fc(x)
    
class NN_M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, u, x):
        x = torch.cat((u, x), dim=1)
        return self.fc(x)
