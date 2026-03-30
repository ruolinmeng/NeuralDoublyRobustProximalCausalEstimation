import torch
import torch.nn as nn

# -------------------------
# Generators (keep a bit deeper, since they model distributions)
# -------------------------
class Z_given_WA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(17, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU()
        )
        self.mu_net = nn.Linear(64, 16)
        self.std_net = nn.Sequential(nn.Linear(64, 16), nn.Softplus())

    def forward(self, cond):
        h = self.base(cond)
        return self.mu_net(h), self.std_net(h)

class W_given_ZA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(17, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU()
        )
        self.mean = nn.Linear(64, 16)
        self.std  = nn.Sequential(nn.Linear(64, 16), nn.Softplus())

    def forward(self, cond):
        h = self.base(cond)
        return self.mean(h), self.std(h)

class A_given_W(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(16, 1)
        )
        self.sig = nn.Sigmoid()

    def forward(self, w):
        return self.sig(self.base(w))


# -------------------------
# Outcome Bridge (linearized)
# -------------------------

class NN_W(nn.Module):
    def __init__(self, u_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(17, u_dim)
        )
    def forward(self, w, eps):
        return self.fc(torch.cat([w, eps], dim=1))

class NN_TW(nn.Module):
    def __init__(self, u_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(u_dim + 16, 1), nn.Sigmoid()
        )
    def forward(self, u, w):
        return self.fc(torch.cat([u, w], dim=1))


class NN_ZW(nn.Module):
    def __init__(self, u_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(16 + u_dim, 1), nn.Sigmoid()
        )

    def forward(self, z, u):
        return self.fc(torch.cat([z, u], dim=1))


class NN_M(nn.Module):
    def __init__(self, u_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(u_dim, 16)
        )

    def forward(self, u):
        return self.fc(u)


# -------------------------
# Treatment Bridge (linearized)
# -------------------------
class h_net(nn.Module):
    def __init__(self, eps_dim=1, u_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(16 + eps_dim, u_dim)
        )

    def forward(self, z, eps):  # z: [B,16], eps: [B,eps_dim]
        return self.fc(torch.cat([z, eps], dim=1))


class g_net(nn.Module):
    def __init__(self, u_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(u_dim, 1),
            nn.Softplus()   # ensure positivity
        )

    def forward(self, u):
        return self.fc(u)


class w_net(nn.Module):
    def __init__(self, u_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(u_dim, 16)
        )

    def forward(self, u):
        return self.fc(u)


class t_net(nn.Module):
    def __init__(self, u_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(u_dim + 16, 1),
            nn.Sigmoid()
        )

    def forward(self, u, z):
        # return self.fc(u)
        return self.fc(torch.cat([u, z], dim=1))

