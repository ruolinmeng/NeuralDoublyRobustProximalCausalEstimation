import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def kde_entropy(u_samples, sigma=1.0, max_samples=1000):
    """
    KDE entropy estimator using a random subsample of u_samples.
    
    Args:
        u_samples: Tensor of shape [N, u_dim]
        sigma: Bandwidth for the Gaussian kernel
        max_samples: Max number of U samples to use (for memory reasons)
        
    Returns:
        Approximate entropy estimate over the subset
    """
    N = u_samples.shape[0]

    if N > max_samples:
        indices = torch.randperm(N, device=u_samples.device)[:max_samples]
        u_samples = u_samples[indices]
        N = max_samples

    # Efficient squared pairwise distance computation
    norms = torch.norm(u_samples, dim=1)**2
    dists = norms[:, None] + norms[None, :] - 2 * u_samples @ u_samples.T
    dists = dists.clamp_min(0)

    # Remove diagonal
    dists = dists[~torch.eye(N, dtype=torch.bool, device=u_samples.device)].view(N, N - 1)

    kernel = torch.exp(-dists / (2 * sigma**2))
    inner = kernel.sum(dim=1) / (N - 1)

    return -torch.mean(torch.log(inner + 1e-8))

