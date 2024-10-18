import torch
import numpy as np
import math

__all__ = ['generate_overlap_image', 'generate_side_by_side_image']
           
def generate_overlap_image(imgA, imgB, pct_imgA=.5):
    return pct_imgA * imgA + (1-pct_imgA) * imgB

def generate_side_by_side_image(imgA, imgB):
    return torch.cat([imgA,imgB], dim=-1)

def generate_mixcut_image(imgA, imgB, gamma=.5):
    return gamma * imgA + (1-gamma) * imgB

# Function to apply CutMix operation
def generate_cutmix_image(imgA, imgB, box):
    x1, y1, x2, y2 = box
    output = imgA.clone()
    output[..., y1:y2, x1:x2] = imgB[..., y1:y2, x1:x2]
    return output

def sample_beta(alpha, beta, size, seed=42):
    """Sample values from a Beta distribution with given alpha and beta parameters."""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Sample from Beta distribution
    dist = torch.distributions.beta.Beta(alpha, beta)
    samples = dist.sample((size,))

    return samples

# Function to generate CutMix parameters
def generate_cutmix_params(alpha, H, W, lam_range=(0,1)):
    min_lam, max_lam = lam_range
    dist = torch.distributions.beta.Beta(alpha, alpha)

    lam = float(dist.sample(()))
    lam = lam * (max_lam-min_lam) + min_lam

    r_x = torch.randint(W, size=(1,))
    r_y = torch.randint(H, size=(1,))
    r = 0.5 * math.sqrt(1.0 - lam)

    r_w_half = int(r * W)
    r_h_half = int(r * H)

    x1 = int(torch.clamp(r_x - r_w_half, min=0))
    y1 = int(torch.clamp(r_y - r_h_half, min=0))
    x2 = int(torch.clamp(r_x + r_w_half, max=W))
    y2 = int(torch.clamp(r_y + r_h_half, max=H))

    box = (x1, y1, x2, y2)
    lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

    return dict(box=box, lam_adjusted=lam_adjusted)

# Function to pre-compute CutMix parameters for N pairs of images
def precompute_cutmix_params(N, alpha, H, W, seed=42):
    torch.manual_seed(seed)
    params_list = [generate_cutmix_params(alpha, H, W) for _ in range(N)]
    return params_list