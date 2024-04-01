import torch
import torch.nn as nn

def make_blobs(N, H, W, iters=1024, kernel_size=3, alpha=0.5, q=0.9, device="cpu"):
    z = torch.randn(N, 1, H, W).to(device)
    a2d = nn.AvgPool2d(kernel_size, 1, kernel_size // 2).to(device)
    for _ in range(iters):
        z = alpha*z + (1-alpha) * a2d(z)
    mask = (z > z.view(N,-1).quantile(q, dim=1).view(N,1,1,1)).long() # (N,1,H,W)
    return mask


