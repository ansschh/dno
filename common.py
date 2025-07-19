"""
common.py
=========
Shared utilities for Delay Neural Operator experiments.

Dependencies: torch, numpy, tqdm, pickle, math, os, argparse, random

(Install with: pip install torch numpy tqdm)
"""

import os, math, pickle, random, argparse, time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------
# 1. Reproducibility Helpers
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------
# 2. Dataset Abstractions
# ---------------------------
class DDEDataset(Dataset):
    """
    Generic DDE dataset reading a pickle of list[(hist, tau, t, y)].
    Provides interpolation & windowing options as needed by variants.

    Each __getitem__ returns a dict with keys:
       'hist' : torch.FloatTensor
       'tau'  : torch.FloatTensor scalar
       't'    : torch.FloatTensor (time points for solution)
       'y'    : torch.FloatTensor (solution values)
       'family': str
    """
    def __init__(self,
                 pkl_path: str,
                 family: Optional[str] = None,
                 force_min_len: Optional[int] = None):
        super().__init__()
        assert os.path.isfile(pkl_path), f"Missing dataset file {pkl_path}"
        with open(pkl_path, "rb") as f:
            self.samples = pickle.load(f)
        self.family = family or "unknown"
        # Optional filter: remove extremely short trajectories
        if force_min_len is not None:
            self.samples = [s for s in self.samples if len(s[2]) >= force_min_len]
        if len(self.samples) == 0:
            raise ValueError("No samples after filtering.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        hist, tau, t, y = self.samples[idx]
        # Ensure numpy arrays
        hist = np.asarray(hist)
        t    = np.asarray(t)
        y    = np.asarray(y)
        # Force last dim = features
        if hist.ndim == 1:
            hist = hist[:, None]
        if y.ndim == 1:
            y = y[:, None]
        return {
            "hist": torch.from_numpy(hist).float(),  # shape (H_hist, D_out)
            "tau": torch.tensor([float(tau)], dtype=torch.float32),
            "t": torch.from_numpy(t).float(),        # (T_len,)
            "y": torch.from_numpy(y).float(),        # (T_len, D_out or 20)
            "family": self.family
        }

def collate_pad(batch: List[Dict[str, Any]]):
    """
    Pads variable-length hist and y sequences to max length in batch.
    Returns masks so losses can ignore padding.
    """
    # Histories
    h_lens = [b["hist"].shape[0] for b in batch]
    y_lens = [b["y"].shape[0] for b in batch]
    H_max  = max(h_lens)
    Y_max  = max(y_lens)
    feat_h = batch[0]["hist"].shape[1]
    feat_y = batch[0]["y"].shape[1]

    hist_pad = torch.zeros(len(batch), H_max, feat_h)
    hist_mask = torch.zeros(len(batch), H_max, dtype=torch.bool)
    y_pad = torch.zeros(len(batch), Y_max, feat_y)
    y_mask = torch.zeros(len(batch), Y_max, dtype=torch.bool)
    tau = torch.zeros(len(batch), 1)
    families = []
    for i,b in enumerate(batch):
        hL, yL = h_lens[i], y_lens[i]
        hist_pad[i,:hL] = b["hist"]
        y_pad[i,:yL]    = b["y"]
        hist_mask[i,:hL] = 1
        y_mask[i,:yL]    = 1
        tau[i] = b["tau"]
        families.append(b["family"])
    return {
        "hist": hist_pad, "hist_mask": hist_mask,
        "y": y_pad, "y_mask": y_mask,
        "tau": tau,
        "families": families
    }

# ---------------------------
# 3. Basic Metrics
# ---------------------------
def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """
    pred, target: (B, T, C) or (B, C, S, X, ...)
    mask: (B, T) -> broadcast over feature dims
    """
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    diff = (pred - target)**2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom

def relative_l2(pred: torch.Tensor, target: torch.Tensor, mask=None):
    if mask is None:
        return (torch.norm(pred-target) / torch.norm(target)).item()
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    num = torch.sum(((pred-target)**2) * mask)
    den = torch.sum((target**2) * mask).clamp_min(1e-12)
    return math.sqrt((num/den).item())

# ---------------------------
# 4. Spectral Convolution (N-D) (lightweight)
# ---------------------------
class SpectralConvND(nn.Module):
    """
    Truncated complex-valued spectral convolution.
    Works for 1D (S), 2D (S,X), or 3D (S,X,Y) if needed.
    Expect input shape: (B, C, D1, D2, ...)
    """
    def __init__(self, in_channels, out_channels, n_modes: Tuple[int, ...]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dims = len(n_modes)
        self.n_modes = n_modes
        # Parameter: real & imag weights for each kept mode block
        self.scale = 1 / (in_channels * out_channels)
        # store as single complex tensor using two real params
        self.weight_real = nn.Parameter(torch.randn(out_channels, in_channels, *n_modes) * self.scale)
        self.weight_imag = nn.Parameter(torch.randn(out_channels, in_channels, *n_modes) * self.scale)

    def compl_weight(self):
        return self.weight_real + 1j * self.weight_imag

    def forward(self, x):
        # x shape (B, C, D1, D2, ...)
        dims = list(range(-self.n_dims, 0))
        x_ft = torch.fft.fftn(x, dim=dims)
        W = self.compl_weight()  # (O, I, m1, m2, ...)
        
        # Ensure n_modes doesn't exceed input dimensions
        actual_modes = []
        for i, mode in enumerate(self.n_modes):
            actual_size = x.shape[2 + i]  # Skip batch and channel dimensions
            actual_modes.append(min(mode, actual_size))
        
        # Slice each dimension to actual modes
        slices = tuple(slice(0, m) for m in actual_modes)
        # Use indexing without unpacking for Python 3.4 compatibility
        if len(slices) == 1:
            x_ft_trunc = x_ft[:, None, :, slices[0]]   # (B,1,C, m1)
        elif len(slices) == 2:
            x_ft_trunc = x_ft[:, None, :, slices[0], slices[1]]   # (B,1,C, m1,m2)
        elif len(slices) == 3:
            x_ft_trunc = x_ft[:, None, :, slices[0], slices[1], slices[2]]   # (B,1,C, m1,m2,m3)
        else:
            raise ValueError(f"Unsupported number of dimensions: {len(slices)}")
            
        # Truncate weight tensor to match actual modes if needed
        W_slices = tuple(slice(0, m) for m in actual_modes)
        if len(W_slices) == 1:
            W_trunc = W[:, :, W_slices[0]]
        elif len(W_slices) == 2:
            W_trunc = W[:, :, W_slices[0], W_slices[1]]
        elif len(W_slices) == 3:
            W_trunc = W[:, :, W_slices[0], W_slices[1], W_slices[2]]
        else:
            raise ValueError(f"Unsupported number of dimensions: {len(W_slices)}")
            
        out_ft_trunc = torch.einsum("boi..., oi... -> bo...", x_ft_trunc, W_trunc)
        # Build a zero-filled tensor same size as x_ft (for inverse FFT)
        out_ft = torch.zeros((x.size(0), self.out_channels, *x.shape[2:]),
                             device=x.device, dtype=torch.cfloat)
        # Use indexing without unpacking for Python 3.4 compatibility
        out_slices = tuple(slice(0, m) for m in actual_modes)
        if len(out_slices) == 1:
            out_ft[:, :, out_slices[0]] = out_ft_trunc
        elif len(out_slices) == 2:
            out_ft[:, :, out_slices[0], out_slices[1]] = out_ft_trunc
        elif len(out_slices) == 3:
            out_ft[:, :, out_slices[0], out_slices[1], out_slices[2]] = out_ft_trunc
        else:
            raise ValueError(f"Unsupported number of dimensions: {len(out_slices)}")
        x_out = torch.fft.ifftn(out_ft, dim=dims).real
        return x_out

# ---------------------------
# 5. Simple Feed-Forward Blocks
# ---------------------------
class ChannelMLP(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, act=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden),
            act(),
            nn.Linear(hidden, out_ch)
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------
# 6. Checkpointing
# ---------------------------
def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "best_metric": best_metric
    }, path)

def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", float("inf"))

# ---------------------------
# 7. Timer Context
# ---------------------------
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.seconds = time.time() - self.start
