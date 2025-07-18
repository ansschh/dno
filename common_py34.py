"""
Common utilities for Delay Neural Operator models.
"""
import os, math, pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import time

# ---------------------------
# 1. Dataset and Data Loading
# ---------------------------

class DDEDataset(Dataset):
    """Dataset for delay differential equations."""
    
    def __init__(self, data_path, max_samples=None):
        """Load dataset from pickle file."""
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
        self.samples = data["samples"]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        # Extract dimensions
        sample = self.samples[0]
        self.hist_dim = sample["hist"].shape[-1] if len(sample["hist"].shape) > 1 else 1
        self.out_dim = sample["y"].shape[-1] if len(sample["y"].shape) > 2 else 1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        hist = torch.tensor(sample["hist"], dtype=torch.float32)
        tau = torch.tensor(sample["tau"], dtype=torch.float32)
        t = torch.tensor(sample["t"], dtype=torch.float32)
        y = torch.tensor(sample["y"], dtype=torch.float32)
        
        # Ensure hist has shape (T, C) and y has shape (T, C)
        if len(hist.shape) == 1:
            hist = hist.unsqueeze(-1)
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)
        
        return {"hist": hist, "tau": tau, "t": t, "y": y}


def collate_pad(batch):
    """Collate function that pads sequences to the same length."""
    # Find max lengths
    max_hist_len = max(x["hist"].shape[0] for x in batch)
    max_y_len = max(x["y"].shape[0] for x in batch)
    
    # Get dimensions
    hist_dim = batch[0]["hist"].shape[1]
    y_dim = batch[0]["y"].shape[1]
    
    # Prepare output tensors
    hist_padded = torch.zeros(len(batch), max_hist_len, hist_dim)
    y_padded = torch.zeros(len(batch), max_y_len, y_dim)
    tau = torch.zeros(len(batch))
    
    # Fill tensors
    for i, sample in enumerate(batch):
        h, y = sample["hist"], sample["y"]
        hist_padded[i, :h.shape[0], :] = h
        y_padded[i, :y.shape[0], :] = y
        tau[i] = sample["tau"]
    
    return {"hist": hist_padded, "tau": tau, "y": y_padded}


# ---------------------------
# 2. Neural Network Components
# ---------------------------

class SpectralConv(nn.Module):
    """Spectral convolution layer using FFT."""
    
    def __init__(self, in_channels, out_channels, n_modes, n_dims=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes if isinstance(n_modes, (list, tuple)) else [n_modes] * n_dims
        self.n_dims = n_dims
        
        # Complex weights for each mode
        shape = [out_channels, in_channels] + self.n_modes
        real = torch.FloatTensor(*shape).uniform_(-0.01, 0.01)
        imag = torch.FloatTensor(*shape).uniform_(-0.01, 0.01)
        self.weights_real = nn.Parameter(real)
        self.weights_imag = nn.Parameter(imag)
    
    def compl_weight(self):
        return torch.complex(self.weights_real, self.weights_imag)
    
    def forward(self, x):
        # x shape (B, C, D1, D2, ...)
        dims = list(range(-self.n_dims, 0))
        x_ft = torch.fft.fftn(x, dim=dims)
        W = self.compl_weight()  # (O, I, m1, m2, ...)
        # Slice each dimension to n_modes
        slices = tuple(slice(0, m) for m in self.n_modes)
        
        # Use indexing without unpacking for Python 3.4 compatibility
        if len(slices) == 1:
            x_ft_trunc = x_ft[:, None, :, slices[0]]   # (B,1,C, m1)
        elif len(slices) == 2:
            x_ft_trunc = x_ft[:, None, :, slices[0], slices[1]]   # (B,1,C, m1,m2)
        elif len(slices) == 3:
            x_ft_trunc = x_ft[:, None, :, slices[0], slices[1], slices[2]]   # (B,1,C, m1,m2,m3)
        else:
            raise ValueError("Unsupported number of dimensions: {}".format(len(slices)))
            
        out_ft_trunc = torch.einsum("boi..., o i ... -> bo...", x_ft_trunc, W)
        
        # Build a zero-filled tensor same size as x_ft (for inverse FFT)
        out_shape = list(x.shape)
        out_shape[1] = self.out_channels
        out_ft = torch.zeros(out_shape, device=x.device, dtype=torch.cfloat)
        
        # Use indexing without unpacking for Python 3.4 compatibility
        if len(slices) == 1:
            out_ft[:, :, slices[0]] = out_ft_trunc
        elif len(slices) == 2:
            out_ft[:, :, slices[0], slices[1]] = out_ft_trunc
        elif len(slices) == 3:
            out_ft[:, :, slices[0], slices[1], slices[2]] = out_ft_trunc
        else:
            raise ValueError("Unsupported number of dimensions: {}".format(len(slices)))
            
        x_out = torch.fft.ifftn(out_ft, dim=dims).real
        return x_out


class ChannelMLP(nn.Module):
    """MLP applied to channel dimension."""
    
    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = 2 * max(in_channels, out_channels)
        
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, x):
        # x: (B, C, ...)
        shape = x.shape
        x_flat = x.reshape(shape[0], shape[1], -1)  # (B, C, N)
        x_flat = x_flat.permute(0, 2, 1)  # (B, N, C)
        y_flat = self.net(x_flat)  # (B, N, C_out)
        y_flat = y_flat.permute(0, 2, 1)  # (B, C_out, N)
        y = y_flat.reshape(shape[0], -1, *shape[2:])  # (B, C_out, ...)
        return y


# ---------------------------
# 3. Metrics and Utilities
# ---------------------------

def masked_mse(pred, target, mask=None):
    """MSE with optional masking for variable-length sequences."""
    if mask is None:
        return torch.mean((pred - target) ** 2)
    else:
        return torch.sum((pred - target) ** 2 * mask) / (torch.sum(mask) + 1e-8)


def relative_l2(pred, target, mask=None):
    """Relative L2 error with optional masking."""
    if mask is None:
        return torch.sqrt(torch.sum((pred - target) ** 2)) / (torch.sqrt(torch.sum(target ** 2)) + 1e-8)
    else:
        return torch.sqrt(torch.sum((pred - target) ** 2 * mask)) / (torch.sqrt(torch.sum(target ** 2 * mask)) + 1e-8)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def save_checkpoint(path, model, optimizer=None, epoch=None, best_val=None):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        ckpt["epoch"] = epoch
    if best_val is not None:
        ckpt["best_val"] = best_val
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None):
    """Load model checkpoint."""
    ckpt = torch.load(path, map_location=get_device())
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self, name=None):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.name:
            print(f"{self.name}: {self.interval:.3f} sec")
        else:
            print(f"Elapsed: {self.interval:.3f} sec")
