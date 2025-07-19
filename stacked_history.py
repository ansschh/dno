"""
stacked_history.py
==================
Implements the stacked history approach where history axis is
treated as another spatial dimension for the FNO architecture.

We approximate u_t+1 = F(u_hist), where u_hist is resampled to S points.
Simplifications:
  * Resampling the history is a simple linear interpolation.
  * Fourier modes are truncated separately in S, X dimensions.
  * Final test metric = simple rMSE over the predicted window.

Run:
  python stacked_history.py --family mackey
  python stacked_history.py --family reaction_diffusion --fourier_modes_s 6 --fourier_modes_x 16
"""

import os, argparse, math, pickle, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common_py34 import (
    set_seed, DDEDataset, collate_pad, SpectralConv, ChannelMLP,
    masked_mse, relative_l2, get_device, save_checkpoint, load_checkpoint, Timer
)
from common_logging import RunLogger

# ---------------------------
# 1. Model Definition
# ---------------------------
class StackedHistoryFNO(nn.Module):
    """
    Input: history tensor (B, H_hist, C) where C=output dim (1 or 20)
    We first resample history to fixed S points (uniform in its own index).
    Then add coordinate embeddings (s in [-1,0], x in [0,1] (or normalized index over 20)).

    Internal representation is (B, C_in, S, X).
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 S: int,
                 X: int,
                 fourier_modes_s: int,
                 fourier_modes_x: int,
                 hidden_channels: int = 64,
                 n_layers: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.S = S
        self.X = X
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden = hidden_channels
        self.n_layers = n_layers

        # Lifting: (in + 2 coords) -> hidden
        self.lift = ChannelMLP(in_channels + 2, hidden_channels)

        # Spectral blocks
        self.spec_layers = nn.ModuleList()
        self.pointwise = nn.ModuleList()
        for _ in range(n_layers):
            self.spec_layers.append(
                SpectralConv(hidden_channels, hidden_channels,
                               n_modes=(fourier_modes_s, fourier_modes_x), n_dims=2)
            )
            self.pointwise.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1))

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj = ChannelMLP(hidden_channels, out_channels)

    def resample_history(self, hist: torch.Tensor) -> torch.Tensor:
        """
        hist: (B, H_hist, C)
        Resample along time dimension to S points (uniform index interpolation).
        For simplicity we use linear interpolation on indices (no actual time values).
        """
        B, H_hist, C = hist.shape
        if H_hist == self.S:
            return hist
        # Create target positions in original index space
        idx_orig = torch.linspace(0, H_hist - 1, H_hist, device=hist.device)
        idx_target = torch.linspace(0, H_hist - 1, self.S, device=hist.device)
        # (B, C, H_hist)
        hist_t = hist.permute(0, 2, 1)
        # Interpolate manually
        # Expand for broadcasting
        idx_orig = idx_orig[None, None, :]
        # For each target index find floor & ceil
        out = []
        for it in idx_target:
            it_floor = torch.clamp(it.floor(), 0, H_hist - 1)
            it_ceil = torch.clamp(it.ceil(), 0, H_hist - 1)
            w = (it - it_floor)
            it_floor = it_floor.long()
            it_ceil = it_ceil.long()
            val = (1 - w) * hist_t[..., it_floor] + w * hist_t[..., it_ceil]
            out.append(val)
        res = torch.stack(out, dim=-1)  # (B,C,S)
        return res.permute(0, 2, 1)     # (B,S,C)

    def add_coords(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, S, X, C) after we broadcast across X.
        Add normalized s and x coords as extra channels
        -> output (B, S, X, C+2)
        """
        B, S, X, C = h.shape
        s_coords = torch.linspace(-1, 0, S, device=h.device).view(1, S, 1, 1).expand(B, S, X, 1)
        x_coords = torch.linspace(0, 1, X, device=h.device).view(1, 1, X, 1).expand(B, S, X, 1)
        return torch.cat([h, s_coords, x_coords], dim=-1)

    def forward(self, hist: torch.Tensor):
        """
        hist: (B, H_hist, C)
        1. Resample to (B, S, C)
        2. Broadcast to (B, S, X, C) if X>1 else unsqueeze dimension
        3. Add coordinates -> (B,S,X,C+2)
        4. Flatten last dim for MLP lift, then permute to (B, hidden, S, X)
        5. n_layers spectral + pointwise conv
        6. Projection back to out_channels over channels dimension
        7. Output predicted "next history window" same shape as input resampled: (B, S, C)
        """
        B, H_hist, C = hist.shape
        h_res = self.resample_history(hist)  # (B,S,C)
        # Broadcast across X:
        h_res = h_res.unsqueeze(2)  # (B,S,1,C)
        if self.X > 1:
            h_res = h_res.expand(B, self.S, self.X, C)

        # Add coords
        h_plus = self.add_coords(h_res)  # (B,S,X,C+2)

        # Lifting: treat features as last dimension
        # Flatten features for linear layer: we process all (S*X) points independently in channel dimension
        z = self.lift(h_plus.view(B * self.S * self.X, C + 2))
        z = z.view(B, self.S, self.X, self.hidden).permute(0, 3, 1, 2)  # (B, hidden, S, X)
        
        # Ensure tensor is 4D for SpectralConv - handle case where X=1 might cause issues
        if self.X == 1 and z.size(-1) == 1:
            # Ensure the last dimension is preserved for 2D spectral convolution
            z = z.contiguous()

        # Spectral blocks
        for spec, pw in zip(self.spec_layers, self.pointwise):
            # Ensure z is 4D for both SpectralConv and Conv2d
            if z.dim() == 3:
                z = z.unsqueeze(-1)  # Add X dimension if missing
            
            z_spec = spec(z)  # (B, hidden, S, X)
            
            # Ensure z_spec has same dimensions as z for addition
            if z_spec.dim() == 3 and z.dim() == 4:
                z_spec = z_spec.unsqueeze(-1)
            elif z_spec.dim() == 4 and z.dim() == 3:
                z = z.unsqueeze(-1)
                
            z = self.act(pw(z) + z_spec)
            z = self.dropout(z)

        # Project each (S,X) point individually
        z = z.permute(0, 2, 3, 1)  # (B,S,X,hidden)
        out = self.proj(z.reshape(B * self.S * self.X, self.hidden))
        
        # Get actual output size and reshape safely
        actual_channels = out.size(-1)
        out = out.reshape(B, self.S, self.X, actual_channels)

        # Collapse spatial if X>1 by averaging across X (for simple next-history prediction)
        # For reaction-diffusion where each "X" is a physical component you might keep it.
        if self.X > 1:
            out = out.mean(2)  # (B,S,C)

        else:
            out = out.squeeze(2)  # (B,S,C)

        return out  # predicted new history window

# ---------------------------
# 2. Training / Evaluation
# ---------------------------
def train_epoch(model, loader, optimizer, device, grad_clip: float = 1.0):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        hist = batch["hist"].to(device)         # (B,H_hist,C)
        # Use last S points of y as target "next window": simple self-supervised shift
        y_full = batch["y"].to(device)

        # Strategy: pick the first S rows of y_full as training target for demonstration
        # (Adjust to more sophisticated next-window if you create windows.)
        target_len = min(model.S, y_full.shape[1])
        target = y_full[:, :target_len, :]
        # Forward
        pred = model(hist)[:, :target_len, :]   # ensure same length
        loss = torch.mean((pred - target)**2)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * hist.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse_sum, rel_list = 0.0, []
    n_samples = 0
    for batch in tqdm(loader, desc="Eval", leave=False):
        hist = batch["hist"].to(device)
        y_full = batch["y"].to(device)
        target_len = min(model.S, y_full.shape[1])
        target = y_full[:, :target_len, :]
        pred = model(hist)[:, :target_len, :]
        mse = torch.mean((pred - target)**2).item()
        mse_sum += mse * hist.size(0)
        rel = relative_l2(pred, target)
        # Move CUDA tensor to CPU before appending to list for numpy conversion
        if isinstance(rel, torch.Tensor):
            rel = rel.cpu().item()
        rel_list.append(rel)
        n_samples += hist.size(0)
    return mse_sum / n_samples, float(np.mean(rel_list))

# ---------------------------
# 3. CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data",
                    help="Root folder containing family subfolders or combined splits.")
    ap.add_argument("--family", type=str, default="mackey",
                    choices=["mackey", "delayed_logistic", "neutral", "reaction_diffusion"])
    ap.add_argument("--train_split", type=str, default=None,
                    help="Path override to train pickle (else uses data_root/combined/<family>_train.pkl)")
    ap.add_argument("--test_split", type=str, default=None,
                    help="Path override to test pickle (else uses data_root/combined/<family>_test.pkl)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--S", type=int, default=32, help="Resampled history length")
    ap.add_argument("--fourier_modes_s", type=int, default=4)
    ap.add_argument("--fourier_modes_x", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_path = args.train_split or os.path.join(args.data_root, "combined", f"{args.family}_train.pkl")
    test_path  = args.test_split  or os.path.join(args.data_root, "combined", f"{args.family}_test.pkl")

    # Initialize the RunLogger
    logger = RunLogger("stacked", args.family, config=vars(args))

    # Load a sample to inspect output dimension
    sample_tmp = torch.load if False else None  # placeholder line to avoid linter complaining
    with open(train_path, "rb") as f:
        train_samples = pickle.load(f)
    first_hist, first_tau, first_t, first_y = train_samples[0]
    first_y = np.asarray(first_y)
    if first_y.ndim == 1:
        D_out = 1
        X = 1
    else:
        # Reaction-diffusion case => treat feature dimension as "spatial width"
        D_out = first_y.shape[1]
        X = D_out if args.family == "reaction_diffusion" else 1
        
    # Adjust fourier_modes_x based on X dimension to avoid dimension mismatch
    if X == 1 and args.fourier_modes_x > 1:
        print(f"Adjusting fourier_modes_x from {args.fourier_modes_x} to 1 to match X dimension")
        args.fourier_modes_x = 1

    # Datasets / loaders
    train_ds = DDEDataset(train_path)
    test_ds  = DDEDataset(test_path)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    model = StackedHistoryFNO(
        in_channels=D_out, out_channels=D_out,
        S=args.S, X=X,
        fourier_modes_s=args.fourier_modes_s,
        fourier_modes_x=args.fourier_modes_x,
        hidden_channels=args.hidden,
        n_layers=args.layers,
        dropout=args.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ckpt_path = os.path.join(args.checkpoint_dir, f"stacked_{args.family}.pt")
    best_rel = float("inf")

    for epoch in range(1, args.epochs + 1):
        with Timer():
            train_loss = train_epoch(model, train_loader, opt, device)
        val_mse, val_rel = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch}] train_mse={train_loss:.4e} val_mse={val_mse:.4e} val_rel_l2={val_rel:.4f}")

        # Log metrics
        logger.log_metric(epoch, "val", val_mse, val_rel)
        logger.log_metric(epoch, "train", train_loss, float("nan"))

        # Save sample predictions
        if epoch % 10 == 0 or epoch == args.epochs:
            with torch.no_grad():
                try:
                    batch = next(iter(test_loader))
                    hist = batch["hist"].to(device)
                    y_full = batch["y"].to(device)
                    tlen = min(model.S, y_full.shape[1])
                    pred = model(hist)[:, :tlen, :]
                    logger.save_prediction_samples(epoch, pred, y_full[:, :tlen, :], prefix="stacked_")
                except StopIteration:
                    pass

        if val_rel < best_rel:
            best_rel = val_rel
            save_checkpoint(ckpt_path, model, opt, epoch, best_rel)
            # Also save as best_checkpoint.pt in run directory
            save_checkpoint(os.path.join(logger.run_dir, "best_checkpoint.pt"), model, opt, epoch, best_rel)
            print(f"  ✔ New best (rel_l2={best_rel:.4f})")

    print("Training complete. Loading best checkpoint for final evaluation ...")
    load_checkpoint(ckpt_path, model)
    final_mse, final_rel = evaluate(model, test_loader, device)
    print(f"FINAL: val_mse={final_mse:.4e} val_relL2={final_rel:.4f}")

if __name__ == "__main__":
    import pickle  # local import to avoid overshadowing earlier code
    main()
