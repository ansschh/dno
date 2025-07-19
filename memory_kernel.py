"""
memory_kernel.py
================
Implements the memory-kernel integral variant.

We approximate:
    du/dt ≈ g(u(t)) + Σ w_j * u(t+s_j)
Discretize history into S points. Use Euler integration to march forward
over the first T_pred steps of the ground-truth y(t).

Run:
    python memory_kernel.py --family mackey
    python memory_kernel.py --family reaction_diffusion --S 64 --euler_dt 0.05
"""

import os, argparse, pickle, math, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common_py34 import (
    set_seed, DDEDataset, collate_pad,
    relative_l2, get_device, save_checkpoint, load_checkpoint, Timer
)
from common_logging import RunLogger

# ---------------------------
# 1. Kernel Module
# ---------------------------
class LearnableKernel(nn.Module):
    """
    Maintains complex frequency weights => performs convolution along history axis.
    hist input shape: (B, S, C)
    """
    def __init__(self, S: int, channels: int):
        super().__init__()
        # Complex weights in frequency domain
        self.S = S
        self.channels = channels
        # param for rfft length = S//2 + 1
        self.freq_weights = nn.Parameter(
            torch.randn(channels, channels, S//2 + 1, dtype=torch.cfloat) * (1 / math.sqrt(channels))
        )

    def forward(self, hist: torch.Tensor):
        # hist: (B,S,C)
        B,S,C = hist.shape
        assert S == self.S, "History length mismatch"
        hist_ft = torch.fft.rfft(hist, dim=1)  # (B, S//2+1, C)
        # Multiply: for each output channel o and input c
        # hist_ft: (B, Freq, C)
        # weights: (C_out, C_in, Freq)
        out_ft = torch.einsum("bfc, ocf -> bfo", hist_ft, self.freq_weights)
        out_time = torch.fft.irfft(out_ft, n=S, dim=1)  # (B,S,C)
        # We interpret the *last* time entry as integral contribution at current t
        return out_time[:, -1, :]  # (B,C)

# ---------------------------
# 2. Local Nonlinear Term g(u)
# ---------------------------
class LocalDynamics(nn.Module):
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels)
        )
    def forward(self, u):
        return self.net(u)

# ---------------------------
# 3. Full Model
# ---------------------------
class MemoryKernelModel(nn.Module):
    def __init__(self, S=32, channels=1, hidden=64):
        super().__init__()
        self.S = S
        self.channels = channels
        self.kernel = LearnableKernel(S, channels)
        self.local = LocalDynamics(channels, hidden)

    def init_history(self, hist_raw: torch.Tensor):
        """
        hist_raw: (B, H_hist, C)
        Resample to S points uniform index space.
        """
        B,H,C = hist_raw.shape
        if H == self.S:
            return hist_raw
        idx_orig = torch.linspace(0, H-1, H, device=hist_raw.device)
        idx_target = torch.linspace(0, H-1, self.S, device=hist_raw.device)
        hist_t = hist_raw.permute(0,2,1)
        out = []
        for it in idx_target:
            f = torch.clamp(it.floor(), 0, H-1).long()
            c = torch.clamp(it.ceil(), 0, H-1).long()
            w = it - it.floor()
            out.append((1-w)*hist_t[...,f] + w*hist_t[...,c])
        res = torch.stack(out, dim=-1)  # (B,C,S)
        return res.permute(0,2,1)       # (B,S,C)

    def forward(self, hist_raw: torch.Tensor, tau: torch.Tensor = None):
        """
        Standard forward method for compatibility with evaluation framework.
        hist_raw: (B, H_hist, C) - raw history data
        tau: (B,) - delay values (not used in this implementation but kept for compatibility)
        Returns: (B, T_pred, C) - predicted trajectory
        """
        # Resample history to fixed S points
        hist_init = self.init_history(hist_raw)  # (B, S, C)
        
        # Use default parameters for Euler integration
        dt = 0.05
        steps = min(400, hist_raw.shape[1])  # Predict same length as input or max 400 steps
        
        return self.euler_roll(hist_init, steps, dt)
    
    def euler_roll(self, hist_init: torch.Tensor, steps: int, dt: float):
        """
        hist_init: (B,S,C) includes the most recent value at index S-1 (time t=0).
        steps: number of forward Euler steps
        Returns trajectories: (B, steps, C)
        """
        hist = hist_init.clone()
        u_t = hist[:, -1, :]  # current state
        traj = []
        for _ in range(steps):
            kernel_term = self.kernel(hist)         # (B,C)
            du = self.local(u_t) + kernel_term
            u_next = u_t + dt * du
            traj.append(u_next)
            # Shift history window
            hist = torch.cat([hist[:, 1:, :], u_next.unsqueeze(1)], dim=1)
            u_t = u_next
        return torch.stack(traj, dim=1)  # (B, steps, C)

# ---------------------------
# 4. Training / Eval
# ---------------------------
def train_epoch(model, loader, opt, device, S, euler_dt, euler_steps):
    model.train()
    total = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        hist_raw = batch["hist"].to(device)
        y = batch["y"].to(device)
        B,T,C = y.shape
        hist = model.init_history(hist_raw)  # (B,S,C)
        steps = min(euler_steps, T)
        pred = model.euler_roll(hist, steps, euler_dt)  # (B,steps,C)
        target = y[:, :steps, :]
        loss = torch.mean((pred - target)**2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * B
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, S, euler_dt, euler_steps):
    model.eval()
    mse_sum, rel_list = 0.0, []
    n = 0
    for batch in tqdm(loader, desc="Eval", leave=False):
        hist_raw = batch["hist"].to(device)
        y = batch["y"].to(device)
        B,T,C = y.shape
        hist = model.init_history(hist_raw)
        steps = min(euler_steps, T)
        pred = model.euler_roll(hist, steps, euler_dt)
        target = y[:, :steps, :]
        mse = torch.mean((pred - target)**2).item()
        mse_sum += mse * B
        rel = relative_l2(pred, target)
        # Move CUDA tensor to CPU before appending to list for numpy conversion
        if isinstance(rel, torch.Tensor):
            rel = rel.cpu().item()
        rel_list.append(rel)
        n += B
    return mse_sum / n, float(np.mean(rel_list))

# ---------------------------
# 5. CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--family", type=str, default="mackey",
                    choices=["mackey","delayed_logistic","neutral","reaction_diffusion"])
    ap.add_argument("--train_split", type=str, default=None)
    ap.add_argument("--test_split", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--S", type=int, default=64, help="Resampled history length")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--euler_dt", type=float, default=0.05)
    ap.add_argument("--euler_steps", type=int, default=400)
    ap.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    train_path = args.train_split or os.path.join(args.data_root, "combined", f"{args.family}_train.pkl")
    test_path  = args.test_split  or os.path.join(args.data_root, "combined", f"{args.family}_test.pkl")
    
    # Initialize the RunLogger
    logger = RunLogger("kernel", args.family, config=vars(args))

    # Inspect output dim
    with open(train_path, "rb") as f:
        samp = pickle.load(f)
    _,_,_, y0 = samp[0]
    y0 = np.asarray(y0)
    if y0.ndim == 1:
        channels = 1
    else:
        channels = y0.shape[1]

    train_ds = DDEDataset(train_path)
    test_ds  = DDEDataset(test_path)
    from common_py34 import collate_pad
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    model = MemoryKernelModel(S=args.S, channels=channels, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ckpt_path = os.path.join(args.checkpoint_dir, f"kernel_{args.family}.pt")
    best_rel = float("inf")

    for epoch in range(1, args.epochs + 1):
        with Timer():
            tr_loss = train_epoch(model, train_loader, opt, device, args.S, args.euler_dt, args.euler_steps)
        val_mse, val_rel = evaluate(model, test_loader, device, args.S, args.euler_dt, args.euler_steps)
        print(f"[Epoch {epoch}] train_mse={tr_loss:.4e} val_mse={val_mse:.4e} val_relL2={val_rel:.4f}")
        
        # Log metrics
        logger.log_metric(epoch, "val", val_mse, val_rel)
        logger.log_metric(epoch, "train", tr_loss, float("nan"))
        
        # Save sample predictions
        if epoch % 10 == 0 or epoch == args.epochs:
            with torch.no_grad():
                try:
                    batch = next(iter(test_loader))
                    hist = batch["hist"].to(device)
                    y = batch["y"].to(device)
                    hist_res = model.init_history(hist)
                    steps = min(args.euler_steps, y.shape[1])
                    pred = model.euler_roll(hist_res, steps, args.euler_dt)
                    logger.save_prediction_samples(epoch, pred, y[:, :steps, :], prefix="kernel_")
                except StopIteration:
                    pass
                    
        if val_rel < best_rel:
            best_rel = val_rel
            save_checkpoint(ckpt_path, model, opt, epoch, best_rel)
            # Also save as best_checkpoint.pt in run directory
            save_checkpoint(os.path.join(logger.run_dir, "best_checkpoint.pt"), model, opt, epoch, best_rel)
            print(f"  ✔ New best (relL2={best_rel:.4f})")

    print("Loading best checkpoint for final evaluation ...")
    load_checkpoint(ckpt_path, model)
    final_mse, final_rel = evaluate(model, test_loader, device, args.S, args.euler_dt, args.euler_steps)
    # Log final evaluation metrics
    logger.log_metric(args.epochs+1, "final", final_mse, final_rel)
    print(f"FINAL: val_mse={final_mse:.4e} val_relL2={final_rel:.4f}")

if __name__ == "__main__":
    main()
