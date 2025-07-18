"""
method_of_steps.py
==================
Implements the Method-of-Steps operator approach.

We learn a one-step map: (history, tau) -> next ΔT segment (of length STEP_OUT)
Then roll forward autoregressively to cover the full future target portion.

Simplifications:
  * History is uniformly resampled to S points.
  * We predict the next STEP_OUT raw points of the ground-truth y.
  * Loss = MSE over predicted segment (teacher forcing to avoid exploding early).

Run:
  python method_of_steps.py --family mackey
  python method_of_steps.py --family delayed_logistic --S 32 --step_out 32 --roll_steps 10
"""

import os, argparse, math, pickle, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    set_seed, DDEDataset, collate_pad, ChannelMLP,
    masked_mse, relative_l2, get_device, save_checkpoint, load_checkpoint, Timer
)
from common_logging import RunLogger

# ---------------------------
# 1. History Encoder (1-D small FNO-like or simple Conv)
# ---------------------------
class SimpleHistEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden=64, S=32, n_layers=3):
        super().__init__()
        layers = []
        C = in_channels
        for i in range(n_layers):
            layers.append(nn.Conv1d(C, hidden, kernel_size=3, padding=1))
            layers.append(nn.GELU())
            C = hidden
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # global summarization
    def forward(self, h):
        # h: (B, S, C) -> transpose to (B,C,S)
        h = h.permute(0, 2, 1)
        z = self.net(h)  # (B, hidden, S)
        return self.pool(z).squeeze(-1)  # (B, hidden)

# ---------------------------
# 2. Step Operator
# ---------------------------
class StepOperator(nn.Module):
    def __init__(self, hist_dim, tau_dim=16, hidden=128, out_len=32, out_channels=1):
        super().__init__()
        self.tau_embed = nn.Sequential(
            nn.Linear(1, tau_dim),
            nn.GELU(),
            nn.Linear(tau_dim, tau_dim),
            nn.GELU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(hist_dim + tau_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_len * out_channels)
        )
        self.out_len = out_len
        self.out_channels = out_channels

    def forward(self, hist_vec, tau):
        tau_e = self.tau_embed(tau)   # (B, tau_dim)
        z = torch.cat([hist_vec, tau_e], dim=-1)
        out = self.predictor(z)
        return out.view(out.size(0), self.out_len, self.out_channels)

# ---------------------------
# 3. Full Model (Wrap encoder + step op + rollout)
# ---------------------------
class MethodOfStepsModel(nn.Module):
    def __init__(self, S=32, step_out=32, roll_steps=5,
                 in_channels=1, out_channels=1,
                 hist_hidden=64, hist_layers=3,
                 tau_dim=16, predictor_hidden=128):
        super().__init__()
        self.S = S
        self.step_out = step_out
        self.roll_steps = roll_steps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = SimpleHistEncoder(in_channels, hist_hidden, S, hist_layers)
        self.stepper = StepOperator(hist_dim=hist_hidden, tau_dim=tau_dim,
                                    hidden=predictor_hidden, out_len=step_out,
                                    out_channels=out_channels)

    def resample_history(self, h):
        # h: (B, H_hist, C)
        B, H_hist, C = h.shape
        if H_hist == self.S:
            return h
        idx_orig = torch.linspace(0, H_hist - 1, H_hist, device=h.device)
        idx_target = torch.linspace(0, H_hist - 1, self.S, device=h.device)
        h_t = h.permute(0,2,1)  # (B,C,H_hist)
        out = []
        for it in idx_target:
            f = torch.clamp(it.floor(), 0, H_hist-1).long()
            c = torch.clamp(it.ceil(), 0, H_hist-1).long()
            w = it - it.floor()
            out.append((1-w)*h_t[...,f] + w*h_t[...,c])
        res = torch.stack(out, dim=-1)  # (B,C,S)
        return res.permute(0,2,1)

    def forward(self, hist, tau):
        # hist: (B,H_hist,C)
        h_res = self.resample_history(hist)  # (B,S,C)
        h_vec = self.encoder(h_res)          # (B,hist_hidden)
        y_next = self.stepper(h_vec, tau)    # (B, step_out, C)
        return y_next

    def rollout(self, hist, tau, steps: int):
        """
        Autoregressively extend history 'steps' times.
        Returns concatenated predictions shape (B, steps*step_out, C)
        """
        h_cur = hist
        preds = []
        for k in range(steps):
            y_next = self.forward(h_cur, tau)  # one segment
            preds.append(y_next)
            # Append predicted segment to history and trim to last S points (using predicted values)
            # Simplest approach: treat predicted y as continuation; re-sample combined.
            h_aug = torch.cat([h_cur, y_next], dim=1)  # (B,H_hist+step_out,C)
            if h_aug.shape[1] > self.S:
                h_cur = h_aug[:, -self.S:, :]
            else:
                h_cur = h_aug
        return torch.cat(preds, dim=1)  # (B, steps*step_out, C)

# ---------------------------
# 4. Train / Eval
# ---------------------------
def train_epoch(model, loader, opt, device):
    model.train()
    total = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        hist = batch["hist"].to(device)
        tau  = batch["tau"].to(device)
        y    = batch["y"].to(device)  # (B, T, C)
        B, T, C = y.shape
        target_seg_len = min(model.step_out, T)  # enforce alignment
        target = y[:, :target_seg_len, :]
        pred = model(hist, tau)[:, :target_seg_len, :]
        loss = torch.mean((pred - target)**2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * B
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, rollout=False):
    model.eval()
    mse_sum, rel_list = 0.0, []
    n = 0
    for batch in tqdm(loader, desc="Eval", leave=False):
        hist = batch["hist"].to(device)
        tau  = batch["tau"].to(device)
        y    = batch["y"].to(device)
        B, T, C = y.shape
        if not rollout:
            target_seg_len = min(model.step_out, T)
            target = y[:, :target_seg_len, :]
            pred = model(hist, tau)[:, :target_seg_len, :]
        else:
            steps = min(model.roll_steps, math.ceil(T / model.step_out))
            pred_full = model.rollout(hist, tau, steps)
            pred = pred_full[:, :T, :]
            target = y
        mse = torch.mean((pred - target)**2).item()
        mse_sum += mse * B
        rel_list.append(relative_l2(pred, target))
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
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--S", type=int, default=32)
    ap.add_argument("--step_out", type=int, default=32)
    ap.add_argument("--roll_steps", type=int, default=10)
    ap.add_argument("--hist_hidden", type=int, default=64)
    ap.add_argument("--hist_layers", type=int, default=3)
    ap.add_argument("--tau_dim", type=int, default=16)
    ap.add_argument("--predictor_hidden", type=int, default=128)
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
    logger = RunLogger("steps", args.family, config=vars(args))

    # Inspect output dimension
    with open(train_path, "rb") as f:
        samples = pickle.load(f)
    _,_,_, y0 = samples[0]
    y0 = np.asarray(y0)
    if y0.ndim == 1:
        out_channels = 1
    else:
        out_channels = y0.shape[1]

    train_ds = DDEDataset(train_path, args.family)
    test_ds  = DDEDataset(test_path, args.family)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    model = MethodOfStepsModel(
        S=args.S,
        step_out=args.step_out,
        roll_steps=args.roll_steps,
        in_channels=out_channels,
        out_channels=out_channels,
        hist_hidden=args.hist_hidden,
        hist_layers=args.hist_layers,
        tau_dim=args.tau_dim,
        predictor_hidden=args.predictor_hidden
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ckpt_path = os.path.join(args.checkpoint_dir, f"steps_{args.family}.pt")
    best_rel = float("inf")

    for epoch in range(1, args.epochs + 1):
        with Timer():
            train_loss = train_epoch(model, train_loader, opt, device)
        val_mse, val_rel = evaluate(model, test_loader, device, rollout=False)
        print(f"[Epoch {epoch}] train_mse={train_loss:.4e} val_mse={val_mse:.4e} val_relL2(one-step)={val_rel:.4f}")
        
        # Log metrics
        logger.log_metric(epoch, "val_one_step", val_mse, val_rel)
        logger.log_metric(epoch, "train", train_loss, float("nan"))
        
        # Save sample predictions
        if epoch % 10 == 0 or epoch == args.epochs:
            with torch.no_grad():
                try:
                    batch = next(iter(test_loader))
                    hist = batch["hist"].to(device)
                    tau = batch["tau"].to(device)
                    y = batch["y"].to(device)
                    seg_len = min(model.step_out, y.shape[1])
                    pred = model(hist, tau)[:, :seg_len, :]
                    logger.save_prediction_samples(epoch, pred, y[:, :seg_len, :], prefix="steps_")
                except StopIteration:
                    pass
        
        if val_rel < best_rel:
            best_rel = val_rel
            save_checkpoint(ckpt_path, model, opt, epoch, best_rel)
            # Also save as best_checkpoint.pt in run directory
            save_checkpoint(os.path.join(logger.run_dir, "best_checkpoint.pt"), model, opt, epoch, best_rel)
            print(f"  ✔ New best (one-step relL2={best_rel:.4f})")

    print("Evaluating best checkpoint with multi-step rollout ...")
    load_checkpoint(ckpt_path, model)
    roll_mse, roll_rel = evaluate(model, test_loader, device, rollout=True)
    # Log the rollout metrics
    logger.log_metric(args.epochs, "rollout", roll_mse, roll_rel)
    print(f"FINAL ROLLOUT: mse={roll_mse:.4e} relL2={roll_rel:.4f}")

if __name__ == "__main__":
    main()
