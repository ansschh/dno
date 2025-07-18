"""
analyze_runs.py
===============
Post-hoc plotting & comparison of Delay-NO variants.

Usage:
    python analyze_runs.py --runs_dir runs --family mackey --out_dir figures/comparison --horizons 50 100 200
"""

import os, argparse, json, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.fft import rfft, rfftfreq

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False
})

METHOD_ORDER = ["stacked", "steps", "kernel"]
COLORS = {"stacked": "#1f77b4", "steps": "#2ca02c", "kernel": "#d62728"}

def load_runs(runs_dir, family):
    run_infos = []
    for path in sorted(glob.glob(os.path.join(runs_dir, f"*_{family}_*"))):
        config_path = os.path.join(path, "config.json")
        metrics_path = os.path.join(path, "metrics.csv")
        if not (os.path.isfile(config_path) and os.path.isfile(metrics_path)):
            continue
        with open(config_path) as f:
            config = json.load(f)
        df = pd.read_csv(metrics_path)
        df["run_dir"] = path
        df["method"] = config["method"]
        df["run_name"] = config["run_name"]
        run_infos.append((config, df))
    return run_infos

def plot_loss_curves(run_infos, out_dir, family):
    plt.figure(figsize=(6,4))
    for cfg, df in run_infos:
        meth = cfg["method"]
        sub = df[df.phase.str.contains("val")]
        plt.plot(sub.epoch, sub.mse.astype(float), label=meth)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title(f"Validation Loss Curves ({family})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"loss_curves_{family}.pdf"))
    plt.savefig(os.path.join(out_dir, f"loss_curves_{family}.png"))
    plt.close()

def latest_samples(run_dir, prefix):
    files = sorted(glob.glob(os.path.join(run_dir, "samples", f"{prefix}*pred.npy")))
    if not files: return []
    # take last epoch's saved group
    last_epoch = max(int(re.search(r"ep(\d+)_", f).group(1)) for f in files)
    pred_files = sorted(glob.glob(os.path.join(run_dir, "samples", f"{prefix}ep{last_epoch:03d}_*pred.npy")))
    pairs = []
    for pf in pred_files:
        tf = pf.replace("_pred.npy", "_true.npy")
        if os.path.isfile(tf):
            pairs.append((np.load(pf), np.load(tf)))
    return pairs

def plot_trajectory_overlays(run_infos, out_dir, family, n_series=3):
    plt.figure(figsize=(10,3*n_series))
    plot_idx = 1
    for cfg, df in run_infos:
        meth = cfg["method"]
        pairs = latest_samples(df.run_dir.iloc[0] if hasattr(df.run_dir,"iloc") else cfg["run_dir"], prefix=meth+"_")
        if not pairs: continue
        for i,(pred,true) in enumerate(pairs[:n_series]):
            plt.subplot(n_series, len(run_infos), plot_idx)
            t = np.arange(true.shape[0])
            plt.plot(t, true[:,0], label="true", lw=2, color="black")
            plt.plot(t, pred[:,0], label=meth, lw=1.5, color=COLORS.get(meth,"gray"))
            if plot_idx == 1: plt.ylabel("u(t)")
            if i == 0: plt.title(meth)
            if plot_idx == len(run_infos): plt.legend()
            plot_idx +=1
    plt.suptitle(f"Trajectory Overlays ({family})", y=0.93)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"trajectory_overlay_{family}.pdf"))
    plt.savefig(os.path.join(out_dir, f"trajectory_overlay_{family}.png"))
    plt.close()

def plot_error_hist(run_infos, out_dir, family):
    rows = []
    for cfg, df in run_infos:
        vs = df[df.phase.str.contains("val")].rel_l2.astype(float).dropna()
        if vs.empty: continue
        rows.append({"method": cfg["method"], "final_rel_l2": vs.iloc[-1]})
    dff = pd.DataFrame(rows)
    plt.figure(figsize=(4,4))
    sns.barplot(data=dff, x="method", y="final_rel_l2", palette=COLORS)
    plt.ylabel("Final rel L2")
    plt.title(f"Final Val relL2 ({family})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"final_relL2_{family}.pdf"))
    plt.savefig(os.path.join(out_dir, f"final_relL2_{family}.png"))
    plt.close()

def spectral_energy(u):
    # u: (T, C) -> use channel 0
    x = u[:,0]
    ft = np.abs(rfft(x - np.mean(x)))
    freq = rfftfreq(len(x), d=1.0)
    return freq[1:], ft[1:]  # skip DC

def plot_spectral(run_infos, out_dir, family):
    plt.figure(figsize=(5,4))
    for cfg, df in run_infos:
        pairs = latest_samples(df.run_dir.iloc[0] if hasattr(df.run_dir,"iloc") else cfg["run_dir"], prefix=cfg["method"]+"_")
        if not pairs: continue
        pred,true = pairs[0]
        f, Ft = spectral_energy(true)
        f2, Fp = spectral_energy(pred)
        plt.loglog(f, Ft, color="black", lw=1.5, label="true" if cfg==run_infos[0][0] else None)
        plt.loglog(f2, Fp, lw=1.3, label=cfg["method"])
    plt.xlabel("Frequency")
    plt.ylabel("|FFT|")
    plt.title(f"Spectral Energy (sample 0) – {family}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"spectral_{family}.pdf"))
    plt.savefig(os.path.join(out_dir, f"spectral_{family}.png"))
    plt.close()

def plot_kernel_heatmap(run_infos, out_dir, family):
    # Only kernel method
    for cfg, df in run_infos:
        if cfg["method"] != "kernel": continue
        # find checkpoint
        ckpts = glob.glob(os.path.join(df.run_dir.iloc[0] if hasattr(df.run_dir,"iloc") else cfg["run_dir"], "best_checkpoint.pt"))
        if not ckpts: continue
        import torch
        state = torch.load(ckpts[0], map_location="cpu")
        # weight name likely: kernel.freq_weights
        weights = state["model_state"]["kernel.freq_weights"].numpy()
        # take magnitude, average over out,in
        mag = np.abs(weights).mean(axis=(0,1))
        plt.figure(figsize=(6,2))
        plt.plot(mag)
        plt.title(f"Kernel Frequency Magnitude – {family}")
        plt.xlabel("Frequency bin")
        plt.ylabel("Mean |W|")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"kernel_freq_{family}.pdf"))
        plt.savefig(os.path.join(out_dir, f"kernel_freq_{family}.png"))
        plt.close()

def plot_field_grid(run_infos, out_dir, family, comp_indices=(0,1,2,3), time_slice=None):
    for cfg, df in run_infos:
        if cfg["method"] not in ["stacked","steps","kernel"]: continue
        pairs = latest_samples(df.run_dir.iloc[0] if hasattr(df.run_dir,"iloc") else cfg["run_dir"], prefix=cfg["method"]+"_")
        if not pairs: continue
        pred,true = pairs[0]
        if pred.shape[1] < 5: continue
        # pred/true shape (T, C) with C=20; choose a time slice near end
        ts = time_slice or min(pred.shape[0]-1, 100)
        fig, axes = plt.subplots(2, len(comp_indices), figsize=(3*len(comp_indices), 5))
        for j,cidx in enumerate(comp_indices):
            axes[0,j].plot(true[:,cidx], color="black")
            axes[0,j].plot(pred[:,cidx], color="red", alpha=0.7)
            axes[0,j].set_title(f"Comp {cidx}")
            # Zoom window last 50 steps
            zstart = max(0, pred.shape[0]-50)
            axes[1,j].plot(true[zstart:,cidx], color="black")
            axes[1,j].plot(pred[zstart:,cidx], color="red", alpha=0.7)
        fig.suptitle(f"{cfg['method']} Reaction-Diffusion Components ({family})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"fieldgrid_{family}_{cfg['method']}.pdf"))
        fig.savefig(os.path.join(out_dir, f"fieldgrid_{family}_{cfg['method']}.png"))
        plt.close()

def make_out_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--family", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="figures/comparison")
    ap.add_argument("--horizons", type=int, nargs="*", default=[50,100,200])
    args = ap.parse_args()

    out_dir = make_out_dir(os.path.join(args.out_dir, args.family))
    run_infos = load_runs(args.runs_dir, args.family)
    if not run_infos:
        print("No runs found.")
        return

    plot_loss_curves(run_infos, out_dir, args.family)
    plot_error_hist(run_infos, out_dir, args.family)
    plot_trajectory_overlays(run_infos, out_dir, args.family)
    plot_spectral(run_infos, out_dir, args.family)
    plot_kernel_heatmap(run_infos, out_dir, args.family)
    
    # Add field grid plot for reaction-diffusion family
    if args.family == "reaction_diffusion":
        plot_field_grid(run_infos, out_dir, args.family)
        
    print(f"Saved figures to {out_dir}")

if __name__ == "__main__":
    main()
