# common_logging.py
import os, json, csv, datetime, socket, getpass, math, random
from typing import Dict, Any, List
import numpy as np
import torch

class RunLogger:
    """
    Minimal experiment logger that:
      * Stores config.json
      * Appends rows to metrics.csv
      * Saves sample predictions for later plotting
    """

    def __init__(self, method_name: str, family: str, base_dir="runs", config: Dict[str,Any]=None):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{method_name}_{family}_{ts}"
        self.run_dir = os.path.join(base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.samples_dir = os.path.join(self.run_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.run_dir, "metrics.csv")
        self._init_metrics_csv()
        meta = {
            "run_name": run_name,
            "method": method_name,
            "family": family,
            "hostname": socket.gethostname(),
            "user": getpass.getuser()
        }
        if config:
            meta.update(config)
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.config = meta
        print(f"[RunLogger] Logging to {self.run_dir}")

    def _init_metrics_csv(self):
        with open(self.metrics_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","phase","mse","rel_l2","extra"])

    def log_metric(self, epoch:int, phase:str, mse:float, rel_l2:float, extra:dict=None):
        extra_str = json.dumps(extra) if extra else ""
        with open(self.metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, phase, f"{mse:.6e}", f"{rel_l2:.6e}", extra_str])

    def save_prediction_samples(self, epoch:int, preds:torch.Tensor, targets:torch.Tensor,
                                max_save:int=5, prefix:str=""):
        """
        preds/targets: (B,T,C) or (B,S,C)
        Save first max_save pairs as .npy
        """
        B = preds.shape[0]
        k = min(max_save, B)
        p = preds.detach().cpu().numpy()
        t = targets.detach().cpu().numpy()
        for i in range(k):
            np.save(os.path.join(self.samples_dir, f"{prefix}ep{epoch:03d}_sample{i:03d}_pred.npy"), p[i])
            np.save(os.path.join(self.samples_dir, f"{prefix}ep{epoch:03d}_sample{i:03d}_true.npy"), t[i])
