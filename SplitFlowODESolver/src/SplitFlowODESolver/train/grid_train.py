from  __future__ import annotations
import os, csv, time, math, random, copy, math, json
import numpy as np



import torch
import torch.nn as nn
from typing import Any, Dict, Iterable, Tuple, Optional

from torch.utils.data import DataLoader
import argparse
import inspect

from pathlib import Path

from dataclasses import dataclass, field
from datetime import datetime

from prodigyopt import Prodigy

from monai.networks.nets import SwinUNETR

from SplitFlowODESolver.utils.brats.brats_transforms import build_entry_loaders, build_brats_loaders
from SplitFlowODESolver.utils.onnx_utils  import build_checker_input
# from SplitFlowODESolver.model import build_default_hybrid

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



# ------------------------------------------
# utils
# ------------------------------------------
def to_float(x: Any, name: str = "value") -> Optional[float]:
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError(f"{name} must be scalar tensor, got {tuple(x.shape)}")
        x = x.detach().cpu().item()
    
    if isinstance(x, (int, float)):
        x = float(x)
        if not match.isfinite(x):
            raise ValueError(f"{name} is not finite: {x}")
        return x

    raise TypeError(f"{name} must be int, float, scalar tensor, got {type(x)}")



@dataclass
class BestTracker:
    mode: str
    value: Optional[float] = None

    def compare_value(self, new_value: float) -> bool:
        if self.value is None:
            return True
        if self.mode == "min":
            return new_value < self.value
        if self.mode == "max":
            return new_value > self.value
        raise ValueError(f"unsupported mode: {self.mode}")
    
    
    def update(self, new_value: float) -> None:
        self.value = float(new_value)

@dataclass
class ExperimentLogger:
    root_dir: str | Path
    exp_name: str
    config: Dict[str, Any]
    resume_dir: Optional[str | Path] = None

    run_dir: Path = field(init=False)
    ckpt_dir: Path = field(init=False)
    metrics_path: Path = field(init=False)
    config_path: Path = field(init=False)
    best_trackers: Dict[str, BestTracker] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)

        if self.resume_dir is not None:
            self.run_dir = Path(self.resume_dir)
        
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.root_dir / f"{timestamp}_{self.exp_name}"
        
        self.ckpt_dir = self.run_dir / "ckpts"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "config.json"

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if not self.config_path.exists():
            self._save_json(self.config_path, self.config)

    @staticmethod
    def _save_json(path: Path, obj: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=3)
    
    def append_metrics(self, row: Dict[str, Any]) -> None:
        clean = {}

        for k,v in row.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    clean[k] = float(v.detach().cpu().item())
                else:
                    continue
            elif isinstance(v, (int, float, str, bool)) or v is None:
                clean[k] = v
            else:
                clean[k] = str(v)
            
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dump(clean, ensure_ascii=False) + "\n")
    
    def register_best_metric(self, metric_name:str, mode: str) -> None:
        self.best_trackers[metric_name] = BestTracker(mode=mode)
    
    def _pack_ckpt(self, *, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], scheduler: Optional[Any], epoch: int, global_step:int, stage_idx: int, stage_name: str, metrics: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduelr is not None and hasattr(scheduler, "state_dict") else None,
            "epoch": epoch,
            "global_step": global_step,
            "stage_idx": stage_idx,
            "stage_name": str(stage_name),
            "metrics": copy.deepcopy(metrics),
            "config": copy.deepcopy(self.config),
        }

        if extra:
            ckpt["extra"] = copy.deepcopy(extra)
        
        return ckpt

    
    def save_ckpt(self, *, filename: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], scheduler: Optional[Any], epoch: int, global_step: int, stage_idx: int, stage_name: str, metrics: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Path:

        path = self.ckpt_dir / filename
        ckpt = self._pack_ckpt(
            model = model,
            optimizer = optimizer,
            scheduler = scheduler,
            epoch = epoch,
            global_step = global_step,
            stage_idx = stage_idx,
            stage_name = stage_name,
            metrics = metrics,
            extra = extra,
        )
        torch.save(ckpt, path)
        
        return path
    
    def save_last(self, *, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], scheduler: Optional[Any], epoch: int, global_step: int, stage_idx: int, stage_name: str, metrics: Dict[str, Any]) -> Path:
        return self.save_ckpt(
            filename = "ckpt_last.pt",
            model = model,
            optimizer = optimizer,
            scheduler = scheduler,
            epoch = epoch,
            global_step = global_step,
            stage_idx = stage_idx,
            stage_name = stage_name,
            metrics = metrics,
            extra = {"kind":"last"},
        )

    def save_best(self, *, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], scheduler: Optional[Any], epoch: int, global_step: int, stage_idx: int, stage_name: str, metrics: Dict[str, Any]) -> Dict[str, Path]:
        saved: Dict[str, Path] = {}

        for metric_name, tracker in self.best_trackers.items():
            value = metrics.get(metric_name)
            if value is None:
                continue
            
            value = to_float(value, metric_name)
            assert value is not None

            if tracker.compare_value(value):
                tracker.update(value)
                path = self.save_ckpt(
                    filename=f"{metric_name}.pt",
                    model = model,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    epoch = epoch,
                    global_step = global_step,
                    stage_idx = stage_idx,
                    stage_name = stage_name,
                    metrics = metrics,
                    extra = {"kind":"best", "best metrics": metric_name, "best_value": value}
                )
                saved[metric_name] = path
        
        return saved

OPTIMIZERS = {
    "adamw": lambda params, cfg: torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"]),
    "sgd": lambda params, cfg: torch.optim.SGD(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], momentum=cfg.get("momentum", 0.0)),
    "prodigy": lambda params, cfg: Prodigy(params, lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-2), betas = (0.9, 0.999), safeguard_warmup = True, use_bias_correction = True)
}

def build_optimizer(model: torch.nn.Module, optimizer_cfg: Dict[str, Any]) -> torch.optim.Optimizer:

    name = str(optimizer_cfg.get("name", "adamw")).lower()
    lr = float(optimizer_cfg.get("lr", 5e-3))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

    params = [p for p in model.parameters() if p.requires_grad]

    if len(params) == 0:
        raise ValueError("No trainable params found for optimizer")

    """if name == "adamw":
        return torch.optim.AdamW(params, lr = lr, weight_decay = weight_decay)
    elif name == "sgd":
        momentum = float(optimizer_cfg.get("momentum", 0.0))
        return torch.optim.SGD(params, lr = lr, weight_decay = weight_decay, momentum = momentum)
    elif name == "prodigy":
        optimizer = Prodigy(model.parameters(), lr = 1e-4, weight_decay = weight_decay, betas = (0.9, 0.999), safeguard_warmup = True, use_bias_correction = True)
        return optimizer"""

    if name is not in OPTIMIZERS:
        raise ValueError(f"unsupported optimizer, got {name}")

    return OPTIMIZERS[name](params, optimizer_cfg)

def build_scheduler(optimizer: torch.optim.Optimizer, scheduler_cfg: Optional[Dict[str, Any]]) -> Optional[Any]:
    if not scheduler_cfg:
        scheduler_cfg = config
    
    name = str(scheduler_cfg.get("name", "none")).lower()

    if name in {"none", ""}:
        return None
    
    if name == "cosine":
        t_max = int(scheduler_cfg["T_max"])
        eta_min = float(scheduler_cfg.get("eta_min", 0.0))

        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = t_max, eta_min = eta_min)
    
    if name == "step":
        step_size = int(scheduler.cfg["step_size"])
        g = float(scheduler_cfg.get("gamma", 0.1))
        
        return torch.optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = step_size, gamma = g)
    
    if name == "plateau":
        mode = str(scheduler_cfg.get("mode", "min"))
        factor = float(scheduler_cfg.get("factor", 0.1))
        patience = int(scheduler_cfg.get("patience", 10))

        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, mode = mode, factor = factor, patience = patience)
    
    raise ValueError(f"unsupported scheduler, got {name}")

def step_scheduler(scheduler: Optional[Any], val_metrics: Dict[str, Any]) -> None:
    if scheduler is None:
        return

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        monitor_val_loss = val_metrics.get("val_loss")
        if monitor_val_loss is None:
            raise ValueError(f"val loss(val_metrics) is none")
        scheduler.step(float(monitor))
    else:
        scheduler.step()

def get_current_lr(optimizer: torch.optim.Optimizer) -> None:
    if len(optimizer.param_groups) == 0:
        raise ValueError(f"[check lr] optimizer no param groups")
    return float(optimizer.param_groups[0]["lr"])


def merge_epoch(*, global_epoch: int, local_epoch:int, stage_idx: int, stage_name: str, optimizer: torch.optim.Optimizer, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "epoch": int(global_epoch),
        "global_epoch": int(global_epoch),
        "stage_epoch": int(local_epoch),
        "stage_idx": int(stage_idx),
        "stage_name": str(stage_name),
        "lr": get_current_lr(optimizer),
    }

    for src in (train_metrics, val_metrics):
        for k, v in src.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                row[k] = float(v.detach().cpu().item())
            elif isinstace(v, (int, float, str, bool)) or v is None:
                row[k] = v
            else:
                row[k] = v
    return row



def run_stage(
    *, trainer, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optioanl[Any], logger:ExperimentLogger, run_name: str, stage_idx: int, stage_name:str, num_epochs: int, global_epoch_start: int = 0, save_every: int = 0
) -> tuple[int, Dict[str, Any]]:

        global_epoch = int(global_epoch_start)
        last_metrics: Dict[str, Any] = {}

        for local_epoch in range(1, num_epochs + 1):
            
            train_metrics = trainer.train_one(local_epoch)
            val_metrics = trainer.validate_one(local_epoch)
        
            if not isinstance(train_metrics, dict) or not isinstance(val_metrics, dict):
                raise TypeError("trainer.train_one , trainer.validate_one must return dict")
            
            metrics = merge_metrics(
                global_epoch = global_epoch,
                local_epoch = local_epoch,
                stage_idx = stage_idx,
                stage_name = stage_name,
                optimizer = optimizer,
                scheduler = scheduler,
                train_metrics = train_metrics,
                val_metrics = val_metrics
            )

            logger.append_metrics(metrics)

            logger.save_last(
                model = model,
                optimizer = optimizer,
                scheduler = scheduler,
                epoch = global_epoch,
                global_step = global_epoch,
                stage_idx = stage_idx,
                stage_name = stage_name,
                metrics = metrics,
            )

            logger.save_best(
                model = model,
                optimizer = optimizer,
                scheduler = scheduler,
                epoch = global_epoch,
                global_step = global_epoch,
                stage_idx = stage_idx,
                stage_name = stage_name,
                metrics = metrics,
            )

            if save_every > 0 and (global_epoch + 1) % save_every == 0:
                logger.save_ckpt(
                    filename=f"epoch_{global_epoch + 1:03d}.pt"
                    model = model,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    epoch = global_epoch,
                    global_step = global_epoch,
                    stage_idx = stage_idx,
                    stage_name = stage_name,
                    metrics = metrics,
                    extra = {
                        "kind" : "periodic",
                        "run_name": run_name,
                        "stage_epoch": local_epoch
                    }
                )
            
            step_scheduler(scheduler, val_metrics)
        
            print(
                f"[run] {run_name}",
                f"[stage] {stage_dix} | {stage_name}",
                f"[epoch] {global_epoch} / {num_epochs}",
                f"[train] {train_metrics} [validate] {val_metrics}"
            )

            last_metrics = metrics
            global_epoch += 1
        
        return global_epoch, last_metrics