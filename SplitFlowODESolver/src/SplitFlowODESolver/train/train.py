from  __future__ import annotations



import argparse
import inspect
import os, copy, math, json

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from pathlib import Path
from typing import Any, Dict, Optional

from dataclasses import dataclass, field
from datetime import datetime
from monai.networks.nets import SwinUNETR

from prodigyopt import Prodigy
from tqdm import tqdm



from monai.losses import DiceCELoss

# from SplitFlowODESolver.models.vit_3d import Light3DVit
from SplitFlowODESolver.utils.brats.brats_transforms import build_entry_loaders, build_brats_loaders
from SplitFlowODESolver.utils.onnx_utils  import build_checker_input
# from SplitFlowODESolver.model import build_default_hybrid

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def remap_brats_labels(y: torch.Tensor) -> torch.Tensor:
    """
    input, output:
        [B, 1, D, H, W]
    """

    y = y.long()
    return torch.where(y == 4, torch.tensor(3, device = y.device, dtype = y.dtype), y)

def build_case_label(y_lesion: torch.Tensor) -> torch.Tensor:
    """
    y_lesion : [B, D, H, W]
    return:
        [B, 1], binary case
    """
    print("👈 :", y_lesion.shape)
    y_case = (y_lesion > 0).flatten(1).any(dim=1).float().unsqueeze(1)
    return y_case

def prep_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    print(batch.keys())
    x = batch["image"].to(device, non_blocking=True)
    y = batch["seg"].to(device, non_blocking=True)
    y = remap_brats_labels(y).long()
    y_idx = y[:, 0]
    y_lesion = (y_idx > 0).float()
    y_case = build_case_label(y_lesion)


    return x, y, y_idx, y_case, y_lesion

class SegLogitsWrapper(nn.Module):
    """
    only raw logits
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)

        if torch.is_tensor(out):
            return out

        if hasattr(out, "seg_logits") and out.seg_logits is not None:
            return out.seg_logits
        
        raise TypeError(f"[error] unsupported model output: {type(out)}")

class _TensorHook:
    def __init__(self, module: nn.Module):
        self.tensor: Optional[torch.Tensor] = None
        self.handle = module.register_forward_hook(self._hook)
    
    def _hook(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError(f"module returns non-tensor type, {type(output)}")
        self.tensor = output

    def clear(self):
        self.tensor = None

    def close(self):
        self.handle.remove()

# Triage for step 1 - 4
class TriageHead(nn.Module):
    def __init__(self, in_channels: int, dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.norm = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, 1)
    
    def forward(self, feat: torch.Tensor) ->  torch.Tensor:
        x = self.pool(feat).flatten(1)
        if x.ndim == 2:
            x = self.norm(x)
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
        else:
            raise ValueError(f"input must be binary tensor, got {x.shape}")

        return x


@dataclass
class MultiTaskOutput:
    seg_logits: torch.Tensor
    case_logits: Optional[torch.Tensor] = None
    aux_seg_logits: Optional[torch.Tensor] = None
    feat: Optional[torch.Tensor] = None

class SwinUnetrMultiTask(nn.Module):

    
    """
    helpers for step 1 - 4
    """
    def __init__(self, model: nn.Module, hook_name: str, triage_in_channels: int, out_channels: int, triage_dim: int = 256, return_feat: bool = False, cls: bool = True, aux_seg: bool = True):
        super().__init__()
        self.model = model
        self.return_feat = return_feat
        self.cls = cls
        self.aux_seg = aux_seg

        module_dict = dict(self.model.named_modules())

        if hook_name not in module_dict:
            n = sorted(module_dict.keys())
            preview = "\n".join(n[:120])
            raise KeyError(
                f"hook name {hook_name} not found in model.named_modules().\n"
                f"first modules:\n{preview}"
            )
        self._feature_hook = _TensorHook(module_dict[hook_name])

        if self.cls:
            self.triage_head = TriageHead(
                in_channels = triage_in_channels,
                dim = triage_dim,
            )
        
        if self.aux_seg:
            self.aux_seg_head = nn.Conv3d(triage_in_channels, out_channels, kernel_size=1)
        
    def freeze_model(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.model.eval()

    def unfreeze_model(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True
        
        self.model.train()
    
    def forward(self, x: torch.Tensor) -> MultiTaskOutput:
        self._feature_hook.clear()

        out = self.model(x)
        
        if torch.is_tensor(out):
            seg_logits = out
        elif hasattr(out, "seg_logits") and out.seg_logits is not None:
            seg_logits = out.seg_logits
        else:
            raise TypeError(f"unsupported type of output from model, {type(out)}")

        feat = self._feature_hook.tensor

        if feat is None and (self.cls or self.aux_seg):
            raise RuntimeError(f"feature hook captured nothing, check name of hook and SwinUNETR structure")
        
        case_logits = None
        aux_seg_logits = None

        if self.cls:
            case_logits = self.triage_head(feat)
        
        if self.aux_seg:
            aux_seg_logits = self.aux_seg_head(feat)
            aux_seg_logits = F.interpolate(
                aux_seg_logits, size = seg_logits.shape[2:], mode = "trilinear", align_corners = False
            )
        
        return MultiTaskOutput(
            seg_logits = seg_logits, case_logits = case_logits, aux_seg_logits = aux_seg_logits, feat = feat if self.return_feat else None,
        )

def print_module_name(model: nn.Module, max_items: int = 300) -> None:
    print("\n[named modules]")
    count = 0
    for i, (name, module) in enumerate(model.named_modules()):
        print(f"{i:03d}: {name} -> {module.__class__.__name__}")
        count += 1
        if i >= max_items - 1:
            print("...truncated")
            break
    print(f"[module] {count}")


def build_swinunetr(
    in_channels: int,
    out_channels: int,
    roi_x:int,
    roi_y:int,
    roi_z:int,
    feature_size: int = 48,
    use_checkpoint: bool = True,
    spatial_dim: int = 3.
) -> nn.Module:

    kwargs: Dict[str, Any] = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "feature_size": feature_size,
        "use_checkpoint": use_checkpoint,
        "spatial_dims": spatial_dim,
    }
    
    signature = inspect.signature(SwinUNETR.__init__)

    if "img_size" in signature.parameters:
        kwargs["img_size"] = (roi_x, roi_y, roi_z)

    if "feature_size" in signature.parameters:
        kwargs["feature_size"] = feature_size
    elif "feat_size" in signature.parameters:
        kwargs["feat_size"] = feat_size
    


    model = SwinUNETR(**kwargs)

    return model

def validate_roi_size(roi_x: int, roi_y: int, roi_z: int) -> None:
    for name, value in [("roi_x", roi_x), ("roi_y", roi_y), ("roi_z", roi_z)]:
        if value <= 0:
            raise ValueError(f"roi size must be positive integers")
        if value % 32 != 0:    
            raise ValueError(f"roi size must be even numbers and divisible by 32")


# ---------------------------------------
# model factory
# ---------------------------------------

def build_model(args) -> nn.Module:

    swinunetr = build_swinunetr(
            in_channels = args.in_channels,
            out_channels = args.out_channels,
            roi_x = args.roi_x,
            roi_y = args.roi_y,
            roi_z = args.roi_z,
            feature_size = args.feature_size,
            use_checkpoint = True,
            spatial_dim = 3,
        )
    
    if args.print_modules:
        print_module_names(swinunetr)

    if args.model_kind == "swinunetr":
        return swinunetr
    
    if args.model_kind == "hybrid":
        triage_in_channels = args.triage_in_channels
        if triage_in_channels <= 0:
            triage_in_channels = args.feature_size * 16
        
        return SwinUNETRMultiTask(
            model = swinunetr,
            hook_name = args.hook_name,
            triage_in_channels = args.triage_in_channels,
            out_channels = args.out_channels,
            triage_dim = args.triage_dim,
            return_feat = False,
            cls = True,
            aux_seg = True
        )
    
    raise ValueError(f"Unsupported model kind, {args.model_kind}")

# --------------------------------------------
# losses, metrics
# --------------------------------------------

class MultiTaskCriterion(nn.Module):
    def __init__(self, lambda_cls: float = 0.1, lambda_aux_seg: float = 0.2):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_aux_seg = lambda_aux_seg
        self.seg_loss = DiceCELoss(
            include_background = True,
            to_onehot_y = True,
            softmax = True,
            lambda_dice = 1.0,
            lambda_ce = 1.0
        )
        self.cls_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, output: Any, y_seg: torch.Tensor, y_case: torch.Tensor) -> Dict[str, torch.Tensor]:
        if isinstance(output, MultiTaskOutput):
            seg_logits = output.seg_logits
            loss_seg = self.seg_loss(seg_logits, y_seg)

            loss_aux = torch.zeros((), device = seg_logits.device)
            if output.aux_seg_logits is not None:
                loss_aux = self.seg_loss(output.aux_seg_logits, y_seg)
            
            num_pos = (y_case == 1).sum().item()
            num_neg = (y_case == 0).sum().item()

            pos_weight = torch.tensor([num_neg / num_pos], dtype = torch.float, device = seg.logits.device)
            cls_loss = self.cls_loss(pos_weight = pos_weight)
            
            loss_cls = torch.zeros((), device = seg_logits.device)
            if output.case_logits is not None:
                loss_cls = cls_loss(output.case_logits.view(-1), y_case.view(-1))
            
            total_loss = loss_seg + self.lambda_aux * loss_aux + self.lambda_cls * loss_cls

            return {
                "loss": total_loss,
                "loss_seg": loss_seg,
                "loss_aux": loss_aux,
                "loss_cls": loss_cls
            }
        
        if hasattr(output, "seg_logits") and output.seg_logits is not None:
            seg_logits = output.seg_logits
            loss_seg = self.seg_loss(seg_logits, y_seg)

            return {
                "loss": loss_seg,
                "loss_seg": loss_seg,
                "loss_aux": torch.zeros((), device = seg_logits.device),
                "loss_cls": torch.zeros((), device = seg_logits.device),
            }

        else:
            raise TypeError(f"output cannot extract seg logits")

def save_ckpt(path: str | Path, *, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any], epoch: int, train_loss: float, val_loss:float, config: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "config": config,
    }

    torch.save(ckpt, path)

    print(f"[train ckpt] saved {path}")

def train_one(*, model, train_loader, optimizer, scheduler, device, epoch, criterion: MultiTaskCriterion) -> Dict[str, float]:

    model.train()

    total_loss = {"loss" : 0.0, "loss_seg" : 0.0, "loss_aux" : 0.0, "loss_cls" : 0.0}
    # total_loss_dice = 0.0
    # total_dice = 0.0

    use_cuda_amp = (device.type == "cuda")
    use_bf = use_cuda_amp and torch.cuda.is_bf16_supported()
    amp_type = torch.bfloat16 if use_bf else torch.float16

    scaler = GradScaler("cuda" ,enabled = (use_cuda_amp and not use_bf))

    pbar = tqdm(train_loader, desc = f'[seg train mode] Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):

        x, y_seg, _, y_case, y_lesion = prep_batch(batch, device)

        optimizer.zero_grad(set_to_none = True)

        with autocast(device_type = device.type, dtype=amp_type):
            logits = model(x)

            logits_fp = logits.float()

            losses = criterion(logits_fp, y_seg, y_case)

        if scaler.is_enabled():   
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        for k in total_loss:
            total_loss[k] += float(losses[k].detach().cpu().item())
        # total_loss_dice += loss_dice.item()
        
        avg_losses = {k: v / max(1, len(train_loader)) for k, v in total_loss.items()}
        pbar.set_postfix(loss=f"{losses['loss'].item():3f}", seg_loss=f"{losses['seg_loss'].item():.3f}", aux_loss = f"{losses['aux_loss'].item():.3f}")
    
    return {f"[train]_{k}": v / max(1, len(train_loader)) for k, v in total_losses.items()}
        
@torch.no_grad()
def validate_one(*, model, val_loader, device: torch.device, epoch: int, criterion: MultiTaskCriterion) -> Dict[str, float]:

    model.eval()
    total_loss = {"loss" : 0.0, "loss_seg" : 0.0, "loss_aux" : 0.0, "loss_cls" : 0.0}

    use_cuda_amp = (device.type == "cuda")
    use_bf = use_cuda_amp and torch.cuda.is_bf16_supported()
    amp_type = torch.bfloat16 if use_bf else torch.float16

    pbar = tqdm(val_loader, desc=f"[seg validate mode] Epoch {epoch}")

    for batch in pbar:
        x, y_seg, _, y_case, y_lesion = prep_batch(batch, device)

        with autocast(device_type = device.type, dtype = amp_type, enabled=use_cuda_amp):
            logits = model(x)
            logits_fp = logits.float()
            losses = criterion(logits_fp, y_seg, y_case)

        for k in total_loss:
            total_loss += float(losses[k].detach().cpu().item())
        avg_val_loss = {k: v / max(1, len(val_loader)) for k, v in total_losses.items()}
        pbar.set_postfix(val_loss=f"{losses['loss'].item():.3f}", seg_val_loss=f"{losses['seg_loss'].item():.3f}", cls_val_loss = f"{losses['cls_loss'].item():.3f}")    
    
    return {f"[validate]_{k}": v / max(1, len(val_loader)) for k, v in total_losses.items()}

def fit(*, model, train_loader, val_loader, optimizer: torch.optim.Optimizer, scheduler: Optional[Any], device: torch.device, epochs: int, ckpt_dir: str | Path, criterion: MultiTaskCriterion, config: Dict[str, Any]) -> None:
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_losses = train_one(
            model = model, train_loader = train_loader, optimizer = optimizer, scheduler = scheduler, device = device, epoch = epoch, criterion = criterion
        )

        val_losses = validate_one(
            model = model, val_loader = val_loader, device = device, epoch = epoch, criterion = criterion
        )

        train_loss = train_losses['loss']
        val_loss = val_losees['loss']

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(
            f"[epoch] {epoch:03d}"
            f"[train loss] {train_loss:.5f}"
            f"[val loss] {val_loss:.5f}"
            f"[learning rate] {optimizer.param_groups[0]['lr']:.6e}"
        )

        save_ckpt(
            Path(ckpt_dir) / "last.pt", model = model, optimizer = optimizer, scheduler = scheduler, epoch = epoch, train_loss = train_loss, val_loss = val_loss, config = config
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(
                Path(ckpt_dir) / "best_val_loss.pt", model = model, optimizer = optimizer, scheduler = scheduler, epoch = epoch, train_loss = train_loss, val_loss = val_loss, config = config
            )

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_dir", type=str, required=True, help="path to .pt files / checkpoints")
    
    parser.add_argument("--model_kind", type=str, default="swinunetr", choices=["swinunetr", "hybrid"])
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--unet_feat_channels", type=int, default=256)

    parser.add_argument("--roi_x", type=int, default=96)
    parser.add_argument("--roi_y", type=int, default=96)
    parser.add_argument("--roi_z", type=int, default=96)

    parser.add_argument("--hook_name", type=str, default="encoder10")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=75)

    parser.add_argument("--print_modules", action="store_true")

    args = parser.parse_args()

    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("available:", torch.cuda.is_available())
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_root = ""
    val_root = ""

    train_entries, val_entries = build_entry_loaders(train_root, val_root)
    _, train_loader, _, val_loader = build_brats_loaders(train_entries, val_entries)

    validate_roi_size(args.roi_x, args.roi_y, args.roi_z)

    # help(SwinUNETR)

    model = build_model(args).to(device)

    optimizer = Prodigy(model.parameters(), lr = 1e-4, weight_decay = 1e-2, betas = (0.9, 0.999), safeguard_warmup = True, use_bias_correction = True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=0.0
    )

    criterion = MultiTaskCriterion(
        lambda_cls = 0.0,
        lambda_aux_seg = 0.2
    )

    fit(model = model, train_loader = train_loader, val_loader = val_loader, optimizer = optimizer, scheduler = scheduler, device = device, epochs = args.epochs, ckpt_dir = args.ckpt_dir, criterion = criterion, config = vars(args))

if __name__ == "__main__":
    main()