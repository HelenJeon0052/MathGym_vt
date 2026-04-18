import argparse
import inspect


import torch
import torch.nn as nn
from typing import Any, Dict

from pathlib import Path

from monai.networks.nets import SwinUNETR



from SplitFlowODESolver.utils.brats.brats_transforms import build_entry_loaders, build_brats_loaders
from SplitFlowODESolver.utils.onnx_utils  import build_checker_input
from SplitFlowODESolver.model import buildbuild_default_hybrid


class SegLogitswrapper(nn.Module):
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


def build_swinunetr(
    in_channels: int,
    out_channels: int,
    roi_x:int,
    roi_y:int,
    roi_z:int,
    feat_size: int = 48,
    use_checkpoint: bool = True,
    spatial_dim: int = 3.
) -> nn.Module:

    kwargs: Dict[str, Any] = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "feat_size": feat_size,
        "use_checkpoint": use_checkpoint,
        "spatial_dims": spatial_dims,
    }

    signiture = inspect.signiture(SwinUNETR.__init__)

    if "img_size" in signiture.parameters:
        kwargs["img_size"] = (roi_x, roi_y, roi_z)
    


    model = SwinUNETR(**kwargs)

    return model


def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:

    if not isinstance(ckpt, dict):
        raise TypeError("[train] ckpt must be a dict object")
    
    identifiers = ["state_dict", "model", "network", "net", "module"]

    for key in identifies:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    
        else:
            state = skpt

    new_state: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module.")]:

        new_state[new_k] = v    
    
    return new_state

def load_ckpt_on_py(model: nn.Module, ckpt_path: str, device: str = "cpu") -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location = device)
    state_dict = _extract_state_dict(ckpt)
    missing, unexpected_value = model.load_state_dict(state_dict, strict=False)


    if missing:
        print(f"[train]-[warn] missing keys {len(missing)}")
        for k in missing[:20]:
            print(" -", k)
        if len(missing) > 20:
            print(" ...")
    if unexpected_value:
        print(f"[train]-[warn] unexpected values {len(unexpected_value)}")
        for k in unexpected_value[:20]:
            print(" -", k)
        if len(unexpected_value) > 20:
            print(" ...")
    
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

    unet = build_swinunetr(
            in_channels = args.in_channels,
            out_channels = args.out_channels,
            roi_x = args.roi_x,
            roi_y = args.roi_y,
            roi_z = args.roi_z,
            feat_size = args.feat_size,
            use_checkpoint = True,
            spatial_dim = 3,
        )
    
    
    if args.model_kind == "swinunetr":
        return unet
    
    if args.model_kine == "hybrid":
        return build_default_hybrid(
            unet = unet,
            unet_feat_channels = args.unet_feat_channels,
            triage_embed_dim = (48, 96, 192),
            triage_depth = (2, 2, 2),
            patch_size = 4,
            triage_num = 256
        )
    
    raise ValueError(f"Unsupported model kind, {args.model_kind}")


def export_onnx(
    model:nn.Module,
    out_path:str,
    in_channels:int,
    roi_x:int,
    roi_y:int,
    roi_z:int,
    opset:int = 18,
    dynamic_batch: bool = True,
    verify: bool = True,
    device: str = "cpu",
) -> None:

    model.eval()
    model.to(device)

    out_path = str(Path(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    

    # loading dataset
    """x = build_checker_input(val_loader)"""

    # dummy for exporting presets to onnx
    image = torch.randn(1, in_channels, roi_x, roi_y, roi_z, device = device)
    
    export_kwargs = dict(
        model = model,
        args = (image,),
        f = out_path,
        input_names = ["image"],
        output_names = ["logits"],
        opset_version = opset,
        dynamo = True,
        external_data = False,
        verify = verify,
    )

    if dynamic_batch:
        export_kwargs["dynamic_shapes"] = {"image": {0: "batch"}}
    
    torch.onnx.export(**export_kwargs)
    print(f"[train] exported to {out_path}")




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, required=True, help="path to .pt files / checkpoints")
    parser.add_argument("--out", type=str, default="swinunetr_brats.onnx", help="output ONNX path")
    
    parser.add_argument("--model_kind", type=str, default="unet", choices=["unet"])
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--out_channels", type=int, default=3) # why is it 3?
    parser.add_argument("--feat_size", type=int, default=48)
    parser.add_argument("--unet_feat_channels", type=int, default=256)

    parser.add_argument("--roi_x", type=int, default=96)
    parser.add_argument("--roi_y", type=int, default=96)
    parser.add_argument("--roi_z", type=int, default=96)

    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--fixed_batch", action="store_true")
    args = parser.parse_args()

    validate_roi_size(args.roi_x, args.roi_y, args.roi_z)
    

    

    model = load_ckpt_on_py(model, args.ckpt, device=args.device)
    model = SegLogitsWrapper(model) # why is model loaded from ckpt

    print(f"model : {type(model)} | {model.keys()}")

    export_onnx(
        model = model,
        out_path = args.out_path,
        in_channels = args.in_channels,
        roi_x = args.roi_x,
        roi_y = args.roi_y,
        roi_z = args.roi_z,
        opset = args.opset,
        dynamic_batch = True,
        verify = args.verify,
        device = args.device,
    )


if __name__ == "__main__":
    main()