from  __future__ import annotations




import argparse
import inspect
import os, copy, math, json

import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from pathlib import Path

from dataclasses import dataclass, field
from datetime import datetime
from monai.networks.nets import SwinUNETR

from train import SegLogitsWrapper, build_swinunetr, build_model


def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:

    if not isinstance(ckpt, dict):
        raise TypeError("[train] ckpt must be a dict object")
    
    identifiers = ["state_dict", "model", "network", "net", "module"]

    state = {}
    for key in identifiers:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    
        else:
            state = ckpt

    new_state: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module.")]

        new_state[new_k] = v    
    
    return new_state

def load_ckpt_on_py(model: nn.Module, ckpt_path: str, device: str = "cpu") -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location = device)
    state_dict = _extract_state_dict(ckpt)
    missing, unexpected_value = model.load_state_dict(state_dict, strict=False)


    if missing:
        print(f"[inference]-[warn] missing keys {len(missing)}")
        for k in missing[:20]:
            print(" -", k)
        if len(missing) > 20:
            print(" ...")
    if unexpected_value:
        print(f"[inference]-[warn] unexpected values {len(unexpected_value)}")
        for k in unexpected_value[:20]:
            print(" -", k)
        if len(unexpected_value) > 20:
            print(" ...")
    
    return model



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
    parser.add_argument("--out_path", type=str, default="swinunetr_brats.onnx", help="output ONNX path")
    
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

    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)

    args = parser.parse_args()

    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("available:", torch.cuda.is_available())
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    model = build_model(args).to(device)
    
    onnx_model = load_ckpt_on_py(model = model, ckpt_path = args.ckpt, device = args.device)
    out_model = SegLogitsWrapper(onnx_model)

    print(f"model type: {type(model).__name__}")

    export_onnx(
        model = onnx_model,
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