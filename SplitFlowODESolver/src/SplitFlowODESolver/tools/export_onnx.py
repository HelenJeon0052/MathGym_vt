import argparse
import torch
import torch.nn as nn


from pathlib import Path

import onnx
import onnxruntime as ort

from SplitFlowODESolver.utils.onnx_utils  import build_checker_input
from SplitFlowODESolver.utils.brats.brats_transforms import build_entry_loaders, build_brats_loaders
from SplitFlowODESolver.tools.check_onnx import print_model_io, check_onnx_graph, make_session, run_inference


class TriageExportWrapper(nn.Module):
    def __init__(self, model:nn.Module, device):
        super().__init__()
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        
        out = self.model(x)
        
        if isinstance(out, dict):
            if "case_logit" in out:
                return out["case_logit"]
            if "seg_logit" in out:
                return out["seg_logit"]
            raise KeyError(f"dict output keys: {list(out.keys())}, {type(out)} == list")

        if isinstance(out, (tuple, list)):
            y = out[-1]
            if not torch.is_tensor(y):
                raise TypeError(f"output must be a tensor != {type(y)}")

            return y

        if torch.is_tensor(out):
            return out

        raise TypeError(f"Unsupported output: {type(out)}")

class HybridExportWrapper(nn.Module):
    def __init__(self, model:nn.Module, device):
        super().__init__()
        self.model = model.eval()
        self.device = device
    
    @torch.no_grad()
    def forward(self, x):
        
        out = self.model(x)

        if isinstance(out, dict):
            seg = out.get("seg_logits", None)
            cls = out.get("case_logit", None)
            if seg is None or cls is None:
                raise KeyError(f"dict output keys: {list(out.keys())}, {type(out)} == list")
            return seg, cls
        
        if isinstance(out, (tuple, list)):
            if len(out) < 2:
                raise ValueError(f"expected out values are 2, but {len(out)}")

            seg, cls = out[0], out[1]
            if not torch.is_tensor(seg) or not torch.is_tensor(cls):
                raise TypeError(f"output must be a tensor != {type(seg)}, {type(cls)}")
            return seg, cls

        raise TypeError(f"Unsupported output: {type(out)}")

def run_export_onnx_post_check(
    *,
    onnx_path: str,
    train_root: str,
    val_root: str,
    use_cuda: bool = True,
) -> None:




    onnx_path = str(Path(onnx_path))
    train_root = str(Path(train_root))
    val_root = str(Path(val_root))

    
    print("\n[export onnx][post check] building loaders")

    train_entries, val_entries = build_entry_loaders(train_root, val_root)
    _, _, _, val_loader = build_brats_loaders(train_entries, val_entries)

    if len(val_loader) == 0:
        raise ValueError(f"[export onnx] no validation dataset is loaded")
    
    x = build_checker_input(val_loader)

    
    print("\n[export onnx][post check] validating exported onnx")
    check_onnx_graph(onnx_path)
    
    session = make_session(onnx_path, use_cuda = use_cuda)
    
    # check model exported
    print_model_io(session) 
    run_inference(session, x)


    print("\n[export onnx] ONNX runtime good")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Post-export ONNX checker")
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)
    parser.add_argument("--cpu_only", action="store_true")

    args = parser.parse_args()

    run_export_onnx_post_check(
        onnx_path = args.onnx_path,
        train_root = args.train_root,
        val_root = args.val_root,
        use_cuda = not args.cpu_only,
    )
