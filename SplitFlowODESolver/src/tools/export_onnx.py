import argparse
import torch
import torch.nn as nn


from pathlib import Path

class TriageExportWrapper(nn.Module):
    def __init__(self, model:nn.Module, device):
        super().__init__()
        self.model = model.eval()
        self.device = device


    @torch.no_grad()
    def forward(self, x):
        out = self.model(x).to(self.device)
        if isinstance(out, dict):
            if "case_logit" in out:
                return out["case_logit"]
            if "logit" in out:
                return out["logit"]
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
    
    @torch.no_grad()
    def forward(self, x)
        out = self.model(x).to(self.device)

        if isinstance(out, dict):
            seg = out.get("seg_logits", None)
            cls = out.get("case_logit", None)
            if seg is None or cls is None:
                raise KeyError(f"dict output keys: {list(out.keys())}, {type(out)} == list")
            return seg, cls
        
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            seg, cls = out[0], out[1]
            if not torch.is_tensor(seg) or not torch.is_tensor(cls):
                raise TypeError(f"output must be a tensor != {type(seg)}, {type(cls)}")
            return seg, cls

        raise TypeError(f"Unsupported output: {type(out)}")
    