import numpy as np
import torch





def build_checker_input(val_loader) -> np.ndarray:

    batch = next(iter(val_loader))
    
    if "image" not in batch:
        raise KeyError(f"[check onnx] Expected key 'image' must be in batch, but keys = {list(batch.keys)}")
    
    x = batch["image"]

    if not torch.is_tensor(x):
        raise ValueError(f"[check onnx] Expected batch['image'] is tensor, but {type(x)}")

    if x.ndim != 5:
        raise ValueError(f"[check onnx] Expected x shape is 5, but {tuple(x.shape)}")
    
    x = x[:1]
    x = x.detach().cpu().numpy().astype("float32", copy=False)

    print("\n[check onnx]")
    print(f"  shape={tuple(x_np.shape)}")
    print(f"  dtype={x_np.dtype}")
    print(f"  min={float(np.min(x_np)):.6f}")
    print(f"  max={float(np.max(x_np)):.6f}")
    print(f"  mean={float(np.mean(x_np)):.6f}")
    print(f"  finite={bool(np.isfinite(x_np).all())}")   

    return np.ascontiguousarray(x)

def validate_checker_input(x: Any) -> np.ndarray:
    """
    Ensure ONNX checker input is:
      - numpy.ndarray
      - float32
      - shape [1, C, D, H, W]
      - contiguous
      - finite
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    if not isinstance(x, np.ndarray):
        raise TypeError(f"Checker input must be numpy.ndarray or torch.Tensor, got {type(x)}")

    x = x.astype(np.float32, copy=False)
    x = np.ascontiguousarray(x)

    if x.ndim != 5:
        raise ValueError(f"Expected checker input shape [B, C, D, H, W], got {x.shape}")

    if x.shape[0] != 1:
        raise ValueError(f"Expected batch size 1 for checker input, got {x.shape[0]}")

    if not np.isfinite(x).all():
        raise ValueError("Checker input contains NaN or Inf.")

    return x


