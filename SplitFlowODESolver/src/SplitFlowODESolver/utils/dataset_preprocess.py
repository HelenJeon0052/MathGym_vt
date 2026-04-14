from pathlib import Path
import numpy as np



from typing import Tuple, Optional
import nibabel as nib




from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    ConcatItemsd,
    ToTensord,
)

def _pick_one(path_dir: str, sfx: str) -> str:
    p = Path(path_dir)
    
    for sfx in ("seg", "t1c","t1n","t2f","t2w"):
        hits = sorted(p.glob(f"*{sfx}.nii.gz"))

        print(f"[files] brats files: {hits}")
    return str(hits[0])

def _unique_match(path_dir:str, patterns, key:str) -> str:
    matches = []

    for pattern in patterns:
        matches.extend(glob.glob(os.path.join(path_dir, pattern)))
    
    matches = sorted(set(matches))
    
    if len(matches) == 0:
        raise FileNotFoundError(
            f"[{key}] no file found in {path_dir}\n"
            f"patterns : {patterns}"
        )
    
    if len(matches) > 1:
        raise RuntimeError(
            f"[{key}] multiple files found in {path_dir}\n"
            f"patterns : {patterns}"
            f"matches : {matches}"
        )

    return matches[0]

def find_brats_case_dir(path_dir: str) -> bool:
    if not os.path.isdir(path_dir):
        return false

    raw_patterns = [
        "*-t1n.nii.gz", "*-t1c.nii.gz", "*-t2w.nii.gz", "-t2f.nii.gz",
        "*-t1n.nii", "*-t1c.nii", "*-t2w.nii", "*-t2f.nii",
    ]

    
    
    has_raw = all(len(glob.glob(os.path.join(path_dir, p))) >= 1 for p in raw_patterns[:4]) \
           or all(len(glob.glob(os.path.join(path_dir, p))) >= 1 for p in raw_patterns[4:])
    
    return has_raw



def build_brats_input(
    path_dir: str,
    *,
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    roi_size: Optional[Tuple[int, int, int]] = None,
    add_batch_dim: bool = True,
) ->  np.ndarray:
    """
    
    returns
    ------------
    x : np.ndarray:
        [1, 4, D, H, W]
        [4, D, H, W]
        dtype: float32
    """

    paths = {
        "seg":_pick_one(path_dir, "seg"),
        "t1c":_pick_one(path_dir, "t1c"),
        "t1n":_pick_one(path_dir, "t1n"),
        "t2f":_pick_one(path_dir, "t2f"),
        "t2w":_pick_one(path_dir, "t2w"),
    }

    label_key = ["seg"] 
    image_keys = ["t1c", "t1n", "t2f", "t2w"]
    txs = [
        LoadImaged(keys=image_keys + label_key),
        EnsureChannelFirstd(keys=image_keys + label_key),
        Orientationd(keys=image_keys + label_key, axcodes="RAS"),
        Spacingd(
            keys=image_keys + label_key,
            pixdim=pixdim,
            mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
        ),
        ConcatItemsd(keys=image_keys, name="image", dim=0),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "seg"])
    ]

    if roi_size is not None:
        from monai.transforms import ResizeWithPadOrCropd
        txs.append(ResizeWithPadOrCropd(keys="image", spatial_size=roi_size))

    tx = Compose(txs)
    out = tx(paths)

    x = out["image"].detach().cpu().numpy().astype(np.float32, copy=False)

    if add_batch_dim:
        x = np.expand_dims(x, axis=0)

    return np.ascontiguousarray(x)