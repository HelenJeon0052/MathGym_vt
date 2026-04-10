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
    hits = sorted(p.glob(f"*_{sfx}.nii.gz"))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected: *_{sfx}.nii.gz, but got multiple set {len(hits)}"
        )
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

    expected files:
        - *_0000.nii.gz -> T1
        - *_0001.nii.gz -> T1ce
        - *_0002.nii.gz -> T2
        - *_0003.nii.gz -> FLAIR
    
    returns
    ------------
    x : np.ndarray:
        [1, 4, D, H, W]
        [4, D, H, W]
        dtype: float32
    """

    paths = {
        "t1":_pick_one(path_dir, "0000"),
        "t1ce":_pick_one(path_dir, "0001"),
        "t2":_pick_one(path_dir, "0002"),
        "flair":_pick_one(path_dir, "0003"),
    }

    txs = [
        LoadImaged(keys=["t1", "t1ce", "t2", "flair"]),
        EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair"]),
        Orientationd(keys=["t1", "t1ce", "t2", "flair"], axcodes="RAS"),
        Spacingd(
            keys=["t1", "t1ce", "t2", "flair"],
            pixdim=pixdim,
            mode=("bilinear", "bilinear", "bilinear", "bilinear"),
        ),
        ConcatItemsd(keys=["t1", "t1ce", "t2", "flair"], name="image", dim=0),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys="image")
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