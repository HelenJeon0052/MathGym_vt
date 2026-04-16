from pathlib import Path
import numpy as np



from typing import Dict, Tuple, Optional, List, Sequence
import nibabel as nib

import re




from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    ConcatItemsd,
    ResizeWithPadOrCropd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
)

from monai.data import CacheDataset, DataLoader

from SplitFlowODESolver.utils.brats.brats_utils import pick_one, unique_match, optional_matches, find_brats_case_dir, resolve_case_dir, resolve_modality, resolve_label, scan_case_dirs, rel, build_case_entries 
from SplitFlowODESolver.utils.brats.stratified_split import create_stratify_labels, stratified_split, summary_split




def make_case_entry(path_dir: str) -> Dict[str, str]:
    paths = {
        "seg":pick_one(path_dir, "seg"),
        "t1c":pick_one(path_dir, "t1c"),
        "t1n":pick_one(path_dir, "t1n"),
        "t2f":pick_one(path_dir, "t2f"),
        "t2w":pick_one(path_dir, "t2w"),
    }

    return {
        "seg":paths["seg"],
        "t1c":paths["t1c"],
        "t1n":paths["t1n"],
        "t2f":paths["t2f"],
        "t2w":paths["t2w"],
    }

def make_entries(root_dir: str) -> List[Dict[str, str]]:

    """case_dirs = [
        os.path.join(root_dir, d)
        for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    entries = [make_case_entry(d) for d in case_dirs]"""

    entries = build_case_entries(root_dir)

    return entries


def validate_pixdim_roi(
    pixdim: Tuple[float, float, float],
    roi_size:Tuple[int, int, int]
    ):
    if not (isinstance(pixdim, (tuple, list)) and len(pixdim) == 3):
        raise ValueError(f"pixdim len == 3, got {pixdim}")
    if not all(isinstance(x, (int, float)) and x > 0 for x in pixdim):
        raise ValueError(f"pixdim values > 0, got {pixdim}")

    if not (isinstance(roi_size, (tuple, list)) and len(roi_size) == 3):
        raise ValueError(f"roi_size len == 3, got {roi_size}")
    if not all(isinstance(x, int) and x > 0 for x in roi_size):
        raise ValueError(f"roi_size values > 0 (96, 128, etc), got {type(roi_size)} {roi_size}")

def validate_pure_entries(entries: List[Dict[str, str]], require_seg: bool = True) -> None:

    if len(entries) == 0:
        raise ValueError(f"entries empty")
    


    passed = set()

    required = {"case_id", "t1c", "t1n", "t2f", "t2w"}

    if require_seg:
        required.add("seg")
    
    for i, entry in enumerate(entries):
        missing = required - set(entry.keys())

        if missing:
            print(f"missing: {missing}")
        
        case_id = entry["case_id"]
        if case_id in passed:
            raise ValueError(f"Duplicated case id found: {case_id}")
        
        passed.add(case_id)

        for key in required - {"case_id"}:
            p = Path(entry[key])
            if not p.is_file():
                raise FileNotFoundError(f"Entry[{i}] key = {key} : file not found: {p}")
            
    print(f"validated >> {len(entries)} loaded successfully")

    return len(entries)


def build_brats_train(
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    roi_size: Tuple[int, int, int] = (96, 96, 96)
):
    
    validate_pixdim_roi(pixdim, roi_size)

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
        ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=roi_size),
        RandCropByPosNegLabeld(
            keys=["image", "seg"],
            label_key="seg",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,

        ),
        RandFlipd(keys=["image", "seg"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "seg"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image", "seg"], spatial_axis=2, prob=0.5),
        RandScaleIntensityd(keys="image", factors = 0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets = 0.1, prob=0.5),
        ToTensord(keys=["image", "seg"])
    ]

    tx = Compose(txs)
    # out = tx(paths)

    return tx



def build_brats_validation(
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    roi_size: Tuple[int, int, int] = (96, 96, 96),
):
    
    validate_pixdim_roi(pixdim, roi_size)

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
        ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=roi_size),
        ToTensord(keys=["image", "seg"])
    ]

    tx = Compose(txs)
    # out = tx(paths)

    return tx


def build_entry_loaders(
    train_root: str | Path,
    val_root: str | Path,
):
    
    train_entries = build_case_entries(train_root, require_seg = True)
    val_entries = build_case_entries(val_root, require_seg = False)

    validate_pure_entries(train_entries, require_seg = True)
    validate_pure_entries(val_entries, require_seg = False)

    return train_entries, val_entries


def build_brats_loaders(
    train_entries: Sequence[Dict[str, str]],
    val_entries: Sequence[Dict[str, str]],
    *,
    pixdim = (1.0, 1.0, 1.0),
    roi_size = (96, 96, 96),
    train_batch_size = 2,
    val_batch_size = 1,
    num_workers = 0,
    cache_rate = 0.0,
):

    num_train_entries = validate_pure_entries(train_entries, require_seg = True)

    if len(train_entries) == num_train_entries:
        try:
            
            stratify_labels = create_stratify_labels(
                entries = train_entries,
                use_volume_bin = True,
            )

            train_files, train_labels, val_files, val_labels = stratified_split(
                items = train_entries,
                labels = stratify_labels,
                ratio = 0.2,
                seed = 42,
            )

            print("len(all_train_entries):", len(train_entries))
            print("len(val_entries):", len(val_entries))
            print("len(stratify_labels):", len(stratify_labels))

            print("len(train_entries):", len(train_files))
            print("len(val_entries):", len(val_files))

            print("len(train_labels):", len(train_labels))
            print("len(val_labels):", len(val_labels))

            print("train unique labels:", sorted(set(train_labels)))
            print("val unique labels:", sorted(set(val_labels)))

        except Exception as e:
            print(f"[Exracting Entries] {e}")
    
    else:
        # train_entries, val_entries, train_labels, val_labels = stratified_train_val_split(root_dir)
        raise ValueError(f"train entries and validation entries must be provided")
        
        
    if len(train_entries) == 0:
        raise ValueError(f"extracting entries has problems")

    else:
        summary_split(train_files, train_labels, val_files, val_labels)

    train_transform = build_brats_train(
        pixdim = pixdim, roi_size = roi_size
    )

    val_transform = build_brats_validation(
        pixdim = pixdim, roi_size = roi_size
    )

    train_dataset = CacheDataset(
        data = train_files,
        transform = train_transform,
        cache_rate = cache_rate,
        num_workers = num_workers,
    )

    val_dataset = CacheDataset(
        data = val_files,
        transform = val_transform,
        cache_rate = cache_rate,
        num_workers = num_workers,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = train_batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = val_batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True,
    )


    if len(train_loader) != 0:
        print(f"train loader is {len(train_loader)}")

    return train_dataset, train_loader, val_dataset, val_loader