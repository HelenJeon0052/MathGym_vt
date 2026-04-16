from pathlib import Path
import numpy as np



import os, random, json
from typing import Tuple, Optional, Dict, List, Sequence
# import nibabel as nib




from SplitFlowODESolver.utils.brats.brats_utils import pick_one, unique_match, optional_matches, find_brats_case_dir, resolve_case_dir, resolve_modality, resolve_label, scan_case_dirs, rel

from SplitFlowODESolver.utils.brats.stratified_split import create_stratify_labels, stratified_split

def build_brats_entry(case_dir:str, base_dir: Optional[str] = None) -> Dict:
    mods = resolve_modality(case_dir)
    label = resolve_label(case_dir)

    entry = {
        "image": [
            rel(mods["t1c"], base_dir),
            rel(mods["t1n"], base_dir),
            rel(mods["t2f"], base_dir),
            rel(mods["t2w"], base_dir),
        ],
        "case_id": os.path.basename(os.path.normpath(case_dir))
    }
    
    if label is not None:
        entry["label"] = rel(label["seg"], base_dir)

    return entry

def build_brats_datalist(
    train_entries: Sequence[Dict[str, str]],
    val_entries: Sequence[Dict[str, str]],
    base_dir: Optional[str] = None,
    *,
    output_json: Optional[str] = None,
    make_train_val_split: bool = True,
    val_ratio: float = 0.2,
    seed: int = 42,
    train_key: str = "training",
    val_key: str = "validation",
    test_key: str = "test",
    use_volume_bin: bool = True,
    n_bins: int = 4,
) ->  Dict:

    datalist: Dict[str, List[Dict]] = {}

    
    if train_entries is not None:

        """
        # to lebel "seg" as "label"
        # import train_root
        # validate train_root is right
        train_case_dirs = _scan_case_dirs(train_root)
        train_entries = [build_brats_entry(d, base_dir=base_dir) for d in train_case_dirs]
        [caution] diffrent structure of json will be created on the basis of build_brats_entry(case_dir:str, base_dir: Optional[str] = None)
        """
        
        unlabeled = [x for x in train_entries if "seg" not in x]

        if unlabeled:
           raise RuntimeError(f"training root is required to contain segmetation labels: {unlabeled[0]['case_id']}")
 

        # no leakage : ids = [entry["case_id"] for entry in entries]   

        if make_train_val_split:
            if not(0.0 < val_ratio < 1.0):
                raise ValueError(f"validation ratio must be larger than 0.0 but smaller than 1.0")
            
            stratify_labels = create_stratify_labels(
                train_entries,
                use_volume_bin = True,
            )

            train_files, train_labels, val_files, val_labels = stratified_split(
                items = train_entries,
                labels = stratify_labels,
                ratio = 0.2,
                seed = 42,
            )

            datalist[train_key] = train_files
            datalist[val_key] = val_files
        
        else:
            datalist[train_key] = train_entries
    else:
        raise ValueError(f"train entries must be implemented")
        
    if val_entries is not None:
        datalist[test_key] = val_entries
        print(f"[datalist] Test:  {len(datalist[test_key])} cases")
    else:
        raise ValueError(f"val entries must be implemented")

    if len(datalist) == 0:
        raise ValueError(f"datalist is empty")
    
    if output_json is not None:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(datalist, f, indent=2, ensure_ascii=False)

    return datalist