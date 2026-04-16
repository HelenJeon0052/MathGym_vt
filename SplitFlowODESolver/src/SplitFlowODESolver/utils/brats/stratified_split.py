import os, glob, random
from collections import defaultdict
import pandas as pd


from typing import Dict, List, Sequence, Tuple, Hashable, TypeVar
import nibabel as nib

import numpy as np
import logging

from SplitFlowODESolver.utils.brats.brats_utils import unique_match


T = TypeVar("T")
logger = logging.getLogger(__name__)

def _pick_one(path_dir: str, sfx: str) -> str:

    pattern = os.path.join(path_dir, f"*-{sfx}.nii.gz")
    matches = sorted(glob.glob(pattern))
    


    if len(matches) == 0:
        raise FileNotFoundError(f"No file Found for {sfx}")
    if len(matches) > 1:
        raise FileNotFoundError(f"Multiple files found for {sfx} in a dir")

    return matches[0]


def find_brats_case_dir(case_dir: str) -> bool:
    if not os.path.isdir(case_dir):
        return false

    raw_patterns = [
        "*-t1n.nii.gz", "*-t1c.nii.gz", "*-t2w.nii.gz", "-t2f.nii.gz",
        "*-t1n.nii", "*-t1c.nii", "*-t2w.nii", "*-t2f.nii",
    ]

    
    
    has_raw = all(len(glob.glob(os.path.join(case_dir, p))) >= 1 for p in raw_patterns[:4]) \
           or all(len(glob.glob(os.path.join(path_dir, p))) >= 1 for p in raw_patterns[4:])
    
    return has_raw

def _resolve_case_dir(case_dir: str) -> str:
    case_dir = os.path.abspath(case_dir)

    if find_brats_case_dir(case_dir):
        return path_dir
    
    if not os.path.isdir(case_dir):
        print(f"Not BraTS directory")

    
    subdirs = [
        os.path.join(case_dir, name)
        for name in sorted(os.listdir(case_dir))
        if os.path.isdir(os.path.join(case_dir, name))
    ]

    case_subdirs = [s for s in subdirs if find_brats_case_dir(s)]

    if len(case_subdirs) == 1:
        return case_subdirs[0]
    
    if len(case_subdirs) > 1:
        raise ValueError(
            f"pass one case folder for testing"
        )
    
    raise ValueError(
        f"Given path: {case_dir}\n"
        "Expected:\n"
        "  - a case folder containing one of:\n"
        "      *-t1n / *-t1c / *-t2w / *-t2f\n"
        "  - or a parent folder containing exactly one such case folder."
    )

def _resolve_modality(case_dir: str) -> Dict[str, str]:

    case_dir = _resolve_case_dir(case_dir)
    case_id = os.path.basename(case_dir)

    paths = {
        "t1c": _unique_match(case_dir, [
            f"{case_id}-t1c.nii.gz", "*-t1c.nii.gz",
            f"{case_id}-t1c.nii", "*-t1c.nii",
        ], "t1c"),
        "t1n": _unique_match(case_dir, [
            f"{case_id}-t1n.nii.gz", "*-t1c.nii.gz",
            f"{case_id}-t1n.nii", "*-t1c.nii",
        ], "t1n"),
        "t2f": _unique_match(case_dir, [
            f"{case_id}-t2f.nii.gz", "*-t2f.nii.gz",
            f"{case_id}-t2f.nii", "*-t2f.nii",
        ], "t2f"),
        "t2w": _unique_match(case_dir, [
            f"{case_id}-t2w.nii.gz", "*-t2w.nii.gz",
            f"{case_id}-t2w.nii", "*-t2w.nii",
        ], "t2w"),
    }

    return paths


def make_case_entry(path_dir: str) -> Dict[str, str]:

    paths = {
        "seg":_pick_one(path_dir, "seg"),
        "t1c":_pick_one(path_dir, "t1c"),
        "t1n":_pick_one(path_dir, "t1n"),
        "t2f":_pick_one(path_dir, "t2f"),
        "t2w":_pick_one(path_dir, "t2w"),
    }

    case_id = os.path.basename(os.path.normpath(path_dir))

    return {
        "case_id": case_id,
        "seg":paths["seg"],
        "t1c":paths["t1c"],
        "t1n":paths["t1n"],
        "t2f":paths["t2f"],
        "t2w":paths["t2w"],
    }

def create_stratified_entries(root_dir: str) -> List[Dict[str, str]]:

    root_dir = os.path.abspath(root_dir)

    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"{root_dir} not dir")
    
    case_dirs = [
        os.path.join(root_dir, name)
        for name in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, name))
    ]

    entries : List[Dict[str, str]] = []

    for case_dir in case_dirs:
        try:
            entries.append(make_case_entry(case_dir))
        except Exception as e:
            raise RuntimeError(f"Failed to parse case dir: {case_dir}") from e
        
    if not entries:
        raise RuntimeError(f"no entry files")
    
    return entries


def _load_seg_array(seg_path:str) -> np.ndarray:
    return np.asarray(nib.load(seg_path).dataobj)

def _case_stats(entry: Dict[str, str]) -> Dict[str, int]:
    seg = _load_seg_array(entry["seg"])
    wt_voxels = int(np.count_nonzero(seg > 0))
    et_present = int(np.any(seg == 4))

    return {
        "wt_voxels" : wt_voxels,
        "et_present" : et_present,
    }

def _create_vol_bins(volumes: Sequence[int], n_bins: int = 4) -> np.ndarray:
    x = np.asarray(volumes, dtype=np.float64)

    if len(x) == 0:
        return np.zeros(0, dtype = np.int64)
    
    edges = np.quantile(x, q = np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)

    if len(edges) <= 2:
        return np.zeros(len(x), dtype=np.int64)

    bins = np.digitize(x, edges[1:-1], right=True)
    return bins.astype(np.int64)

def create_stratify_labels(
    entries: Sequence[Dict[str, str]],
    *,
    use_volume_bin: bool = True,
    n_bins: int = 4,
) -> List[str]:
    stats = [_case_stats(entry) for entry in entries]
    et_flags = [s["et_present"] for s in stats]

    if not use_volume_bin:
        return [f"et{et}" for et in et_flags]
    
    wt_vol = [s["wt_voxels"] for s in stats]
    wt_bins = _create_vol_bins(wt_vol, n_bins=n_bins)

    labels = [
        f"et{et}_wt{wt_bin}"
        for et, wt_bin in zip(et_flags, wt_bins)
    ]

    return labels


def stratified_split(
    items: Sequence[T],
    labels: Sequence[Hashable],
    *,
    ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[T], List[T], List[Hashable], List[Hashable]]:
    
    if len(items) != len(labels):
        raise ValueError(
            f"got {len(items)} != {len(labels)}"
        )
    
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio must be in I(0, 1), got {ratio}")
    
    rng = random.Random(seed)
    buckets: Dict[Hashable, List[Tuple[T, Hashable]]] = defaultdict(list)

    for item, label in zip(items, labels):
        buckets[label].append((item, label))

    train_pairs: List[Tuple[T, Hashable]] = []
    test_pairs: List[Tuple[T, Hashable]] = []

    for label, pairs in buckets.items():
        pairs = pairs[:]
        rng.shuffle(pairs)

        n_total = len(pairs)
        n_test = int(round(n_total * ratio))


        if n_total >= 2:
            n_test = max(1, min(n_test, n_total - 1))
        else:
            n_test = 0

        train_pairs.extend(pairs[n_test:])
        test_pairs.extend(pairs[:n_test])
    
    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)

    train_items = [item for item, _ in train_pairs]
    train_labels = [label for _, label in train_pairs]

    test_items = [item for item, _ in test_pairs]
    test_labels = [label for _, label in test_pairs]

    return train_items, train_labels, test_items, test_labels

def stratified_train_val_split(
    root_dir: str,
    *,
    ratio: float = 0.2,
    seed: int = 42,
    use_volume_bin: bool = True,
    n_bins: int = 4,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str], List[str]]:

    
    entries = create_stratified_entries(root_dir)
    # build_case_entries(root_dir)
    stratify_labels = create_stratify_labels(
        entries, use_volume_bin=use_volume_bin, n_bins=n_bins
    )

    train_entries, val_entries, train_labels, val_labels = stratified_split(
        items = entries,
        labels = stratify_labels,
        ratio = ratio,
        seed = seed,
    )

    return train_entries, train_labels, val_entries, val_labels

def show_label_distribution(name:str, labels: Sequence, *, use_logger = False, verbose = True):
    if isinstance(labels, (list, tuple)):
        try:
            s = pd.Series(labels)
        except Exception as e:
            s = pd.Series([str(x) for x in labels])
    else:
        s = pd.Series(labels)

    counts = s.astype(str).value_counts().sort_index()
    pct = s.astype(str).value_counts(normalize=True).sort_index() * 100
    summary = pd.DataFrame({'count': counts, 'pct': pct.round(2)})

    sample = s.head(10).to_list()

    out_lines = [
        f"{name} stratify distribution:",
        summary.to_string(),
        f"Sample {name} labels (up to 10): {sample}"
    ]
    out = "\n\n".join(out_lines)

    if use_logger:
        logger.info(out)
    elif verbose:
        print(out)


def summary_split(
    train_entries: Sequence[Dict[str, str]],
    train_labels: Sequence[Dict[str, str]],
    val_entries: Sequence[Dict[str, str]],
    val_labels: Sequence[str],
    verbose: bool = True
):
    print(f"train cases: {len(train_entries)}")
    print(f"val cases:   {len(val_entries)}")

    if len(train_entries) != 0:
        for i, entry in enumerate(train_entries[5:10]):
            print(f"{i} entry {entry}")



    if verbose: 
        show_label_distribution("train", train_labels, verbose=True)
        show_label_distribution("val", val_labels, verbose=True)
        
        if isinstance(val_entries, (list, tuple)):
            print("First item is:", type(val_entries[0]), val_entries[0])
            for e, y in list(zip(val_entries, val_labels))[:10]:
                print(e["case_id"], y)
        else:
            raise ValueError(f"val entries not an list")

    train_ids = {x["case_id"] for x in train_entries}
    val_ids = {x["case_id"] for x in val_entries}

    overlap = train_ids & val_ids
    print(f"overlap cases: {len(overlap)}")

    if overlap:
        print("overlap example:", list(sorted(overlap))[:5])