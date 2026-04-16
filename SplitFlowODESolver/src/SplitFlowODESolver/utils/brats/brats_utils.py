from pathlib import Path
import os, glob



from typing import Tuple, Optional, List, Dict

import numpy as np

import re




# -------------------------------------------------
# constants
# -------------------------------------------------

CASE_DIR_RE = re.compile(r"^BraTS-GLI-\d{5}-\d{3}$")

MODALITY_SFX = {
    "seg": "-seg.nii.gz",
    "t1c": "-t1c.nii.gz",
    "t1n": "-t1n.nii.gz",
    "t2f": "-t2f.nii.gz",
    "t2w": "-t2w.nii.gz",
}

# -------------------------------------------------
# common utils
# -------------------------------------------------

def pick_one(path_dir: str, sfx: str) -> str:
    p = Path(path_dir)
    
    hits = sorted(p.glob(f"*{sfx}.nii.gz"))

    if not hits:
        print(f"[files] brats files have different suffix")
    
    return str(hits[0])

def unique_match(case_dir:str, patterns, key:str) -> str:
    matches = []

    for pattern in patterns:
        matches.extend(glob.glob(os.path.join(case_dir, pattern)))
    
    matches = sorted(set(matches))
    
    if len(matches) == 0:
        raise FileNotFoundError(
            f"[{key}] no file found in {case_dir}\n"
            f"patterns : {patterns}"
        )
    
    if len(matches) > 1:
        raise RuntimeError(
            f"[{key}] multiple files found in {case_dir}\n"
            f"patterns : {patterns}"
            f"matches : {matches}"
        )

    return matches[0]

def optional_matches(case_dir: str, patterns: List[str]) -> Optional[str]:
    matches = []

    for pattern in patterns:
        matches.extend(glob.glob(os.path.join(case_dir, pattern)))
    
    matches = sorted(set(matches))
    
    if len(matches) == 0:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            f"[{label}] multiple files found in {case_dir}\n"
            f"patterns : {patterns}"
            f"matches : {matches}"
        )

    return matches[0]

def optional_matches_sfx(case_dir: str, sfx: str) -> Optional[str]:
    
    matches = []
    
    matches = sorted(case_dir.glob(f"*{sfx}"))
    
    if len(matches) == 0:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            f"[{label}] multiple files found in {case_dir}\n"
            f"matches : {matches}"
        )

    return matches[0]

def find_brats_case_dir(case_dir: str) -> bool:
    if not os.path.isdir(case_dir):
        return false

    raw_patterns = [
        "*-t1n.nii.gz", "*-t1c.nii.gz", "*-t2w.nii.gz", "*-t2f.nii.gz",
    ]
    
    
    """has_raw = all(len(glob.glob(os.path.join(case_dir, p))) >= 1 for p in raw_patterns[:4]) \
           or all(len(glob.glob(os.path.join(case_dir, p))) >= 1 for p in raw_patterns[4:])"""
    raw = all(glob.glob(os.path.join(case_dir, p)) for p in raw_patterns)
    
    return raw

def resolve_case_dir(case_dir: str) -> str:
    case_dir = os.path.abspath(case_dir)

    if find_brats_case_dir(case_dir):
        return case_dir
    
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
        "    or\n"
        "      *_0000 / *_0001 / *_0002 / *_0003\n"
        "  - or a parent folder containing exactly one such case folder."
    )

def resolve_modality(case_dir: str) -> Dict[str, str]:

    case_dir = resolve_case_dir(case_dir)
    case_id = os.path.basename(case_dir)

    paths = {
        "t1c": unique_match(case_dir, [
            f"{case_id}-t1c.nii.gz", "*-t1c.nii.gz",
            f"{case_id}-t1c.nii", "*-t1c.nii",
        ], "t1c"),
        "t1n": unique_match(case_dir, [
            f"{case_id}-t1n.nii.gz", "*-t1n.nii.gz",
            f"{case_id}-t1n.nii", "*-t1n.nii",
        ], "t1n"),
        "t2f": unique_match(case_dir, [
            f"{case_id}-t2f.nii.gz", "*-t2f.nii.gz",
            f"{case_id}-t2f.nii", "*-t2f.nii",
        ], "t2f"),
        "t2w": unique_match(case_dir, [
            f"{case_id}-t2w.nii.gz", "*-t2w.nii.gz",
            f"{case_id}-t2w.nii", "*-t2w.nii",
        ], "t2w"),
    }
    
    return paths


def resolve_label(case_dir:str) -> Optional[str]:
    case_id = os.path.basename(os.path.normpath(case_dir))
    label_path = {
        "seg": optional_matches(case_dir, [
            f"{case_id}-seg.nii.gz", "*-seg.nii.gz",
            f"{case_id}-seg.nii", "*-seg.nii"
        ]),
    }
    
    return label_path

def scan_case_dirs(root_dir: str) -> List[str]:


    root_dir = os.path.abspath(root_dir)
    # print(f"root_dir : {root_dir}")
    if not os.path.isdir(root_dir):
        print(f"{root_dir} is not a directory")
    
    case_dirs = []

    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path) and find_brats_case_dir(path):
            case_dirs.append(path)

    if not case_dirs:
        raise RuntimeError(f"No BraTS case dir found")

    return case_dirs

def rel(path: str, base_dir: Optional[str]) -> str:
    if base_dir is None:
        return os.path.abspath(path)
    return os.path.relpath(path, start = base_dir)


# -------------------------------------------------
# directory loop utils
# -------------------------------------------------

def is_brats_dir(path: Path) -> bool:
    return path.is_dir() and CASE_DIR_RE.match(path.name) is not None


def find_case_dirs(root_dir: str | Path) -> List[Path]:
    
    root = Path(root_dir).expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(f"[Loop: case_dir] Not a directory")
    
    case_dirs = sorted([p for p in root.iterdir() if is_brats_dir(p)])

    if not case_dirs:
        raise RuntimeError(f"[Loop: case_dir] No BraTS case directories")
    
    return case_dirs

def resolve_dir(case_dir: str | Path, require_seg: bool) -> Dict[str, str]:

    case_path = Path(case_dir).expanduser().resolve()

    if not is_brats_dir(case_dir):
        raise ValueError(f"Not a case dir under : {case_path}")
    
    entry: Dict[str, str] = {"case_id": case_path.name}

    seg_keys = ["seg"]
    required_keys = ["t1n", "t1c", "t2w", "t2f"]

    for key in required_keys:
        f = optional_matches_sfx(case_path, MODALITY_SFX[key])
        if f is None:
            raise FileNotFoundError(f"Missing required modality: {key} in {case_dir}")
        
        entry[key] = str(f)

    seg_file = optional_matches_sfx(case_path, MODALITY_SFX["seg"])
    
    if require_seg and seg_file is None:
        raise FileNotFoundError(f"Missing required segmentation files")
    
    if seg_file is not None:
        entry["seg"] = str(seg_file)

    return entry

def build_case_entries(root_dir: str | Path, require_seg: bool = True) -> List[Dict[str, str]]:

    case_dirs = find_case_dirs(root_dir)
    entries = [resolve_dir(case_dir, require_seg=require_seg) for case_dir in case_dirs]
    return entries

def to_monai_entry(entry: dict) -> dict:

    out = {
        "case_id": entry["case_id"],
        "image": [entry["t1c"], entry["t1n"], entry["t2f"], entry["t2w"]]
    }

    if "seg" in entry:
        out["label"] = entry["seg"]
        
    return out