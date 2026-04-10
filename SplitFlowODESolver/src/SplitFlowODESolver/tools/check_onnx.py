from __future__ import annotations



import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Sequence

import onnx
import onnxruntime as ort
import numpy as np



def print_model_io(session: ort.InferenceSession) -> None:
    print("\n[Inputs]")
    for i, inp in enumerate(session.get_inputs()):
        print(f"({i}) name={inp.name}, shape={inp.shape}, type={inp.type}")
    
    print("\n[Outputs]")
    for i, out in enumerate(session.get_outputs()):
        print(f"({i}) name={out.name}, shape={out.shape}, type={out.type}")




def brats_input(
    path_dir: str
)-> np.ndarray:
    br_dst:np.ndarray = build_brats_input(path_dir)

    return br_dst


# TO-DO : dataset check func


def summarize_outputs(outputs: List[np.ndarray], output_names: List[str]) -> None:
    print("\n[Rumtime Outputs]")
    for name, y in zip(output_names, outputs):
        print(
            f"name={name},"
            f"shape={tuple(y.shape)},"
            f"dtype={y.dtype},"
            f"min={float(np.min(y)):.3f},"
            f"max={float(np.max(y)):.3f},"
            f"mean={float(np.mean(y)):.3f}"
        )

def check_onxx_graph(model_path:str) -> None:
    print(f"\n[from onnx] loading graph: {model_path}")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("[from onnx] graph check healthy")

def make_session(model_path: str, use_cuda: bool =  True) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    if use_cuda:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
    
    print(f"[from ort] providers requested : {providers}")
    session = ort.InferenceSession(model_path, providers=providers)
        if session:
            print(f"[from ort] providers active": {session.get_providers()})
        else:
            print(f"[from ort] providers not active: check session obj")
    
    return session

def run_inference(session: ort.InferenceSession, x: np.ndarray) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1:
        raise RuntimeError(
            f"expected input len == 1, got {len(inputs)}"
        )

    input_name = inputs[0].name
    output_names = [o.name for o in outputs]
    print(f"\n[Inference] input : {input_name}")

    print(f"[Inference] x.shape: {x.shape} | x.dtype: {x.dtype}")
    y_list = session.run(output_names, {input_name: x})
    summarize_outputs(y_list, output_names)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check exported ONNX model")

    parser.add_argument("--model", type=str, required=True, help="path to .onnx file")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--depth", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--cpu_only", action="store_true")

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    model_path = str(Path(args.model))
    check_onnx_graph(model_path)

    session = make_session(model_path, use_cuda=True)
    print_model_io(session)

    x = brats_input(
        path_dir
    )

    run_inference(session, x)




if __name__ == "__main__":
    main()