#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


MODEL_TYPES = (
    "pilotnet",
    "mobilenet_v3_small",
    "resnet18",
    "vit_tiny",
    "pilotnet_phase",
    "mobilenet_v3_small_phase",
    "resnet18_phase",
    "vit_tiny_phase",
    "resnet18_lidar",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export .pth policy checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="")
    parser.add_argument("--input-width", type=int, default=160)
    parser.add_argument("--input-height", type=int, default=90)
    parser.add_argument("--use-phase", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--opset", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import numpy as np
    import torch

    from policy_models import create_policy_model, model_uses_lidar, model_uses_phase

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_type = args.model_type or checkpoint.get("model_type")
    if not model_type:
        raise ValueError("--model-type is required when checkpoint does not contain model_type")
    input_width = int(checkpoint.get("input_width", args.input_width))
    input_height = int(checkpoint.get("input_height", args.input_height))
    use_phase = bool(args.use_phase or checkpoint.get("use_phase", False) or model_uses_phase(model_type))
    use_lidar = bool(checkpoint.get("use_lidar", False) or model_uses_lidar(model_type))

    device = choose_device(args.device, torch)
    model = create_policy_model(
        model_type,
        input_width=input_width,
        input_height=input_height,
        use_phase=use_phase,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = torch.zeros(1, 3, input_height, input_width, device=device)
    inputs = (image,)
    input_names = ["image"]
    dynamic_axes = {"image": {0: "batch"}, "steer_norm": {0: "batch"}}
    if use_lidar:
        lidar = torch.zeros(1, 2, 360, device=device)
        inputs = (image, lidar)
        input_names.append("lidar")
        dynamic_axes["lidar"] = {0: "batch"}
    elif use_phase:
        phase = torch.zeros(1, 1, device=device)
        inputs = (image, phase)
        input_names.append("phase")
        dynamic_axes["phase"] = {0: "batch"}

    torch.onnx.export(
        model,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=["steer_norm"],
        dynamic_axes=dynamic_axes,
        opset_version=max(12, args.opset),
    )
    print(f"saved ONNX: {output_path}")
    print("ONNX/TensorRT FP16 can be tested later on Jetson Orin Nano if TorchScript is too slow.")

    try:
        import onnxruntime as ort
    except ImportError:
        print("WARN onnxruntime is not installed; skipping ONNX Runtime smoke test.")
        return

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    feed = {"image": np.zeros((1, 3, input_height, input_width), dtype=np.float32)}
    if use_lidar:
        feed["lidar"] = np.zeros((1, 2, 360), dtype=np.float32)
    elif use_phase:
        feed["phase"] = np.zeros((1, 1), dtype=np.float32)
    output = session.run(None, feed)[0]
    print(f"onnxruntime output_shape={output.shape} sample={float(output.reshape(-1)[0]):.6f}")


def choose_device(name: str, torch_module):
    if name == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if name == "cuda" and not torch_module.cuda.is_available():
        print("WARN CUDA requested but unavailable; falling back to CPU.")
        return torch_module.device("cpu")
    return torch_module.device(name)


if __name__ == "__main__":
    main()
