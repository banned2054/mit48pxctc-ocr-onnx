from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypedDict


class ValidationResult(TypedDict):
    width: int
    char_logits_shape: list[int]
    char_logits_max_abs_diff: float
    color_values_shape: list[int]
    color_values_max_abs_diff: float


class ValidationMetadata(TypedDict):
    onnx_checker: str
    widths: list[int]
    max_abs_diff: float
    results: list[ValidationResult]


def max_abs_diff(a: Any, b: Any) -> float:
    import numpy as np

    return float(np.max(np.abs(a - b)))


def validate(checkpoint_path: Path, alphabet_path: Path, onnx_path: Path, widths: list[int]) -> ValidationMetadata:
    import onnx
    import onnxruntime as ort
    import torch

    from mit48pxctc_ocr_onnx.model import load_model

    model = load_model(checkpoint_path, alphabet_path)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX checker: ok ({onnx_path})")

    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    print("Inputs:")
    for item in session.get_inputs():
        print(f"  {item.name}: {item.shape} {item.type}")
    print("Outputs:")
    for item in session.get_outputs():
        print(f"  {item.name}: {item.shape} {item.type}")

    results: list[ValidationResult] = []
    for width in widths:
        torch.manual_seed(width)
        sample = torch.randn(1, 3, 48, width, dtype=torch.float32)
        with torch.inference_mode():
            torch_logits, torch_colors = model(sample)
        ort_logits, ort_colors = session.run(None, {input_name: sample.numpy()})
        logits_diff = max_abs_diff(torch_logits.numpy(), ort_logits)
        colors_diff = max_abs_diff(torch_colors.numpy(), ort_colors)

        print(
            f"width={width}: "
            f"logits {tuple(ort_logits.shape)} diff={logits_diff:.6g}; "
            f"colors {tuple(ort_colors.shape)} diff={colors_diff:.6g}"
        )
        results.append(
            {
                "width": width,
                "char_logits_shape": list(ort_logits.shape),
                "char_logits_max_abs_diff": logits_diff,
                "color_values_shape": list(ort_colors.shape),
                "color_values_max_abs_diff": colors_diff,
            }
        )

    return {
        "onnx_checker": "ok",
        "widths": widths,
        "max_abs_diff": max(
            max(item["char_logits_max_abs_diff"], item["color_values_max_abs_diff"]) for item in results
        ),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MIT 48px CTC ONNX against PyTorch.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alphabet", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, default=Path("dist/mit48pxctc_ocr.onnx"))
    parser.add_argument("--widths", type=int, nargs="+", default=[512, 1024, 1536])
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    metadata = validate(args.checkpoint, args.alphabet, args.onnx, args.widths)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
