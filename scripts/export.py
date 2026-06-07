from __future__ import annotations

import argparse
from pathlib import Path


def export_onnx(
    checkpoint_path: Path,
    alphabet_path: Path,
    output_path: Path,
    sample_width: int,
    opset: int,
    dynamic_width: bool,
) -> None:
    import torch

    from mit48pxctc_ocr_onnx.model import load_model

    model = load_model(checkpoint_path, alphabet_path)
    sample = torch.zeros(1, 3, 48, sample_width, dtype=torch.float32)

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_width:
        dynamic_axes = {
            "image": {0: "batch", 3: "width"},
            "char_logits": {0: "batch", 1: "time"},
            "color_values": {0: "batch", 1: "time"},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            sample,
            output_path.as_posix(),
            input_names=["image"],
            output_names=["char_logits", "color_values"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MIT 48px CTC OCR checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alphabet", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist/mit48pxctc_ocr.onnx"))
    parser.add_argument("--sample-width", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fixed-width", action="store_true", help="Export with a fixed input width.")
    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        alphabet_path=args.alphabet,
        output_path=args.output,
        sample_width=args.sample_width,
        opset=args.opset,
        dynamic_width=not args.fixed_width,
    )
    print(f"Exported: {args.output}")


if __name__ == "__main__":
    main()
