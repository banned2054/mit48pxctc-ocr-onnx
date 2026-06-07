from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any


SOURCE_PROJECT = "zyddnys/manga-image-translator"
SOURCE_RELEASE = "beta-0.3"
SOURCE_RELEASE_URL = "https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3"
SOURCE_ARCHIVE_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr-ctc.zip"
)
GITHUB_REPO_URL = "https://github.com/banned2054/mit48pxctc-ocr-onnx"
HF_REPO_URL = "https://huggingface.co/banned404/mit48pxctc-ocr-onnx"


BADGES = textwrap.dedent(
    f"""\
    [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-banned404%2Fmit48pxctc--ocr--onnx-yellow?logo=huggingface)]({HF_REPO_URL})
    [![GitHub](https://img.shields.io/badge/GitHub-banned2054%2Fmit48pxctc--ocr--onnx-black?logo=github)]({GITHUB_REPO_URL})
    """
).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while chunk := fp.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(module_name: str) -> str:
    module = __import__(module_name)
    return str(module.__version__)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_readme(validation: dict[str, Any]) -> str:
    lines = []
    for result in validation["results"]:
        lines.append(
            "width={width}: logits diff={logits:.6g}; colors diff={colors:.6g}".format(
                width=result["width"],
                logits=result["char_logits_max_abs_diff"],
                colors=result["color_values_max_abs_diff"],
            )
        )
    validation_text = "\n".join(lines)
    badges = BADGES.replace("\n", "\n        ")
    validation_text = validation_text.replace("\n", "\n        ")

    return textwrap.dedent(
        f"""\
        ---
        license: gpl-3.0
        library_name: onnx
        tags:
        - onnx
        - ocr
        - ctc
        - manga
        - text-recognition
        pipeline_tag: image-to-text
        ---

        # MIT 48px CTC OCR ONNX

        {badges}

        [简体中文](README.zh-CN.md)

        This repository provides an ONNX conversion of the 48px CTC OCR model used by
        [manga-image-translator](https://github.com/zyddnys/manga-image-translator).

        The ONNX artifact is derived from the upstream PyTorch checkpoint
        `ocr-ctc.ckpt` from the `{SOURCE_RELEASE}` release asset `ocr-ctc.zip`.

        ## Files

        ```text
        mit48pxctc_ocr.onnx
        alphabet-all-v5.txt
        metadata.json
        LICENSE
        NOTICE
        ```

        ## Source

        - Upstream project: <https://github.com/zyddnys/manga-image-translator>
        - Upstream release: <{SOURCE_RELEASE_URL}>
        - Source archive: `ocr-ctc.zip`
        - Source checkpoint: `ocr-ctc.ckpt`
        - Source alphabet: `alphabet-all-v5.txt`

        ## Model Contract

        Input:

        - name: `image`
        - dtype: `float32`
        - shape: `[batch, 3, 48, width]`
        - color order: BGR
        - normalization: `(uint8_pixel - 127.5) / 127.5`

        Outputs:

        - `char_logits`: `[batch, time, vocab_size]`
        - `color_values`: `[batch, time, 6]`

        `char_logits` is not softmaxed. `color_values` is not clamped. The first
        dictionary entry is the CTC blank token. The special token `<SP>` represents a
        normal space.

        ## Validation

        The ONNX export was checked with `onnx.checker` and compared against the
        PyTorch checkpoint with ONNX Runtime CPU execution.

        ```text
        {validation_text}
        ```

        ## Export

        The model was exported with:

        ```bash
        uv run --extra export python scripts/export.py \\
          --checkpoint origin_model/ocr-ctc.ckpt \\
          --alphabet origin_model/alphabet-all-v5.txt \\
          --output dist/mit48pxctc_ocr.onnx
        ```

        ## License

        This ONNX conversion and the accompanying files are distributed under
        GPL-3.0-only. See `LICENSE`.

        The upstream project is GPL-3.0 licensed. Upstream authorship and copyright
        remain with the original authors and contributors of manga-image-translator and
        the model authors. See `NOTICE` for source attribution and redistribution
        authorization details.
        """
    )


def build_readme_zh(validation: dict[str, Any]) -> str:
    lines = []
    for result in validation["results"]:
        lines.append(
            "width={width}: logits diff={logits:.6g}; colors diff={colors:.6g}".format(
                width=result["width"],
                logits=result["char_logits_max_abs_diff"],
                colors=result["color_values_max_abs_diff"],
            )
        )
    validation_text = "\n".join(lines)
    badges = BADGES.replace("\n", "\n        ")
    validation_text = validation_text.replace("\n", "\n        ")

    return textwrap.dedent(
        f"""\
        # MIT 48px CTC OCR ONNX

        {badges}

        [English](README.md)

        本仓库提供 manga-image-translator 使用的 48px CTC OCR 模型的 ONNX 转换版本。

        ONNX 产物派生自上游 `{SOURCE_RELEASE}` release asset `ocr-ctc.zip` 中的
        PyTorch checkpoint `ocr-ctc.ckpt`。

        ## 文件

        ```text
        mit48pxctc_ocr.onnx
        alphabet-all-v5.txt
        metadata.json
        LICENSE
        NOTICE
        ```

        ## 上游来源

        - 上游项目：<https://github.com/zyddnys/manga-image-translator>
        - 上游发布：<{SOURCE_RELEASE_URL}>
        - 源压缩包：`ocr-ctc.zip`
        - 源 checkpoint：`ocr-ctc.ckpt`
        - 源字典：`alphabet-all-v5.txt`

        ## 模型接口

        输入：

        - name: `image`
        - dtype: `float32`
        - shape: `[batch, 3, 48, width]`
        - color order: BGR
        - normalization: `(uint8_pixel - 127.5) / 127.5`

        输出：

        - `char_logits`: `[batch, time, vocab_size]`
        - `color_values`: `[batch, time, 6]`

        `char_logits` 未经过 softmax。`color_values` 未经过 clamp。字典第一项是
        CTC blank token。特殊 token `<SP>` 表示普通空格。

        ## 验证

        ONNX 导出已通过 `onnx.checker`，并使用 ONNX Runtime CPU execution 与
        PyTorch checkpoint 输出进行对比。

        ```text
        {validation_text}
        ```

        ## 导出

        导出命令：

        ```bash
        uv run --extra export python scripts/export.py \\
          --checkpoint origin_model/ocr-ctc.ckpt \\
          --alphabet origin_model/alphabet-all-v5.txt \\
          --output dist/mit48pxctc_ocr.onnx
        ```

        ## 许可

        本 ONNX 转换产物及配套文件按 GPL-3.0-only 分发。详见 `LICENSE`。

        上游项目使用 GPL-3.0 许可。上游作者和贡献者保留 manga-image-translator
        以及模型的原始作者身份和版权。来源署名和再分发授权说明见 `NOTICE`。
        """
    )


def build_metadata(
    archive_path: Path,
    checkpoint_path: Path,
    alphabet_path: Path,
    onnx_path: Path,
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": "mit48pxctc_ocr",
        "format": "onnx",
        "license": "GPL-3.0-only",
        "opset": 17,
        "source": {
            "project": SOURCE_PROJECT,
            "release": SOURCE_RELEASE,
            "release_url": SOURCE_RELEASE_URL,
            "archive_url": SOURCE_ARCHIVE_URL,
            "archive": archive_path.name,
            "checkpoint": checkpoint_path.name,
            "alphabet": alphabet_path.name,
            "archive_sha256": sha256(archive_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "alphabet_sha256": sha256(alphabet_path),
        },
        "artifact": {
            "onnx": "mit48pxctc_ocr.onnx",
            "onnx_sha256": sha256(onnx_path),
        },
        "export": {
            "command": (
                "uv run --extra export python scripts/export.py "
                "--checkpoint origin_model/ocr-ctc.ckpt "
                "--alphabet origin_model/alphabet-all-v5.txt "
                "--output dist/mit48pxctc_ocr.onnx"
            ),
            "dynamic_axes": ["batch", "width", "time"],
            "torch_version": package_version("torch"),
            "onnx_version": package_version("onnx"),
            "onnxruntime_version": package_version("onnxruntime"),
        },
        "validation": validation,
    }


def build_release(
    archive_path: Path,
    checkpoint_path: Path,
    alphabet_path: Path,
    onnx_path: Path,
    validation_metadata_path: Path,
    output_dir: Path,
) -> None:
    validation = load_json(validation_metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(onnx_path, output_dir / "mit48pxctc_ocr.onnx")
    shutil.copy2(alphabet_path, output_dir / "alphabet-all-v5.txt")
    shutil.copy2(Path("LICENSE"), output_dir / "LICENSE")
    shutil.copy2(Path("NOTICE"), output_dir / "NOTICE")
    (output_dir / "README.md").write_text(build_readme(validation), encoding="utf-8")
    (output_dir / "README.zh-CN.md").write_text(build_readme_zh(validation), encoding="utf-8")
    metadata = build_metadata(archive_path, checkpoint_path, alphabet_path, onnx_path, validation)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Hugging Face release folder.")
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alphabet", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("hf-release"))
    args = parser.parse_args()

    build_release(
        archive_path=args.source_archive,
        checkpoint_path=args.checkpoint,
        alphabet_path=args.alphabet,
        onnx_path=args.onnx,
        validation_metadata_path=args.validation_metadata,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
