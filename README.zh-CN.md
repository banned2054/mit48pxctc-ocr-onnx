# MIT 48px CTC OCR ONNX

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-banned404%2Fmit48pxctc--ocr--onnx-yellow?logo=huggingface)](https://huggingface.co/banned404/mit48pxctc-ocr-onnx)
[![GitHub](https://img.shields.io/badge/GitHub-banned2054%2Fmit48pxctc--ocr--onnx-black?logo=github)](https://github.com/banned2054/mit48pxctc-ocr-onnx)
[![Publish ONNX to Hugging Face](https://github.com/banned2054/mit48pxctc-ocr-onnx/actions/workflows/publish-hf.yml/badge.svg)](https://github.com/banned2054/mit48pxctc-ocr-onnx/actions/workflows/publish-hf.yml)

[English](README.md)

本仓库保存 manga-image-translator 使用的 48px CTC OCR 模型的 ONNX 导出流程、验证脚本和模型接口约定。

这些 Python 代码用于让转换后的 ONNX 产物可以复现：

- 保留导出所需的 PyTorch 模型结构；
- 将 `ocr-ctc.ckpt` 导出为 ONNX；
- 对比 ONNX Runtime 与 PyTorch 的输出；
- 文档化原始输出约定和 CTC 后处理。

## 上游来源

- 上游项目：https://github.com/zyddnys/manga-image-translator
- 上游发布：https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3
- 源模型压缩包：`ocr-ctc.zip`
- 压缩包内 checkpoint：`ocr-ctc.ckpt`
- 压缩包内字典：`alphabet-all-v5.txt`

导出的 ONNX 模型派生自上游 checkpoint，因此分发时必须同时保留 `NOTICE` 中的署名和许可说明。

## 文件

公开发布产物应包含：

```text
mit48pxctc_ocr.onnx
alphabet-all-v5.txt
metadata.json
LICENSE
NOTICE
```

`metadata.json` 至少应记录源压缩包 URL、源文件哈希、导出命令、opset、PyTorch 版本、ONNX Runtime 版本和验证宽度。下游 ONNX 使用者通常不需要 Python 导出源码，除非他们想复现或验证转换过程。

## 导出

在干净环境中安装导出依赖：

```powershell
uv sync --extra export
```

从上游 checkpoint 和字典导出：

```powershell
uv run python scripts/export.py `
  --checkpoint path\to\ocr-ctc.ckpt `
  --alphabet path\to\alphabet-all-v5.txt `
  --output dist\mit48pxctc_ocr.onnx
```

默认情况下，ONNX 输入带有动态 batch 和 width 轴。如果目标运行时需要固定输入宽度，可以使用 `--fixed-width`。

这个命令会生成 .NET 运行时集成所需的 ONNX 输出约定。

## 验证

```powershell
uv run python scripts/validate.py `
  --checkpoint path\to\ocr-ctc.ckpt `
  --alphabet path\to\alphabet-all-v5.txt `
  --onnx dist\mit48pxctc_ocr.onnx `
  --widths 512 1024 1536
```

验证脚本会运行 `onnx.checker`，打印 ONNX 输入/输出 schema，并将 ONNX Runtime 输出张量与 PyTorch 输出张量做对比。

## 通过 GitHub Actions 发布

`.github/workflows/publish-hf.yml` 可以从源文件复现公开 ONNX 发布：

1. 下载上游 `ocr-ctc.zip` release asset；
2. 解出 `ocr-ctc.ckpt` 和 `alphabet-all-v5.txt`；
3. 导出 `dist/mit48pxctc_ocr.onnx`；
4. 对比 ONNX Runtime 与 PyTorch 输出；
5. 生成 Hugging Face 发布目录；
6. 上传到 Hugging Face。

在 GitHub 仓库中创建名为 `HF_TOKEN` 的 repository secret，内容为拥有目标模型仓库写入权限的 Hugging Face token。

默认上传目标：

```text
banned404/mit48pxctc-ocr-onnx
```

该 workflow 会在 `main` 的相关文件发生 push 时运行，也可以从 GitHub Actions 页面手动启动。手动运行时可以覆盖 Hugging Face repo ID。

## 输入约定

ONNX 模型接受一个输入：

- name: `image`
- dtype: `float32`
- shape: `[batch, 3, 48, width]`
- color order: BGR，用于 OpenCV 风格预处理
- normalization: `(uint8_pixel - 127.5) / 127.5`

原始推理流程会把每个文本行图像 pad 到同一 batch 宽度。为了兼容，先将文本行区域 resize/crop 到高度 48，再用黑色像素 pad 宽度，然后归一化。在归一化后的输入空间中，pad 像素为 `-1.0`。原始实现会把宽度控制在大约 8100 像素以下；过大的宽度可能超过位置编码限制或运行时内存。

## 输出约定

ONNX 模型返回原始 forward 输出张量：

- `char_logits`: `[batch, time, vocab_size]`
- `color_values`: `[batch, time, 6]`

`char_logits` 未经过 softmax。`color_values` 未经过 clamp。可以使用 `mit48pxctc_ocr_onnx.postprocess.decode_ctc_top1` 得到与 PyTorch 实现一致的基础 top-1 CTC 折叠结果。

字典第一项是 CTC blank token。特殊 token `<SP>` 会被辅助函数转换为空格。

## ONNX Runtime 示例（.NET）

模型可以直接通过 ONNX Runtime 使用。下面示例假设输入图像已经裁剪为单行文本区域。对于整页 OCR，需要先运行文本检测器，裁剪每个文本行，再批量输入模型。

```csharp
using System;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

const int TextHeight = 48;
const int MaxWidth = 8100;
const int ExtraPad = 135;
const int BlankId = 0;

using var session = new InferenceSession("mit48pxctc_ocr.onnx");
var inputName = session.InputMetadata.Keys.First();
var dictionary = File.ReadAllLines("alphabet-all-v5.txt")
                     .Select(line => line.TrimEnd('\r', '\n'))
                     .ToArray();

using var source = Cv2.ImRead("text-line.png", ImreadModes.Color);
using var line = ResizeToHeight(source, TextHeight);

var contentWidth = Math.Min(line.Width, MaxWidth);
var inputWidth = contentWidth + ExtraPad;
var input = new DenseTensor<float>(new[] { 1, 3, TextHeight, inputWidth });
input.Buffer.Span.Fill(-1f);

for (var y = 0; y < TextHeight; y++)
{
    for (var x = 0; x < contentWidth; x++)
    {
        var pixel = line.At<Vec3b>(y, x);
        input[0, 0, y, x] = (pixel.Item0 - 127.5f) / 127.5f;
        input[0, 1, y, x] = (pixel.Item1 - 127.5f) / 127.5f;
        input[0, 2, y, x] = (pixel.Item2 - 127.5f) / 127.5f;
    }
}

using var outputs = session.Run(new[]
{
    NamedOnnxValue.CreateFromTensor(inputName, input)
});

var outputArray = outputs.ToArray();
var charLogits = outputArray[0].AsTensor<float>();
var colorValues = outputArray[1].AsTensor<float>();

Console.WriteLine(DecodeCtcTop1(charLogits, dictionary));
```

`color_values` 每个 timestep 有六个值：前景 RGB 和背景 RGB。转换为 byte 颜色时，先将每个值 clamp 到 `0..1`，再乘以 `255`。

## 许可

本项目使用 GPL-3.0-only 许可。详见 `LICENSE`。

上游项目使用 GPL-3.0 许可。来自上游作者的额外模型再分发授权应随本仓库或发布材料一起保存。详见 `NOTICE`。
