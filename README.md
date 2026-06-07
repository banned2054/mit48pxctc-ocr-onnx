# MIT 48px CTC OCR ONNX

This repository keeps the ONNX export workflow, validation scripts, and model
contract for the 48px CTC OCR model used by manga-image-translator.

The Python code is intended to make the converted ONNX artifact reproducible:

- keep the PyTorch architecture needed for export;
- export `ocr-ctc.ckpt` to ONNX;
- validate ONNX Runtime outputs against PyTorch;
- document the raw output contract and CTC post-processing.

## Upstream

- Upstream project: https://github.com/zyddnys/manga-image-translator
- Upstream release: https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3
- Source model archive: `ocr-ctc.zip`
- Source checkpoint inside archive: `ocr-ctc.ckpt`
- Source alphabet inside archive: `alphabet-all-v5.txt`

The exported ONNX model is derived from the upstream checkpoint and therefore
must be distributed with the attribution and license notices in `NOTICE`.

## Files

Expected public release assets:

```text
mit48pxctc_ocr.onnx
alphabet-all-v5.txt
metadata.json
LICENSE
NOTICE
```

`metadata.json` should record at least the source archive URL, source file
hashes, export command, opset, PyTorch version, ONNX Runtime version, and
validation widths. Downstream ONNX consumers do not need the Python export
source unless they want to reproduce or verify the conversion.

## Export

Install the export dependencies in a fresh environment:

```powershell
uv sync --extra export
```

Export from the upstream checkpoint and alphabet:

```powershell
uv run python scripts/export.py `
  --checkpoint path\to\ocr-ctc.ckpt `
  --alphabet path\to\alphabet-all-v5.txt `
  --output dist\mit48pxctc_ocr.onnx
```

By default the ONNX input has dynamic batch and width axes. Use `--fixed-width`
when a fixed input width is required by the target runtime.

This command produces the ONNX output contract used by the .NET runtime
integration.

## Validate

```powershell
uv run python scripts/validate.py `
  --checkpoint path\to\ocr-ctc.ckpt `
  --alphabet path\to\alphabet-all-v5.txt `
  --onnx dist\mit48pxctc_ocr.onnx `
  --widths 512 1024 1536
```

The validation script runs `onnx.checker`, prints the ONNX input/output schema,
and compares ONNX Runtime output tensors against PyTorch output tensors.

## Publish with GitHub Actions

The workflow `.github/workflows/publish-hf.yml` can reproduce the public ONNX
release from source:

1. download the upstream `ocr-ctc.zip` release asset;
2. extract `ocr-ctc.ckpt` and `alphabet-all-v5.txt`;
3. export `dist/mit48pxctc_ocr.onnx`;
4. validate ONNX Runtime outputs against PyTorch outputs;
5. generate the Hugging Face release folder;
6. upload the folder to Hugging Face.

Create a GitHub repository secret named `HF_TOKEN` with a Hugging Face token
that has write access to the target model repository.

By default the workflow uploads to:

```text
banned404/mit48pxctc-ocr-onnx
```

The workflow runs on relevant pushes to `main` and can also be started manually
from the GitHub Actions tab. Manual runs allow overriding the Hugging Face repo
ID.

## Input Contract

The ONNX model accepts one input:

- name: `image`
- dtype: `float32`
- shape: `[batch, 3, 48, width]`
- color order: BGR for OpenCV-style preprocessing
- normalization: `(uint8_pixel - 127.5) / 127.5`

The original inference path pads each text-line image to a common batch width.
For compatibility, resize/crop text line regions to height 48, pad width with
black pixels, then normalize. In normalized input space, padded pixels are `-1.0`.
The original implementation keeps width below roughly 8100 pixels; very large widths may exceed positional
encoding limits or runtime memory.

## Output Contract

The ONNX model returns raw tensors from the model forward pass:

- `char_logits`: `[batch, time, vocab_size]`
- `color_values`: `[batch, time, 6]`

`char_logits` is not softmaxed. `color_values` is not clamped. Use
`mit48pxctc_ocr_onnx.postprocess.decode_ctc_top1` for the same basic top-1 CTC
collapse used by the PyTorch implementation.

The first dictionary entry is the CTC blank token. The special token `<SP>` is
converted to a normal space by the helper.

## ONNX Runtime Example (.NET)

The model can be used directly from ONNX Runtime. This example assumes the input
image is already cropped to a single text-line region. For full-page OCR, run a
text detector first, crop each detected line, then feed the line images in
batches.

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
// Foreground/background RGB predictions for each timestep.
var colorValues = outputArray[1].AsTensor<float>();

Console.WriteLine(DecodeCtcTop1(charLogits, dictionary));

static Mat ResizeToHeight(Mat source, int height)
{
    if (source.Empty())
    {
        return new Mat(height, height, MatType.CV_8UC3, Scalar.Black);
    }

    var bgr = source.Channels() switch
    {
        3 => source.Clone(),
        4 => ConvertColor(source, ColorConversionCodes.BGRA2BGR),
        _ => ConvertColor(source, ColorConversionCodes.GRAY2BGR)
    };

    if (bgr.Height == height)
    {
        return bgr;
    }

    var width = Math.Max(1, (int)MathF.Round(height / (float)bgr.Height * bgr.Width));
    var resized = new Mat();
    Cv2.Resize(bgr, resized, new Size(width, height));
    bgr.Dispose();
    return resized;
}

static Mat ConvertColor(Mat source, ColorConversionCodes code)
{
    var converted = new Mat();
    Cv2.CvtColor(source, converted, code);
    return converted;
}

static string DecodeCtcTop1(Tensor<float> logits, string[] dictionary)
{
    var batch = logits.Dimensions[0];
    var time = logits.Dimensions[1];
    var classCount = logits.Dimensions[2];
    if (batch != 1)
    {
        throw new NotSupportedException("This sample decodes one line at a time.");
    }

    var text = new StringBuilder();
    var lastToken = BlankId;
    for (var t = 0; t < time; t++)
    {
        var token = ArgMax(logits, t, classCount);
        if (token != lastToken && token != BlankId)
        {
            var value = dictionary[token];
            text.Append(value == "<SP>" ? " " : value);
        }

        lastToken = token;
    }

    return text.ToString();
}

static int ArgMax(Tensor<float> logits, int time, int classCount)
{
    var bestIndex = 0;
    var bestValue = logits[0, time, 0];
    for (var i = 1; i < classCount; i++)
    {
        var value = logits[0, time, i];
        if (value > bestValue)
        {
            bestValue = value;
            bestIndex = i;
        }
    }

    return bestIndex;
}
```

`color_values` has six values per timestep: foreground RGB followed by
background RGB. Clamp each value to `0..1` and multiply by `255` when converting
to byte colors.

## License

This project is licensed under GPL-3.0-only. See `LICENSE`.

The upstream project is GPL-3.0 licensed. Additional model redistribution
authorization from the upstream author should be preserved with this repository
or release. See `NOTICE`.

