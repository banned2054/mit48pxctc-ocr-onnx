from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


POSITIONAL_ENCODING_KEYS = (
    "encoders.layers.0.pe.pe",
    "encoders.layers.1.pe.pe",
    "encoders.layers.2.pe.pe",
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return x + self.pe[:, offset : offset + x.size(1), :]


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        batch_first: bool = False,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pe = PositionalEncoding(d_model, max_len=3072)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool | None = None,
    ) -> torch.Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.self_attn(
            self.pe(x),
            self.pe(x),
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return self.dropout2(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return out + residual


class ResNet(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, layers: list[int]) -> None:
        super().__init__()
        self.output_channel_block = [output_channel // 4, output_channel // 2, output_channel, output_channel]
        self.inplanes = output_channel // 8
        self.conv0_1 = nn.Conv2d(input_channel, output_channel // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(output_channel // 8)
        self.conv0_2 = nn.Conv2d(output_channel // 8, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.maxpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(self.output_channel_block[0], layers[0])
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[0], kernel_size=3, padding=1, bias=False)

        self.maxpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(self.output_channel_block[1], layers[1])
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[1], kernel_size=3, padding=1, bias=False)

        self.maxpool3 = nn.AvgPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(self.output_channel_block[2], layers[2])
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[2], kernel_size=3, padding=1, bias=False)

        self.layer4 = self._make_layer(self.output_channel_block[3], layers[3])
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4_3 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.BatchNorm2d(self.inplanes),
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
            )

        layers: list[nn.Module] = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0_2(F.relu(self.bn0_1(self.conv0_1(x))))
        x = self.conv1(F.relu(self.bn1(self.layer1(self.maxpool1(x)))))
        x = self.conv2(F.relu(self.bn2(self.layer2(self.maxpool2(x)))))
        x = self.conv3(F.relu(self.bn3(self.layer3(self.maxpool3(x)))))
        x = self.layer4(x)
        x = self.conv4_1(F.relu(self.bn4_1(x)))
        x = self.conv4_2(F.relu(self.bn4_2(x)))
        return self.bn4_3(x)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, input_channel: int, output_channel: int = 128) -> None:
        super().__init__()
        self.ConvNet = ResNet(input_channel, output_channel, [4, 6, 8, 6, 3])

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.ConvNet(input_tensor)


class OCR(nn.Module):
    def __init__(self, dictionary: list[str], max_len: int = 768) -> None:
        super().__init__()
        self.max_len = max_len
        self.dictionary = dictionary
        self.dict_size = len(dictionary)
        self.backbone = ResNetFeatureExtractor(3, 320)
        enc = CustomTransformerEncoderLayer(320, 8, 320 * 4, dropout=0.05, batch_first=True, norm_first=True)
        self.encoders = nn.TransformerEncoder(enc, 3)
        self.char_pred_norm = nn.Sequential(nn.LayerNorm(320), nn.Dropout(0.1), nn.GELU())
        self.char_pred = nn.Linear(320, self.dict_size)
        self.color_pred1 = nn.Sequential(nn.Linear(320, 6))

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(img).squeeze(2)
        feats = self.encoders(feats.permute(0, 2, 1))
        pred_char_logits = self.char_pred(self.char_pred_norm(feats))
        pred_color_values = self.color_pred1(feats)
        return pred_char_logits, pred_color_values


def load_dictionary(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fp:
        return [line.rstrip("\n") for line in fp]


def load_model(checkpoint_path: Path, alphabet_path: Path) -> OCR:
    dictionary = load_dictionary(alphabet_path)
    model = OCR(dictionary, 768)
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(state)
    for key in POSITIONAL_ENCODING_KEYS:
        state_dict.pop(key, None)
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if key not in POSITIONAL_ENCODING_KEYS]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint does not match OCR model. missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def _extract_state_dict(state: Any) -> OrderedDict[str, torch.Tensor]:
    raw_state = state["model"] if isinstance(state, dict) and "model" in state else state
    if not isinstance(raw_state, OrderedDict) and not isinstance(raw_state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(raw_state)!r}")
    return OrderedDict(raw_state)
