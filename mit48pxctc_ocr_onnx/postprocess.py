from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DecodedChar:
    text: str
    logprob: float
    fg: tuple[int, int, int]
    bg: tuple[int, int, int]
    timestep: int
    token_id: int


def load_dictionary(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fp:
        return [line.rstrip("\n") for line in fp]


def decode_ctc_top1(
    char_logits: np.ndarray,
    color_values: np.ndarray,
    dictionary: list[str],
    blank: int = 0,
) -> list[list[DecodedChar]]:
    if char_logits.ndim != 3:
        raise ValueError(f"char_logits must have shape [batch, time, vocab], got {char_logits.shape}")
    if color_values.ndim != 3 or color_values.shape[:2] != char_logits.shape[:2] or color_values.shape[2] != 6:
        raise ValueError(f"color_values must have shape [batch, time, 6], got {color_values.shape}")

    logprobs = _log_softmax(char_logits, axis=2)
    pred_ids = np.argmax(logprobs, axis=2)
    colors = np.clip(color_values, 0.0, 1.0)

    decoded: list[list[DecodedChar]] = []
    for batch_index in range(char_logits.shape[0]):
        last_token = blank
        line: list[DecodedChar] = []
        for timestep in range(char_logits.shape[1]):
            token_id = int(pred_ids[batch_index, timestep])
            if token_id != last_token and token_id != blank:
                token = dictionary[token_id]
                text = " " if token == "<SP>" else token
                rgb = (colors[batch_index, timestep] * 255.0).astype(np.int32)
                line.append(
                    DecodedChar(
                        text=text,
                        logprob=float(logprobs[batch_index, timestep, token_id]),
                        fg=(int(rgb[0]), int(rgb[1]), int(rgb[2])),
                        bg=(int(rgb[3]), int(rgb[4]), int(rgb[5])),
                        timestep=timestep,
                        token_id=token_id,
                    )
                )
            last_token = token_id
        decoded.append(line)
    return decoded


def _log_softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
