from __future__ import annotations

from typing import Literal

import numpy as np


PredictorName = Literal["raw", "diff1", "linear2"]


def _as_int16(block: np.ndarray) -> np.ndarray:
    if block.dtype != np.uint8:
        raise TypeError("block must be a uint8 numpy array")
    return block.astype(np.int16)


def raw_encode(block: np.ndarray) -> np.ndarray:
    return _as_int16(block)


def raw_decode(residual: np.ndarray) -> np.ndarray:
    return _clip_to_uint8(residual)


def diff1_encode(block: np.ndarray) -> np.ndarray:
    values = _as_int16(block)
    residual = np.empty(values.size, dtype=np.int16)
    if values.size == 0:
        return residual
    residual[0] = values[0]
    residual[1:] = values[1:] - values[:-1]
    return residual


def diff1_decode(residual: np.ndarray) -> np.ndarray:
    _validate_residual(residual)
    values = np.cumsum(residual.astype(np.int32))
    return _clip_to_uint8(values)


def linear2_encode(block: np.ndarray) -> np.ndarray:
    values = _as_int16(block)
    residual = np.empty(values.size, dtype=np.int16)
    if values.size == 0:
        return residual
    residual[0] = values[0]
    if values.size == 1:
        return residual
    residual[1] = values[1]
    prediction = 2 * values[1:-1].astype(np.int32) - values[:-2].astype(np.int32)
    residual[2:] = values[2:].astype(np.int32) - prediction
    return residual


def linear2_decode(residual: np.ndarray) -> np.ndarray:
    _validate_residual(residual)
    values = np.empty(residual.size, dtype=np.int32)
    if residual.size == 0:
        return values.astype(np.uint8)
    values[0] = residual[0]
    if residual.size == 1:
        return _clip_to_uint8(values)
    values[1] = residual[1]
    for index in range(2, residual.size):
        prediction = 2 * values[index - 1] - values[index - 2]
        values[index] = prediction + residual[index]
    return _clip_to_uint8(values)


def encode_residual(block: np.ndarray, predictor: PredictorName) -> np.ndarray:
    if predictor == "raw":
        return raw_encode(block)

    if predictor == "diff1":
        return diff1_encode(block)

    if predictor == "linear2":
        return linear2_encode(block)

    raise ValueError(f"unknown predictor: {predictor}")


def decode_residual(residual: np.ndarray, predictor: PredictorName) -> np.ndarray:
    if predictor == "raw":
        return raw_decode(residual)

    if predictor == "diff1":
        return diff1_decode(residual)

    if predictor == "linear2":
        return linear2_decode(residual)

    raise ValueError(f"unknown predictor: {predictor}")


def _validate_residual(residual: np.ndarray) -> None:
    if not np.issubdtype(residual.dtype, np.signedinteger):
        raise TypeError("residual must be a signed integer numpy array")


def _clip_to_uint8(values: np.ndarray) -> np.ndarray:
    if np.any((values < 0) | (values > 255)):
        raise ValueError("decoded values are outside the uint8 range")
    return values.astype(np.uint8)
