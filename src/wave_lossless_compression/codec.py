from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from . import rice
from .predictors import PredictorName, decode_residual, encode_residual

MAGIC = b"WLC1"

_PREDICTOR_TO_ID: dict[PredictorName, int] = {"raw": 0, "diff1": 1, "linear2": 2}
_ID_TO_PREDICTOR: dict[int, PredictorName] = {v: k for k, v in _PREDICTOR_TO_ID.items()}

_HEADER_FORMAT = ">4sII"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)


@dataclass
class CompressionStats:
    original_bytes: int
    compressed_bytes: int
    predictor_counts: dict[str, int]

    @property
    def ratio(self) -> float:
        """original_size / compressed_size, i.e. how many times smaller."""
        return self.original_bytes / self.compressed_bytes

    @property
    def bits_per_sample(self) -> float:
        return self.compressed_bytes * 8 / self.original_bytes


def _iter_blocks(n: int, block_size: int):
    for start in range(0, n, block_size):
        yield start, min(block_size, n - start)


def compress(
    data: np.ndarray,
    block_size: int,
    predictors: tuple[PredictorName, ...] = ("raw", "diff1", "linear2"),
) -> tuple[bytes, CompressionStats]:
    if data.dtype != np.uint8:
        raise TypeError("data must be a uint8 numpy array")
    if data.ndim != 1:
        raise ValueError("data must be 1-dimensional")

    n = data.size
    metadata = bytearray()
    bit_chunks: list[np.ndarray] = []
    predictor_counts = {name: 0 for name in predictors}

    for start, length in _iter_blocks(n, block_size):
        block = data[start : start + length]

        best_choice = None
        for predictor in predictors:
            residual = encode_residual(block, predictor)
            unsigned = rice.zigzag_encode(residual)
            k, cost = rice.best_k(unsigned)
            if best_choice is None or cost < best_choice[0]:
                best_choice = (cost, predictor, k, unsigned)

        _, predictor, k, unsigned = best_choice
        predictor_counts[predictor] += 1
        metadata.append((_PREDICTOR_TO_ID[predictor] << 6) | k)
        bit_chunks.append(rice.encode_block_bits(unsigned, k))

    all_bits = np.concatenate(bit_chunks) if bit_chunks else np.empty(0, dtype=np.uint8)
    packed = np.packbits(all_bits).tobytes()

    header = struct.pack(_HEADER_FORMAT, MAGIC, n, block_size)
    payload = header + bytes(metadata) + packed

    stats = CompressionStats(
        original_bytes=n,
        compressed_bytes=len(payload),
        predictor_counts=predictor_counts,
    )
    return payload, stats


def decompress(payload: bytes) -> np.ndarray:
    magic, n, block_size = struct.unpack_from(_HEADER_FORMAT, payload, 0)
    if magic != MAGIC:
        raise ValueError("not a wave-lossless-compression payload")

    offset = _HEADER_SIZE
    n_blocks = -(-n // block_size) if n else 0
    metadata = payload[offset : offset + n_blocks]
    offset += n_blocks

    decoder = rice.BlockDecoder(payload[offset:])
    output = np.empty(n, dtype=np.uint8)

    for block_index, (start, length) in enumerate(_iter_blocks(n, block_size)):
        meta_byte = metadata[block_index]
        predictor = _ID_TO_PREDICTOR[meta_byte >> 6]
        k = meta_byte & 0x3F

        unsigned = decoder.decode(length, k)
        residual = rice.zigzag_decode(unsigned).astype(np.int16)
        output[start : start + length] = decode_residual(residual, predictor)

    return output
