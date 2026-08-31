from __future__ import annotations

import numpy as np

MAX_K = 16


def zigzag_encode(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.int64)
    return ((values << 1) ^ (values >> 63)).astype(np.uint64)


def zigzag_decode(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64)
    signed = values.astype(np.int64)
    return (signed >> 1) ^ -(signed & 1)


def best_k(unsigned_values: np.ndarray, max_k: int = MAX_K) -> tuple[int, int]:
    """Return (k, cost_in_bits) minimizing the Rice code size for these values."""
    n = unsigned_values.size
    if n == 0:
        return 0, 0
    values = unsigned_values.astype(np.uint64)
    best_k_value = 0
    best_cost = None
    for k in range(max_k + 1):
        cost = n * (k + 1) + int(np.right_shift(values, np.uint64(k)).sum())
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_k_value = k
        elif cost > best_cost * 2:
            break
    return best_k_value, int(best_cost)


def encode_block_bits(unsigned_values: np.ndarray, k: int) -> np.ndarray:
    """Rice-encode values with parameter k into a flat array of 0/1 bits (uint8)."""
    n = unsigned_values.size
    if n == 0:
        return np.empty(0, dtype=np.uint8)

    values = unsigned_values.astype(np.uint64)
    q = (values >> np.uint64(k)).astype(np.int64)
    r = (values & np.uint64((1 << k) - 1)).astype(np.uint64) if k > 0 else None

    counts = (q + 1 + k).astype(np.int64)
    total = int(counts.sum())
    group_start = np.empty(n, dtype=np.int64)
    group_start[0] = 0
    np.cumsum(counts[:-1], out=group_start[1:])

    pos = np.arange(total, dtype=np.int64) - np.repeat(group_start, counts)
    q_rep = np.repeat(q, counts)

    bits = np.zeros(total, dtype=np.uint8)
    bits[pos < q_rep] = 1

    if k > 0:
        is_remainder = pos >= (q_rep + 1)
        rem_pos = pos[is_remainder] - (q_rep[is_remainder] + 1)
        shift = (k - 1 - rem_pos).astype(np.uint64)
        r_rep = np.repeat(r, counts)[is_remainder]
        bits[is_remainder] = ((r_rep >> shift) & np.uint64(1)).astype(np.uint8)

    return bits


class BlockDecoder:
    """Sequentially decodes Rice codes from a bitstream held as ASCII '0'/'1' bytes."""

    def __init__(self, payload: bytes) -> None:
        unpacked = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        self._ascii_bits = (unpacked + 48).astype(np.uint8).tobytes()
        self._pos = 0

    def decode(self, count: int, k: int) -> np.ndarray:
        values = np.empty(count, dtype=np.uint64)
        ascii_bits = self._ascii_bits
        pos = self._pos
        for i in range(count):
            zero_idx = ascii_bits.find(b"0", pos)
            q = zero_idx - pos
            pos = zero_idx + 1
            if k > 0:
                remainder = int(ascii_bits[pos : pos + k], 2)
                pos += k
            else:
                remainder = 0
            values[i] = (q << k) | remainder
        self._pos = pos
        return values
