import numpy as np

from wave_lossless_compression import rice


def test_zigzag_roundtrip():
    values = np.array([-3, -1, 0, 1, 3, 510, -510], dtype=np.int64)
    unsigned = rice.zigzag_encode(values)
    assert np.all(unsigned >= 0)
    restored = rice.zigzag_decode(unsigned)
    assert np.array_equal(restored, values)


def test_encode_decode_roundtrip_various_k():
    rng = np.random.default_rng(42)
    unsigned = rng.integers(0, 1000, size=500).astype(np.uint64)

    for k in range(0, 12):
        bits = rice.encode_block_bits(unsigned, k)
        packed = np.packbits(bits).tobytes()
        decoder = rice.BlockDecoder(packed)
        restored = decoder.decode(unsigned.size, k)
        assert np.array_equal(restored, unsigned), f"mismatch at k={k}"


def test_best_k_gives_minimal_or_near_minimal_cost():
    rng = np.random.default_rng(0)
    unsigned = rng.integers(0, 32, size=2000).astype(np.uint64)
    k, cost = rice.best_k(unsigned, max_k=10)

    costs = []
    for candidate in range(11):
        c = 2000 * (candidate + 1) + int((unsigned >> np.uint64(candidate)).sum())
        costs.append(c)
    assert cost == min(costs)
    assert k == int(np.argmin(costs))


def test_empty_block():
    empty = np.empty(0, dtype=np.uint64)
    k, cost = rice.best_k(empty)
    assert k == 0
    assert cost == 0
    bits = rice.encode_block_bits(empty, 0)
    assert bits.size == 0
