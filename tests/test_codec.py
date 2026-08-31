import numpy as np
import pytest

from wave_lossless_compression import codec

PREDICTOR_SETS = [
    ("raw",),
    ("diff1",),
    ("linear2",),
    ("raw", "diff1", "linear2"),
]


@pytest.mark.parametrize("predictors", PREDICTOR_SETS)
@pytest.mark.parametrize("block_size", [1, 7, 64, 4096])
def test_roundtrip_random_data(predictors, block_size):
    rng = np.random.default_rng(7)
    data = rng.integers(0, 256, size=10_000, dtype=np.uint8)

    payload, stats = codec.compress(data, block_size=block_size, predictors=predictors)
    restored = codec.decompress(payload)

    assert np.array_equal(data, restored)
    assert stats.original_bytes == data.size
    assert stats.compressed_bytes == len(payload)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 17, 4097])
def test_roundtrip_edge_sizes(n):
    data = (np.arange(n) % 256).astype(np.uint8)
    payload, stats = codec.compress(data, block_size=8)
    restored = codec.decompress(payload)
    assert np.array_equal(data, restored)


def test_roundtrip_smooth_signal_compresses_well():
    t = np.linspace(0, 4 * np.pi, 20_000)
    data = (128 + 100 * np.sin(t)).astype(np.uint8)

    payload, stats = codec.compress(data, block_size=1024)
    restored = codec.decompress(payload)

    assert np.array_equal(data, restored)
    assert stats.ratio > 1.0


def test_invalid_dtype_rejected():
    with pytest.raises(TypeError):
        codec.compress(np.zeros(10, dtype=np.int16), block_size=4)


def test_bad_magic_rejected():
    with pytest.raises(ValueError):
        codec.decompress(b"not-a-valid-payload-------")
