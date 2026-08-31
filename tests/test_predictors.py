import numpy as np
import pytest

from wave_lossless_compression.predictors import (
    PredictorName,
    decode_residual,
    encode_residual,
)

PREDICTORS: tuple[PredictorName, ...] = ("raw", "diff1", "linear2")


@pytest.mark.parametrize("predictor", PREDICTORS)
def test_roundtrip_random_block(predictor: PredictorName):
    rng = np.random.default_rng(1)
    block = rng.integers(0, 256, size=1000, dtype=np.uint8)
    residual = encode_residual(block, predictor)
    restored = decode_residual(residual, predictor)
    assert np.array_equal(restored, block)


@pytest.mark.parametrize("predictor", PREDICTORS)
@pytest.mark.parametrize("size", [0, 1, 2, 3])
def test_roundtrip_edge_sizes(predictor: PredictorName, size: int):
    block = np.arange(size, dtype=np.uint8)
    residual = encode_residual(block, predictor)
    restored = decode_residual(residual, predictor)
    assert np.array_equal(restored, block)


def test_unknown_predictor_raises():
    block = np.zeros(4, dtype=np.uint8)
    with pytest.raises(ValueError):
        encode_residual(block, "bogus")  # type: ignore[arg-type]
