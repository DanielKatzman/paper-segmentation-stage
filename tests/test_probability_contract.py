import numpy as np
import pytest

from universal_segmenter.inference.cpu_runtime import run_probability
from universal_segmenter.inference.preprocess import (
    LetterboxTransform,
    restore_probability,
)


class _Input:
    name = "image"


class _Session:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [_Input()]

    def run(self, _output_names, _inputs):
        return [self.output]


def test_onnx_roundoff_is_clipped_to_probability_contract():
    output = np.asarray(
        [[[[ -1.0e-7, 0.25], [0.75, 1.0 + 1.0e-7]]]],
        dtype=np.float32,
    )
    result = run_probability(
        _Session(output), np.zeros((1, 3, 2, 2), dtype=np.float32))
    assert result.dtype == np.float32
    assert float(result.min()) == 0.0
    assert float(result.max()) == 1.0


def test_materially_invalid_onnx_probability_is_rejected():
    output = np.full((1, 1, 2, 2), 1.01, dtype=np.float32)
    with pytest.raises(ValueError, match="outside the permitted numerical range"):
        run_probability(
            _Session(output), np.zeros((1, 3, 2, 2), dtype=np.float32))


def test_probability_restoration_preserves_strict_range():
    probability = np.asarray(
        [[-1.0e-7, 0.25], [0.75, 1.0 + 1.0e-7]], dtype=np.float32)
    transform = LetterboxTransform(
        original_hw=(7, 9),
        canvas_hw=(2, 2),
        left=0,
        top=0,
        resized_width=2,
        resized_height=2,
    )
    result = restore_probability(probability, transform)
    assert result.shape == (7, 9)
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
