import numpy as np

from universal_segmenter.inference.postprocess import apply_postprocess


def test_largest_non_border_prefers_complete_target_over_border_distractor():
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, :25] = True
    mask[25:85, 35:95] = True
    result = apply_postprocess(mask, "largest_non_border")
    assert not result[:, :25].any()
    assert result[25:85, 35:95].all()


def test_raw_policy_preserves_components():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:5, 2:5] = True
    mask[10:15, 10:15] = True
    assert np.array_equal(apply_postprocess(mask, "raw"), mask)

