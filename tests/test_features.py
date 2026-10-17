import numpy as np
import pytest
from feature_extractor import assemble_features
from temporal import resample_sequence
from config import TOTAL_FEATURES_DIM


def points():
    rng = np.random.default_rng(4)
    return rng.uniform(0.3, 0.6, (21, 3)).astype(np.float32)


def test_translation_keeps_shape_but_preserves_trajectory():
    p = points()
    a, _ = assemble_features({"Right": p}, None, 0.75)
    b, _ = assemble_features({"Right": p + [0.2, 0.1, 0]}, None, 0.75)
    np.testing.assert_allclose(a[73:146], b[73:146], atol=1e-6)
    assert not np.allclose(a[149:152], b[149:152])


def test_relation_changes_when_hands_move_apart():
    p = points()
    a, _ = assemble_features({"Left": p, "Right": p + [0.1, 0, 0]}, None, 0.75)
    b, _ = assemble_features({"Left": p, "Right": p + [0.4, 0, 0]}, None, 0.75)
    assert b[152] > a[152]


def test_body_reference_distinguishes_articulation():
    p = points()
    pose = np.array(
        [[0.5, 0.1, 0], [0.48, 0.2, 0], [0.52, 0.2, 0], [0.3, 0.4, 0], [0.7, 0.4, 0]]
    )
    a, _ = assemble_features({"Right": p}, pose, 0.75)
    b, _ = assemble_features({"Right": p}, pose + [0, 0.2, 0], 0.75)
    assert a[155] == b[155] == 1
    assert not np.allclose(a[149:152], b[149:152])


def test_aspect_conversion_preserves_geometry():
    p = points()
    q = p.copy()
    q[:, 1] /= 2
    a, _ = assemble_features({"Right": p}, None, 1.0)
    b, _ = assemble_features({"Right": q}, None, 2.0)
    np.testing.assert_allclose(a[73:146], b[73:146], atol=1e-6)


def test_sequence_resampling_preserves_motion_across_fps():
    outputs = []
    for n in (11, 21):
        t = np.linspace(0, 2, n)
        x = np.zeros((n, TOTAL_FEATURES_DIM))
        x[:, 73] = 1
        x[:, 149] = t
        outputs.append(resample_sequence(x, t))
    np.testing.assert_allclose(*outputs, atol=1e-6)


@pytest.mark.parametrize(
    "times",
    [
        [0, 0.2, 0.4, 1.0, 1.5, 2.0],
        [0, 0.2, 0.4, 0.4, 1.0, 2.0],
        [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    ],
)
def test_discontinuous_or_short_sequences_rejected(times):
    with pytest.raises(ValueError):
        resample_sequence(np.zeros((6, TOTAL_FEATURES_DIM)), times)


def test_nonfinite_features_rejected():
    x = np.zeros((21, TOTAL_FEATURES_DIM))
    x[2, 0] = np.nan
    with pytest.raises(ValueError):
        resample_sequence(x, np.linspace(0, 2, 21))
