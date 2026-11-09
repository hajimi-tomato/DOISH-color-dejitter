from dataclasses import asdict
import cv2
import numpy as np
import pytest

from color_dejitter import core


@pytest.fixture
def noisy():
    rng = np.random.default_rng(19)
    image = np.full((96, 96, 3), [110, 145, 190], dtype=np.float32)
    return np.clip(image + rng.normal(0, 6, image.shape), 0, 255).astype(np.uint8)


def test_zero_strength_is_identity_and_never_loads_model(noisy, monkeypatch):
    def forbidden(*args):
        raise AssertionError('zero strength must not run segmentation')
    monkeypatch.setattr(core, 'get_foreground_mask', forbidden)
    result, diff = core.color_dejitter(noisy, overall_strength=0, s_enabled=True, v_enabled=True)
    np.testing.assert_array_equal(result, noisy)
    assert not diff.any()


def test_zero_stages_are_identity(noisy):
    result, _ = core.color_dejitter(noisy, h_denoise_strength=0, s_enabled=True, s_denoise_strength=0, v_enabled=True, v_denoise_strength=0)
    np.testing.assert_array_equal(result, noisy)


def test_chroma_noise_reduced_and_luminance_preserved():
    rng = np.random.default_rng(42)
    lab = np.full((96, 96, 3), [60, 12, 18], np.float32)
    lab[..., 1:] += rng.normal(0, 4, (96, 96, 2))
    image = np.clip(np.rint(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)*255), 0, 255).astype(np.uint8)
    result, _ = core.color_dejitter(image, overall_strength=1)
    before = cv2.cvtColor(image.astype(np.float32)/255, cv2.COLOR_BGR2Lab)
    after = cv2.cvtColor(result.astype(np.float32)/255, cv2.COLOR_BGR2Lab)
    assert after[..., 1:].std(axis=(0, 1)).mean() < before[..., 1:].std(axis=(0, 1)).mean() * 0.65
    assert np.abs(after[..., 0]-before[..., 0]).mean() < 0.3


@pytest.mark.parametrize('name,value', [
    ('h_smooth_radius', 3), ('h_denoise_threshold', 2), ('h_denoise_strength', 0),
    ('h_use_soft_threshold', False), ('s_enabled', False), ('s_smooth_radius', 3),
    ('s_denoise_threshold', 0.2), ('s_denoise_strength', 0), ('s_use_soft_threshold', False),
    ('v_enabled', False), ('v_smooth_radius', 3), ('v_denoise_threshold', 0.015),
    ('v_denoise_strength', 0), ('v_use_soft_threshold', False),
    ('edge_protection', 0), ('low_sat_skip', 1), ('overall_strength', 0.2),
])
def test_controls_have_observable_effect(noisy, name, value):
    # Add a real edge and a color ramp so spatial/range controls have something to preserve.
    image = noisy.copy()
    image[:, 48:, 0] += 25
    image[..., 2] = np.clip(image[..., 2].astype(float) + np.linspace(-10, 10, 96)[None], 0, 255)
    args = asdict(core.Settings(s_enabled=True, v_enabled=True, background_enabled=False, overall_strength=1))
    base, _ = core.color_dejitter(image, **args)
    args[name] = value
    changed, _ = core.color_dejitter(image, **args)
    assert np.any(base != changed), f'{name} had no effect'


def test_background_toggle_and_global_strength(noisy):
    bg = np.zeros(noisy.shape[:2], np.uint8)
    # Known flat background with low uncorrelated noise.
    rng = np.random.default_rng(3)
    image = np.clip(np.full((192, 192, 3), [110,145,190]) + rng.normal(0, 1, (192,192,3)),0,255).astype(np.uint8)
    bg = np.zeros(image.shape[:2], np.uint8)
    args = dict(h_denoise_strength=0, v_enabled=True, v_denoise_strength=1, v_use_soft_threshold=False, foreground_mask=bg, report=lambda _: None)
    full, _ = core.color_dejitter(image, overall_strength=1, **args)
    half, _ = core.color_dejitter(image, overall_strength=0.5, **args)
    zero, _ = core.color_dejitter(image, overall_strength=0, **args)
    without, _ = core.color_dejitter(image, overall_strength=1, background_enabled=False, **args)
    np.testing.assert_array_equal(zero, image)
    np.testing.assert_array_equal(half, cv2.addWeighted(image, 0.5, full, 0.5, 0))
    assert np.any(full != without)
    assert len(np.unique(full.reshape(-1,3), axis=0)) == 1


def test_subtle_gradient_is_not_flattened():
    gray = np.tile(np.linspace(120,140,96,dtype=np.uint8), (96,1))
    image = np.repeat(gray[...,None],3,axis=2)
    messages = []
    args = dict(v_enabled=True, v_denoise_strength=1, foreground_mask=np.zeros((96,96),np.uint8), report=messages.append)
    with_bg,_ = core.color_dejitter(image, **args)
    without_bg,_ = core.color_dejitter(image, background_enabled=False, **args)
    np.testing.assert_array_equal(with_bg, without_bg)
    assert len(np.unique(with_bg[...,0])) > 10
    assert any('渐变' in text for text in messages)


def test_disabled_v_does_not_request_segmentation(noisy, monkeypatch):
    monkeypatch.setattr(core, 'get_foreground_mask', lambda _: pytest.fail('V disabled'))
    core.color_dejitter(noisy, v_enabled=False)


def test_missing_model_reports_fallback(noisy, monkeypatch):
    from color_dejitter.segment import SegmentationUnavailable
    def missing(_):
        raise SegmentationUnavailable('missing')
    monkeypatch.setattr(core, 'get_foreground_mask', missing)
    messages = []
    result,_ = core.color_dejitter(noisy, v_enabled=True, report=messages.append)
    assert result.shape == noisy.shape
    assert any('已跳过' in text for text in messages)


def test_nlm_tile_seams_match_full_image():
    rng = np.random.default_rng(11)
    image = np.clip(128+rng.normal(0,6,(83,91)),0,255).astype(np.uint8)
    tiled = core._nlm(image, 8, 5, 5, tile=24)
    full = cv2.fastNlMeansDenoising(image, None, 8, 5, 11)
    assert np.abs(tiled.astype(int)-full.astype(int)).max() <= 1


@pytest.mark.parametrize('args', [{'overall_strength':float('nan')}, {'h_denoise_strength':80}, {'v_enabled':'false'}, {'h_smooth_radius':2.5}])
def test_invalid_settings_rejected(noisy, args):
    with pytest.raises(ValueError):
        core.color_dejitter(noisy, **args)


def test_invalid_pixels_are_unchanged(noisy):
    valid = np.ones(noisy.shape[:2],bool)
    valid[:10] = False
    out,_ = core.color_dejitter(noisy, valid_mask=valid)
    np.testing.assert_array_equal(out[:10], noisy[:10])
