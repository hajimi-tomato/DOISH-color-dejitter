"""Conservative Lab denoising. Public strengths are fractions (0..1)."""
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Tuple
import warnings

import cv2
import numpy as np


@dataclass(frozen=True)
class Settings:
    overall_strength: float = 0.75
    h_smooth_radius: int = 15
    h_denoise_threshold: float = 8.0
    h_denoise_strength: float = 0.8
    h_use_soft_threshold: bool = True
    s_enabled: bool = False
    s_smooth_radius: int = 15
    s_denoise_threshold: float = 0.05
    s_denoise_strength: float = 0.8
    s_use_soft_threshold: bool = True
    v_enabled: bool = False
    v_smooth_radius: int = 15
    v_denoise_threshold: float = 0.05
    v_denoise_strength: float = 0.5
    v_use_soft_threshold: bool = True
    edge_protection: float = 50.0
    low_sat_skip: float = 0.1
    background_enabled: bool = True

    def validated(self):
        limits = {
            'overall_strength': (0, 1), 'h_smooth_radius': (1, 30),
            'h_denoise_threshold': (0, 40), 'h_denoise_strength': (0, 1),
            's_smooth_radius': (1, 60), 's_denoise_threshold': (0, 0.4),
            's_denoise_strength': (0, 1), 'v_smooth_radius': (1, 30),
            'v_denoise_threshold': (0, 0.2), 'v_denoise_strength': (0, 1),
            'edge_protection': (0, 200), 'low_sat_skip': (0, 1),
        }
        values = asdict(self)
        for name, (lo, hi) in limits.items():
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'{name} 必须是有限数值')
            if not lo <= value <= hi:
                raise ValueError(f'{name} 必须在 {lo}–{hi} 之间')
            if name.endswith('_radius'):
                if int(value) != value:
                    raise ValueError(f'{name} 必须是整数')
                values[name] = int(value)
        for name in values.keys() - limits.keys():
            if not isinstance(values[name], bool):
                raise ValueError(f'{name} 必须是布尔值')
        return Settings(**values)


def _nlm(channel, strength, radius, template, tile=512):
    """Search/template halos keep tile boundaries out of retained pixels."""
    if strength <= 0:
        return channel.copy()
    channel = np.ascontiguousarray(channel)
    h, w = channel.shape
    out = np.empty_like(channel)
    halo = radius + template // 2
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y1, x1 = min(y + tile, h), min(x + tile, w)
            ya, xa = max(0, y - halo), max(0, x - halo)
            yb, xb = min(h, y1 + halo), min(w, x1 + halo)
            patch = cv2.fastNlMeansDenoising(channel[ya:yb, xa:xb], None, float(strength), template, radius * 2 + 1)
            out[y:y1, x:x1] = patch[y-ya:y1-ya, x-xa:x1-xa]
    return out


def _gate(delta, threshold, soft):
    magnitude = np.linalg.norm(delta, axis=-1, keepdims=True) if delta.ndim == 3 else np.abs(delta)
    limit = max(float(threshold), 1e-6)
    if soft:
        return 1.0 / (1.0 + (magnitude / limit) ** 2)
    return (magnitude <= limit).astype(np.float32)


def _flat_background(lab, background):
    """Inspect ORIGINAL image cell medians to reject coherent gradients/shadows."""
    h, w = background.shape
    if np.count_nonzero(background) < max(64, h * w * 0.005):
        return False
    if np.any(np.std(lab[background], axis=0) > [5.0, 4.0, 4.0]):
        return False
    medians = []
    ys, xs = np.linspace(0, h, 7, dtype=int), np.linspace(0, w, 7, dtype=int)
    for ya, yb in zip(ys[:-1], ys[1:]):
        for xa, xb in zip(xs[:-1], xs[1:]):
            selected = lab[ya:yb, xa:xb][background[ya:yb, xa:xb]]
            if len(selected) >= 16:
                medians.append(np.median(selected, axis=0))
    if len(medians) < 4:
        return False
    return bool(np.all(np.ptp(medians, axis=0) <= [0.8, 1.0, 1.0]))


def get_foreground_mask(image_bgr):
    from .segment import foreground_mask
    return foreground_mask(image_bgr)


def color_dejitter(
    image_bgr: np.ndarray,
    overall_strength: float = 0.75,
    h_smooth_radius: int = 15,
    h_denoise_threshold: float = 8.0,
    h_denoise_strength: float = 0.8,
    h_use_soft_threshold: bool = True,
    s_enabled: bool = False,
    s_smooth_radius: int = 15,
    s_denoise_threshold: float = 0.05,
    s_denoise_strength: float = 0.8,
    s_use_soft_threshold: bool = True,
    v_enabled: bool = False,
    v_smooth_radius: int = 15,
    v_denoise_threshold: float = 0.05,
    v_denoise_strength: float = 0.5,
    v_use_soft_threshold: bool = True,
    edge_protection: float = 50.0,
    low_sat_skip: float = 0.1,
    *,
    background_enabled: bool = True,
    foreground_mask: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    report: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return BGR uint8 result and 5x difference without resizing.

    H = a/b NL-Means; S = chroma edge-preserving smoothing;
    V = L NL-Means and conservative background unification.
    Without V, L is fixed in float Lab (output gamut/8-bit rounding still apply).
    Zero overall strength returns bit-exact original pixels.
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) < 1:
        raise ValueError('输入必须是非空 H×W×3 的 uint8 BGR 图片')
    args = locals()
    st = Settings(**{name: args[name] for name in Settings.__dataclass_fields__}).validated()
    emit = report or (lambda message: warnings.warn(message, RuntimeWarning, stacklevel=2))
    if valid_mask is not None and valid_mask.shape != image_bgr.shape[:2]:
        raise ValueError('有效像素蒙版尺寸不匹配')
    if st.overall_strength == 0 or not (st.h_denoise_strength * st.h_denoise_threshold or st.s_enabled and st.s_denoise_strength * st.s_denoise_threshold or st.v_enabled and st.v_denoise_strength * st.v_denoise_threshold):
        return image_bgr.copy(), np.zeros_like(image_bgr)

    lab = cv2.cvtColor(image_bgr.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    original_lab = lab.copy()
    template = min(11, 3 + 2 * int(st.edge_protection / 50))
    if st.h_denoise_strength > 0 and st.h_denoise_threshold > 0:
        ab8 = np.clip(np.rint(lab[..., 1:] + 128), 0, 255).astype(np.uint8)
        den = np.stack([_nlm(ab8[..., i], st.h_denoise_threshold, st.h_smooth_radius, template) for i in range(2)], axis=-1)
        delta = den.astype(np.float32) - ab8.astype(np.float32)
        lab[..., 1:] += st.h_denoise_strength * delta * _gate(delta, st.h_denoise_threshold * 2, st.h_use_soft_threshold)

    if st.s_enabled and st.s_denoise_strength > 0 and st.s_denoise_threshold > 0:
        bgr = np.clip(np.rint(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR) * 255), 0, 255).astype(np.uint8)
        smooth = cv2.edgePreservingFilter(bgr, flags=cv2.RECURS_FILTER, sigma_s=float(st.s_smooth_radius * 3), sigma_r=st.s_denoise_threshold * 2)
        smooth_lab = cv2.cvtColor(smooth.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
        delta = smooth_lab[..., 1:] - lab[..., 1:]
        lab[..., 1:] += st.s_denoise_strength * delta * _gate(delta, st.s_denoise_threshold * 100, st.s_use_soft_threshold)

    if st.low_sat_skip > 0:
        hsv = cv2.cvtColor(image_bgr.astype(np.float32) / 255, cv2.COLOR_BGR2HSV)
        weight = np.clip((hsv[..., 1] - st.low_sat_skip / 2) / (st.low_sat_skip / 2), 0, 1)
        weight *= np.clip((hsv[..., 2] - 0.05) / 0.10, 0, 1)
        lab[..., 1:] = original_lab[..., 1:] + weight[..., None] * (lab[..., 1:] - original_lab[..., 1:])

    if st.v_enabled and st.v_denoise_strength > 0 and st.v_denoise_threshold > 0:
        l8 = np.clip(np.rint(lab[..., 0] * 2.55), 0, 255).astype(np.uint8)
        den = _nlm(l8, st.v_denoise_threshold * 255, st.v_smooth_radius, template)
        delta = (den.astype(np.float32) - l8.astype(np.float32)) / 2.55
        lab[..., 0] += st.v_denoise_strength * delta * _gate(delta, st.v_denoise_threshold * 200, st.v_use_soft_threshold)
        if st.background_enabled:
            if foreground_mask is None:
                from .segment import SegmentationUnavailable
                try:
                    foreground_mask = get_foreground_mask(image_bgr)
                except SegmentationUnavailable as exc:
                    emit(f'背景纯色化已跳过：{exc}；H/S/V 降噪仍然生效。')
            if foreground_mask is not None:
                if foreground_mask.shape != image_bgr.shape[:2] or not np.isfinite(foreground_mask).all():
                    raise ValueError('前景蒙版尺寸或数值无效')
                fg = np.clip(foreground_mask.astype(np.float32) / 255, 0, 1)
                bg = fg < 0.05
                if valid_mask is not None:
                    bg &= valid_mask.astype(bool)
                if _flat_background(original_lab, bg):
                    mean = np.mean(original_lab[bg], axis=0)
                    weight = cv2.GaussianBlur(1-fg, (0, 0), sigmaX=max(0.5, st.v_smooth_radius / 6)) * (1-fg)
                    if valid_mask is not None:
                        weight *= valid_mask.astype(np.float32)
                    delta = mean - lab
                    gate = _gate(delta, st.v_denoise_threshold * 200, st.v_use_soft_threshold)
                    lab += st.v_denoise_strength * weight[..., None] * gate * delta
                    emit('已统一近似纯色背景；前景边缘采用柔和过渡。')
                else:
                    emit('背景含渐变、纹理或有效区域不足，已保留背景结构。')

    candidate = np.clip(np.rint(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR) * 255), 0, 255).astype(np.uint8)
    unchanged = np.all(lab == original_lab, axis=2)
    candidate[unchanged] = image_bgr[unchanged]
    result = cv2.addWeighted(image_bgr, 1-st.overall_strength, candidate, st.overall_strength, 0)
    if valid_mask is not None:
        result[~valid_mask.astype(bool)] = image_bgr[~valid_mask.astype(bool)]
    diff = np.minimum(cv2.absdiff(image_bgr, result).astype(np.uint16) * 5, 255).astype(np.uint8)
    return result, diff
