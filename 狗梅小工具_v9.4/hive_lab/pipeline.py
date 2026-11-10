"""Experimental image transforms, independent of the detector transport.

These are hypotheses to measure, not a reconstruction of Hive's private model.
All outputs retain input dimensions; random texture uses a recorded seed.
"""
from dataclasses import asdict, dataclass, replace
import math

import cv2
import numpy as np

from color_dejitter.core import Settings, color_dejitter


@dataclass(frozen=True)
class Recipe:
    label: str
    scale: float = 1.0
    fine: float = 0.0
    medium: float = 0.0
    grain: float = 0.0
    doish: bool = False
    deep: str = ''


RECIPES = {
    'deep_grain': Recipe('DRUNet · 细颗粒清理', deep='grain'),
    'deep_texture': Recipe('DRUNet-PD · 相关纹理清理', deep='texture'),
    'deep_flat': Recipe('DRUNet-PD + L0 · 平涂清理', deep='flat'),
    'deep_hive': Recipe('DRUNet-PD · Hive 实测候选', deep='hive'),
    'doish': Recipe('DOISH 色彩去抖', doish=True),
    'resample': Recipe('重采样单项', scale=0.85),
    'frequency': Recipe('多尺度细节单项', fine=0.4, medium=0.12),
    'texture': Recipe('亮度颗粒单项', grain=1.5),
    'balanced': Recipe('组合 · 温和', scale=0.9, fine=0.25, medium=0.06, grain=0.8),
    'strong': Recipe('组合 · 较强', scale=0.75, fine=0.5, medium=0.15, grain=1.5),
}


def process(source, recipe_id, seed=2026, add_doish=False):
    if recipe_id not in RECIPES:
        raise ValueError('未知处理方案')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError('随机种子必须是 0–4294967295 的整数')
    recipe = RECIPES[recipe_id]
    if recipe.deep:
        from restoration.core import PRESETS, restore
        return restore(source, replace(PRESETS[recipe.deep][1], doish=bool(add_doish)))
    original = source.bgr
    bgr = original.copy()
    if recipe.doish or add_doish:
        bgr, _ = color_dejitter(bgr, **asdict(Settings(background_enabled=False)),
                               valid_mask=source.alpha > 0 if source.alpha is not None else None)
    height, width = bgr.shape[:2]
    if recipe.scale < 1:
        small = cv2.resize(bgr, (max(1, round(width * recipe.scale)), max(1, round(height * recipe.scale))), interpolation=cv2.INTER_AREA)
        bgr = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    if recipe.fine or recipe.medium or recipe.grain:
        # Two Gaussian residual bands in Lab L. Chroma is left alone here.
        lab = cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        light = lab[..., 0]
        near = cv2.GaussianBlur(light, (0, 0), 0.65)
        far = cv2.GaussianBlur(light, (0, 0), 1.4)
        light = light - recipe.fine * (light - near) - recipe.medium * (near - far)
        if recipe.grain:
            noise = np.random.default_rng(seed).standard_normal(light.shape).astype(np.float32)
            # Reduce added texture near black/white, in units of 8-bit luminance.
            envelope = np.sqrt(np.clip(light / 100 * (1 - light / 100), 0, 0.25)) * 2
            light += noise * recipe.grain * (100 / 255) * envelope
        lab[..., 0] = np.clip(light, 0, 100)
        bgr = np.rint(np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0, 1) * 255).astype(np.uint8)
    if source.alpha is not None:
        bgr[source.alpha == 0] = original[source.alpha == 0]
    return bgr


def quality(original, output, alpha=None):
    """Full-resolution RGB PSNR/MAE; preview-scale luminance SSIM.

    Transparent images are compared composited over white, consistently.
    These metrics are fidelity proxies, not perceptual quality guarantees.
    """
    a, b = original.astype(np.float32), output.astype(np.float32)
    if alpha is not None:
        weight = alpha[..., None].astype(np.float32) / 255
        a, b = a * weight + 255 * (1 - weight), b * weight + 255 * (1 - weight)
    delta = a - b
    mse = float(np.mean(delta ** 2))
    psnr = None if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    h, w = a.shape[:2]
    if max(h, w) > 1024:
        size = (max(1, round(w * 1024 / max(h, w))), max(1, round(h * 1024 / max(h, w))))
        a, b = cv2.resize(a, size, interpolation=cv2.INTER_AREA), cv2.resize(b, size, interpolation=cv2.INTER_AREA)
    a, b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    ssim = ((2 * ma * mb + 6.5025) * (2 * cov + 58.5225)) / ((ma * ma + mb * mb + 6.5025) * (va + vb + 58.5225))
    return {'mae_rgb': round(float(np.mean(np.abs(delta))), 5),
            'psnr_db': None if psnr is None else round(psnr, 5),
            'ssim_1024': round(float(np.clip(np.mean(ssim), -1, 1)), 6)}
