"""
Color De-Jitter Core Algorithm
色彩去抖动核心算法 v9.4

v9.4 anime-seg 分割 + NL-Means + edgePreservingFilter：
- 使用 anime-segmentation 模型（ONNX）做前景/背景分割
- 背景区域：色度统一替换为区域均值（真正的纯色）
- 前景区域：NL-Means 色度降噪 + edgePreservingFilter 保边平滑
- 分割通过 subprocess 调用系统 Python 3.9 + onnxruntime（因 Python 3.14 不支持 onnxruntime）
"""

import cv2
import numpy as np
import subprocess
import tempfile
import os
from typing import Tuple


# ─────────────────────────────────────────────────────────────
# anime-seg 分割
# ─────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SEGMENT_SCRIPT = os.path.join(_SCRIPT_DIR, "segment.py")
_SYSTEM_PYTHON = "/usr/bin/python3"


def get_foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    调用 anime-segmentation 获取前景 mask。
    返回 0~255 的灰度图，255=前景（人物），0=背景。
    如果分割失败，返回全 255（不分割，全部当前景处理）。
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f_in:
            input_path = f_in.name
            cv2.imwrite(input_path, image_bgr)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f_out:
            output_path = f_out.name

        result = subprocess.run(
            [_SYSTEM_PYTHON, _SEGMENT_SCRIPT, input_path, output_path],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0 and os.path.exists(output_path):
            mask = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # 清理
                os.unlink(input_path)
                os.unlink(output_path)
                return mask

        # 失败时清理
        for p in [input_path, output_path]:
            if os.path.exists(p):
                os.unlink(p)
    except Exception:
        pass

    # 分割失败，返回全前景
    return np.full(image_bgr.shape[:2], 255, dtype=np.uint8)


def color_dejitter(image_bgr: np.ndarray,
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
                   ) -> Tuple[np.ndarray, np.ndarray]:

    # ── 参数映射 ──────────────────────────────────────────────────

    chroma_h = max(1.0, float(h_denoise_threshold))
    luma_h = float(h_denoise_strength) / 100.0 * 3.0

    search_size = int(h_smooth_radius) * 2 + 1
    search_size = max(11, min(search_size, 61))
    if search_size % 2 == 0:
        search_size += 1

    template_size = max(3, min(int(edge_protection / 200.0 * 10) + 3, 11))
    if template_size % 2 == 0:
        template_size += 1

    # ── 第零步：anime-seg 前景/背景分割 ──────────────────────────

    fg_mask = get_foreground_mask(image_bgr)
    fg_binary = (fg_mask > 128).astype(np.uint8)
    bg_binary = 1 - fg_binary
    has_bg = np.sum(bg_binary) > 100

    # ── 第一步：NL-Means 色度降噪（全图）──────────────────────────

    result_bgr = cv2.fastNlMeansDenoisingColored(
        image_bgr, None,
        luma_h, chroma_h,
        template_size, search_size
    )

    # ── 第二步：edgePreservingFilter 色度平坦化（可选，前景区域）──

    if s_enabled:
        sigma_s = float(s_smooth_radius) * 3.0
        sigma_s = max(10.0, min(sigma_s, 200.0))
        sigma_r = float(s_denoise_threshold) * 2.0
        sigma_r = max(0.05, min(sigma_r, 0.8))

        lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        L_ch = lab[:, :, 0]

        smoothed_bgr = cv2.edgePreservingFilter(
            result_bgr,
            flags=cv2.RECURS_FILTER,
            sigma_s=sigma_s,
            sigma_r=sigma_r
        )

        smoothed_lab = cv2.cvtColor(smoothed_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        lab_out = np.stack([L_ch, smoothed_lab[:, :, 1], smoothed_lab[:, :, 2]], axis=-1)
        lab_out = np.clip(np.round(lab_out), 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)

    # ── 第三步：背景纯色化（仅当背景是近似纯色时）─────────────────

    bg_is_flat = False
    if has_bg:
        # 转 Lab
        lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

        # 背景区域的 L/a/b 均值和标准差
        bg_mask_bool = bg_binary.astype(bool)
        bg_mean_L = np.mean(lab[:, :, 0][bg_mask_bool])
        bg_mean_a = np.mean(lab[:, :, 1][bg_mask_bool])
        bg_mean_b = np.mean(lab[:, :, 2][bg_mask_bool])

        bg_std_L = np.std(lab[:, :, 0][bg_mask_bool])
        bg_std_a = np.std(lab[:, :, 1][bg_mask_bool])
        bg_std_b = np.std(lab[:, :, 2][bg_mask_bool])

        # 判断背景是否接近纯色：三个通道的标准差都很小
        # 纯色背景的标准差通常 < 10，复杂背景（渐变、纹理）> 15
        bg_is_flat = (bg_std_L < 12.0) and (bg_std_a < 8.0) and (bg_std_b < 8.0)

        if bg_is_flat:
            # 纯色背景：L/a/b 全部替换为均值
            lab[:, :, 0] = np.where(bg_mask_bool, bg_mean_L, lab[:, :, 0])
            lab[:, :, 1] = np.where(bg_mask_bool, bg_mean_a, lab[:, :, 1])
            lab[:, :, 2] = np.where(bg_mask_bool, bg_mean_b, lab[:, :, 2])

            # 边缘过渡
            fg_soft = cv2.GaussianBlur(fg_mask.astype(np.float32), (0, 0), sigmaX=2.0) / 255.0
            fg_soft_3ch = np.stack([fg_soft] * 3, axis=-1)

            lab_out = np.clip(np.round(lab), 0, 255).astype(np.uint8)
            bg_unified_bgr = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)

            result_bgr = (
                result_bgr.astype(np.float32) * fg_soft_3ch +
                bg_unified_bgr.astype(np.float32) * (1.0 - fg_soft_3ch)
            )
            result_bgr = np.clip(np.round(result_bgr), 0, 255).astype(np.uint8)
        # else: 复杂背景，只做降噪（已在第一二步完成），不做纯色化

    # ── 低饱和度区域保护（只作用于前景）──────────────────────────

    if low_sat_skip > 0:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        s_norm = hsv[:, :, 1].astype(np.float32) / 255.0
        v_norm_hsv = hsv[:, :, 2].astype(np.float32) / 255.0

        blend_weight = np.clip((s_norm - low_sat_skip * 0.5) / (low_sat_skip * 0.5 + 1e-6), 0, 1)
        blend_weight = np.where(v_norm_hsv < 0.15, 0.0, blend_weight)

        # 纯色背景区域不做低饱和度保护（已经纯色化了）
        if has_bg and bg_is_flat:
            blend_weight = np.where(bg_binary.astype(bool), 1.0, blend_weight)

        blend_weight_3ch = np.stack([blend_weight] * 3, axis=-1).astype(np.float32)

        result_bgr = (
            result_bgr.astype(np.float32) * blend_weight_3ch +
            image_bgr.astype(np.float32) * (1.0 - blend_weight_3ch)
        )
        result_bgr = np.clip(np.round(result_bgr), 0, 255).astype(np.uint8)

    # ── 整体强度混合（只作用于前景）──────────────────────────────

    if overall_strength < 1.0:
        blended = cv2.addWeighted(
            image_bgr, 1.0 - overall_strength,
            result_bgr, overall_strength,
            0
        )
        # 背景区域保持纯色，不参与混合
        if has_bg:
            fg_soft2 = cv2.GaussianBlur(fg_mask.astype(np.float32), (0, 0), sigmaX=2.0) / 255.0
            fg_soft2_3ch = np.stack([fg_soft2] * 3, axis=-1)
            result_bgr = (
                blended.astype(np.float32) * fg_soft2_3ch +
                result_bgr.astype(np.float32) * (1.0 - fg_soft2_3ch)
            )
            result_bgr = np.clip(np.round(result_bgr), 0, 255).astype(np.uint8)
        else:
            result_bgr = blended

    # ── 差异图 ───────────────────────────────────────────────────

    diff = cv2.absdiff(image_bgr, result_bgr)
    diff_amplified = np.clip(diff.astype(np.float32) * 5, 0, 255).astype(np.uint8)

    return result_bgr, diff_amplified
