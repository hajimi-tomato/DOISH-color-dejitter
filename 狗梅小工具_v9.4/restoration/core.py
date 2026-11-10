"""Full-resolution restoration with learned denoising, L0 and protection masks."""
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from color_dejitter.core import Settings as ColorSettings, color_dejitter
from .keepcolor_numpy import denoise, l0_smooth, tiled
from .model import get_predictor


@dataclass(frozen=True)
class Settings:
    mode: str = 'pd2'
    sigma: float = 12
    refine: float = 3
    amount: float = 0.9
    texture: float = 0.15
    lam: float = 0.0003
    edges: float = 0.65
    color: float = 0.8
    doish: bool = False
    device: str = 'auto'
    tile: int = 256

    def validated(self):
        for name, low, high in [('sigma',0,50),('refine',0,10),('amount',0,1),('texture',0,1),
                                ('lam',0.00001,0.01),('edges',0,1),('color',0,1)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{name} 参数越界')
        if self.mode not in ('standard','pd2') or self.device not in ('auto','cpu'):
            raise ValueError('未知降噪模式 / 设备')
        if self.tile not in (128,256,512) or not isinstance(self.doish,bool):
            raise ValueError('分块大小 / 色彩开关无效')
        return self


PRESETS = {
    'grain': ('生成图 · 细颗粒', Settings(mode='standard',sigma=8,refine=0,texture=0,amount=0.85,edges=0.6)),
    'texture': ('生成图 · 相关纹理', Settings()),
    'flat': ('插画 · 平涂去纹理', Settings(sigma=16,texture=0.5,amount=1,edges=0.8)),
    'hive': ('Hive 实测候选 · 高频残差清理', Settings(sigma=18,refine=4,amount=1,texture=0.15,edges=0.25,color=0.75)),
}


def edge_mask(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0,0), 0.9)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)/8
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)/8
    edges = np.clip(np.sqrt(gx*gx+gy*gy)/0.12,0,1)
    return cv2.GaussianBlur(cv2.dilate(edges, np.ones((3,3),np.uint8)),(0,0),0.8)


def fill_transparent(rgb, alpha):
    if alpha is None or np.all(alpha>0):
        return rgb
    valid = alpha>0
    if not valid.any():
        return rgb
    _, labels = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    filled = rgb.copy()
    filled[~valid] = rgb[valid][labels[~valid]-1]
    return filled


def restore(source, settings=None, protect=None, report=None, predictor=None):
    settings = (settings or Settings()).validated()
    h, w = source.bgr.shape[:2]
    if h*w > 16_777_216:
        raise ValueError('深度清理限制 1677 万像素，请先裁切；不会静默缩小导出')
    if protect is not None:
        protect = np.asarray(protect, dtype=np.float32)
        if protect.shape != (h,w) or not np.isfinite(protect).all() or protect.min()<0 or protect.max()>1:
            raise ValueError('保护蒙版需与原图同尺寸，取值 0–1')
    if settings.amount == 0 or (protect is not None and np.all(protect == 1)) or (source.alpha is not None and not source.alpha.any()):
        return source.bgr.copy()
    if settings.sigma == 0 and settings.texture == 0 and not settings.doish:
        return source.bgr.copy()
    original = cv2.cvtColor(source.bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    image = fill_transparent(original, source.alpha)
    if settings.sigma > 0:
        predict = predictor or get_predictor(settings.device, report)
        tensor = image.transpose(2,0,1)[None]
        output = denoise(predict, tensor, settings.sigma, settings.mode, settings.refine, settings.tile, report)[0].transpose(1,2,0)
    else:
        output = image.copy()
    edges = edge_mask(image)
    if settings.texture > 0:
        if report:
            report('L0 色块纹理平滑')
        def smooth(part):
            return l0_smooth(part[0].transpose(1,2,0), settings.lam).transpose(2,0,1)[None]
        smoothed = tiled(smooth, output.transpose(2,0,1)[None], tile=512, overlap=64)[0].transpose(1,2,0)
        output += settings.texture*(1-edges[...,None])*(smoothed-output)
    if settings.doish:
        bgr = cv2.cvtColor(np.rint(np.clip(output,0,1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        bgr, _ = color_dejitter(bgr, **asdict(ColorSettings(background_enabled=False)))
        output = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    if settings.color > 0:
        # Restore only low-frequency chroma drift; fine chroma noise stays denoised.
        original_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        output_lab = cv2.cvtColor(np.ascontiguousarray(output), cv2.COLOR_RGB2LAB)
        drift = cv2.GaussianBlur(original_lab[...,1:]-output_lab[...,1:],(0,0),8)
        output_lab[...,1:] += settings.color*drift
        output = np.clip(cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB),0,1)
    weight = settings.amount*(1-settings.edges*edges)
    if protect is not None:
        weight *= 1-protect
    if source.alpha is not None:
        weight[source.alpha==0] = 0
    result = np.rint(np.clip(original+weight[...,None]*(output-original),0,1)*255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
