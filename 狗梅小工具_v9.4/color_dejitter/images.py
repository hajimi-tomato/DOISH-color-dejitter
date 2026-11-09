"""Image I/O with bounded large PNG metadata, alpha and explicit color management."""
from dataclasses import dataclass, field
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, PngImagePlugin

MAX_PIXELS = 32_000_000
MAX_FILE_BYTES = 100 * 1024**2
# Some AI PNGs carry multi-megabyte compressed prompts/workflows.
PngImagePlugin.MAX_TEXT_CHUNK = 16 * 1024**2
PngImagePlugin.MAX_TEXT_MEMORY = 64 * 1024**2
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))


@dataclass
class SourceImage:
    bgr: np.ndarray
    alpha: np.ndarray | None = None
    text: dict = field(default_factory=dict)
    icc: bytes | None = None


def load_image(path):
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('图片文件超过 100 MiB，请先压缩或裁切')
    with Image.open(path) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError('图片超过 3200 万像素，请先裁切；程序不会静默缩小导出图片')
        if getattr(image, 'is_animated', False):
            raise ValueError('暂不处理动画，请先导出单帧图片')
        if image.mode in ('I', 'I;16', 'I;16B', 'I;16L', 'F'):
            raise ValueError('当前处理链为 8-bit，请先将高位深图片转换为 8-bit sRGB')
        image.load()
        texts = {k: v for k, v in getattr(image, 'text', {}).items() if isinstance(v, str)}
        icc = image.info.get('icc_profile')
        image = ImageOps.exif_transpose(image)
        rgba = image.convert('RGBA')
        alpha = np.array(rgba.getchannel('A')) if rgba.getchannel('A').getextrema()[0] < 255 else None
        rgb = image.convert('RGB')
        if icc:
            try:
                profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                color_input = image if image.mode in ('RGB', 'CMYK', 'L', 'LAB') else rgb
                rgb = ImageCms.profileToProfile(color_input, profile, SRGB, outputMode='RGB')
            except Exception as exc:
                raise ValueError('图片的 ICC 配置无法转换为 sRGB，请先在图像软件中转换色彩空间') from exc
        return SourceImage(cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR), alpha, texts, SRGB.tobytes() if icc else None)


def as_pil(bgr, alpha=None):
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if alpha is not None:
        image.putalpha(Image.fromarray(alpha))
    return image


def save_png(path, bgr, source=None):
    image = as_pil(bgr, source.alpha if source else None)
    info = PngImagePlugin.PngInfo()
    for name, value in (source.text if source else {}).items():
        info.add_itxt(name, value, zip=True)
    image.save(path, format='PNG', pnginfo=info, compress_level=4,
               **({'icc_profile': source.icc} if source and source.icc else {}))


def preview_data(bgr, alpha=None, max_edge=1600):
    """Only the browser preview is scaled. Export always uses original pixels."""
    import base64
    image = as_pil(bgr, alpha)
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format='PNG', compress_level=1)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
