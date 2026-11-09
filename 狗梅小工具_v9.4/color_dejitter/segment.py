"""In-process ONNX segmentation with a reusable session and bounded mask cache."""
import argparse
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import sys
import threading

import cv2
import numpy as np


class SegmentationUnavailable(RuntimeError):
    pass


def app_dir():
    return Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent.parent


def model_path():
    return Path(os.environ.get('DOISH_MODEL_PATH', app_dir() / 'models' / 'isnetis.onnx')).expanduser().resolve()


_lock = threading.RLock()
_session = None
_session_key = None
_masks = OrderedDict()
_CACHE_BYTES = 32 * 1024 * 1024


def _get_session(path):
    global _session, _session_key
    if not path.is_file():
        raise SegmentationUnavailable('未找到分割模型，请在界面下载模型或运行 download_model.py')
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _lock:
        if key != _session_key:
            try:
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = min(4, os.cpu_count() or 1)
                session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
            except Exception as exc:
                raise SegmentationUnavailable(f'分割模型无法加载（{type(exc).__name__}），请检查 onnxruntime 或重新下载模型') from exc
            _session, _session_key = session, key
            _masks.clear()
        return _session


def get_mask(input_img, onnx_session, s=1024):
    """RGB uint8 -> original-size uint8 mask; aspect ratio and padding preserved."""
    h0, w0 = input_img.shape[:2]
    if min(h0, w0) < 1:
        raise ValueError('图片不能为空')
    scale = s / max(h0, w0)
    h, w = max(1, round(h0 * scale)), max(1, round(w0 * scale))
    ph, pw = s - h, s - w
    canvas = np.zeros((s, s, 3), np.float32)
    resized = cv2.resize(input_img.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas[ph//2:ph//2+h, pw//2:pw//2+w] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])
    output = onnx_session.run(None, {onnx_session.get_inputs()[0].name: tensor})[0]
    pred = np.squeeze(output)
    if pred.shape != (s, s) or not np.isfinite(pred).all():
        raise ValueError('分割模型返回了无效的蒙版')
    pred = cv2.resize(pred[ph//2:ph//2+h, pw//2:pw//2+w], (w0, h0))
    return np.rint(np.clip(pred, 0, 1) * 255).astype(np.uint8)


def foreground_mask(image_bgr):
    with _lock:
        session = _get_session(model_path())
        key = (image_bgr.shape, hashlib.sha256(image_bgr.tobytes()).digest())
        if key in _masks:
            _masks.move_to_end(key)
            return _masks[key].copy()
        try:
            mask = get_mask(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), session)
        except Exception as exc:
            raise SegmentationUnavailable(f'分割推理失败（{type(exc).__name__}）') from exc
        if mask.nbytes <= _CACHE_BYTES:
            _masks[key] = mask
            while len(_masks) > 4 or sum(m.nbytes for m in _masks.values()) > _CACHE_BYTES:
                _masks.popitem(last=False)
        return mask.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_image')
    parser.add_argument('output_mask')
    args = parser.parse_args()
    data = np.fromfile(args.input_image, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        parser.error('无法读取输入图片')
    mask = foreground_mask(image)
    ok, png = cv2.imencode('.png', mask)
    if not ok:
        raise RuntimeError('蒙版编码失败')
    png.tofile(args.output_mask)


if __name__ == '__main__':
    main()
