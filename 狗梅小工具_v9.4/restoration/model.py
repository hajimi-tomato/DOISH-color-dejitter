"""Hash-verified DRUNet ONNX with bounded inference and optional DirectML."""
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import urllib.request

import numpy as np

from color_dejitter.segment import app_dir

MODEL_SHA256 = '39e42853791f9255551cb1dc534c8e778dcbd9316bd07fff8097ab38e0408d70'
MODEL_URL = 'https://github.com/hajimi-tomato/DOISH-color-dejitter/releases/download/models-drunet-v1/drunet_color.onnx'
SOURCE_SHA256 = '479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4'
_lock = threading.RLock()
_session = None
_key = None


def model_path():
    return Path(os.environ.get('DOISH_DRUNET_MODEL', app_dir()/'models/drunet_color.onnx')).resolve()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(report=print):
    with _lock:
        path = model_path()
        if path.is_file() and sha256(path) == MODEL_SHA256:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            request = urllib.request.Request(MODEL_URL, headers={'User-Agent':'DOISH-Image-Lab'})
            with urllib.request.urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(dir=path.parent, suffix='.part', delete=False) as stream:
                temporary = Path(stream.name)
                count = 0
                while chunk := response.read(1024*1024):
                    stream.write(chunk)
                    count += len(chunk)
                    if count > 160*1024*1024:
                        raise ValueError('下载大小异常')
                    report(f'DRUNet 模型下载：{count/1024**2:.0f} / 125 MiB')
            if sha256(temporary) != MODEL_SHA256:
                raise ValueError('DRUNet 模型 SHA-256 校验失败')
            temporary.replace(path)
            return path
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)


def get_predictor(device='auto', report=None):
    global _session, _key
    if device not in ('auto', 'cpu'):
        raise ValueError('设备必须是 auto 或 cpu')
    path = model_path()
    if not path.is_file():
        raise RuntimeError('DRUNet 模型未安装，请先点击「下载 / 校验降噪模型」')
    import onnxruntime as ort
    stamp = (str(path), path.stat().st_size, path.stat().st_mtime_ns, device)
    with _lock:
        if stamp != _key:
            if sha256(path) != MODEL_SHA256:
                raise RuntimeError('DRUNet 权重校验失败，请重新下载模型')
            options = ort.SessionOptions()
            options.intra_op_num_threads = min(4, os.cpu_count() or 1)
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = ['CPUExecutionProvider']
            if device == 'auto' and 'DmlExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'DmlExecutionProvider')
            try:
                session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            except Exception:
                if len(providers) == 1:
                    raise
                if report:
                    report('DirectML 初始化失败，改用 CPU')
                session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
            _session, _key = session, stamp
        session = _session
    if report:
        report('推理设备：'+session.get_providers()[0])
    def predict(value):
        with _lock:
            try:
                output = session.run(['output'], {'input':np.ascontiguousarray(value, dtype=np.float32)})[0]
            except Exception as exc:
                raise RuntimeError('DRUNet 推理失败；显存不足时请将分块调为 128 或选择 CPU') from exc
        if output.shape != (value.shape[0], 3, *value.shape[2:]) or not np.isfinite(output).all():
            raise RuntimeError('降噪模型输出异常，已停止导出')
        return output
    return predict
