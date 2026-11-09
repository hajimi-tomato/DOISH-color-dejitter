"""Download the original anime-segmentation ONNX weights with SHA-256 verification."""
import hashlib
from pathlib import Path
import tempfile
import threading
import urllib.request

from color_dejitter.segment import model_path

URL = 'https://huggingface.co/skytnt/anime-seg/resolve/main/isnetis.onnx'
SHA256 = 'f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99'
_lock = threading.Lock()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(report=print):
    with _lock:
        target = model_path()
        if target.is_file() and sha256_file(target) == SHA256:
            report('分割模型已就绪，SHA-256 校验通过。')
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            request = urllib.request.Request(URL, headers={'User-Agent': 'DOISH-color-dejitter'})
            with urllib.request.urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(dir=target.parent, suffix='.part', delete=False) as dst:
                temporary = Path(dst.name)
                digest = hashlib.sha256()
                count, last = 0, -1
                while chunk := response.read(1024 * 1024):
                    dst.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                    if count // (16 * 1024 * 1024) != last:
                        last = count // (16 * 1024 * 1024)
                        report(f'正在下载分割模型：{count / 1024**2:.0f} MiB / 约 168 MiB')
            if digest.hexdigest() != SHA256:
                raise ValueError('模型 SHA-256 不匹配，已拒绝安装；请检查上游版本或下载完整性')
            temporary.replace(target)
            report('分割模型已下载并通过 SHA-256 校验。')
            return target
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    download()
