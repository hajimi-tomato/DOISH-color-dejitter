"""Validated, atomic local preset storage; built-ins cannot be overwritten."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import threading

from .core import Settings
from .segment import app_dir

DEFAULT = Settings()
BUILTINS = {
    '默认': DEFAULT,
    '大狗叫': replace(DEFAULT, h_denoise_threshold=12, s_enabled=True, s_denoise_threshold=0.1, v_enabled=True, v_denoise_strength=0.5),
    '轻度修正': replace(DEFAULT, overall_strength=0.6, h_denoise_threshold=6, h_denoise_strength=0.5),
    '强力去噪': replace(DEFAULT, overall_strength=0.9, h_denoise_threshold=18, v_enabled=True, v_denoise_threshold=0.025, v_denoise_strength=0.3, background_enabled=False),
    '全通道修正': replace(DEFAULT, s_enabled=True, v_enabled=True),
    '耄耋哈': replace(DEFAULT, overall_strength=1, h_denoise_threshold=24, h_denoise_strength=1, s_enabled=True, s_denoise_threshold=0.2, s_denoise_strength=1, v_enabled=True, v_denoise_strength=1, low_sat_skip=0.05),
}
_lock = threading.RLock()


def data_dir():
    return Path(os.environ.get('DOISH_DATA_DIR', app_dir() / 'runtime')).expanduser().resolve()


class PresetStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else data_dir() / 'presets.json'

    def custom(self):
        with _lock:
            if not self.path.exists():
                return {}
            try:
                if self.path.stat().st_size > 1024**2:
                    raise ValueError('文件过大')
                document = json.loads(self.path.read_text(encoding='utf-8'))
                if document['version'] != 1 or not isinstance(document['presets'], dict):
                    raise ValueError('不支持的格式')
                result = {}
                for name, values in document['presets'].items():
                    self._name(name)
                    result[name] = Settings(**values).validated()
                return result
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'预设文件损坏或格式不兼容，原文件已保留：{self.path}') from exc

    def all(self):
        return {**BUILTINS, **self.custom()}

    @staticmethod
    def _name(name):
        if not isinstance(name, str) or not name.strip() or len(name) > 48 or any(ord(c) < 32 for c in name):
            raise ValueError('预设名称需为 1–48 个可见字符')
        if name in BUILTINS:
            raise ValueError('内置预设不能覆盖或删除，请使用新名称')
        return name.strip()

    def _write(self, values):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=self.path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({'version': 1, 'presets': {k: asdict(v) for k, v in values.items()}}, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)

    def save(self, name, settings):
        name = self._name(name)
        # Check the trimmed name as well (e.g. " 默认 ").
        self._name(name)
        with _lock:
            values = self.custom()
            values[name] = settings.validated()
            self._write(values)
        return name

    def delete(self, name):
        name = self._name(name)
        with _lock:
            values = self.custom()
            if name not in values:
                raise ValueError('预设不存在')
            del values[name]
            self._write(values)
