"""Build the Windows folder distribution using the active virtual environment."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    if os.name != 'nt':
        raise SystemExit('此脚本用于 Windows；macOS/Linux 请从 Python 源码启动。')
    root = Path(__file__).resolve().parent
    args = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
            '--name', 'DOISH', '--console', '--add-data', f'{root / "web"};web',
            '--collect-all', 'gradio', '--collect-all', 'gradio_client',
            '--collect-all', 'onnxruntime', '--collect-all', 'groovy',
            '--collect-all', 'safehttpx', '--collect-all', 'hf_gradio',
            '--recursive-copy-metadata', 'gradio',
            '--hidden-import', 'hf_gradio', str(root / 'launcher.py')]
    subprocess.run(args, cwd=root, check=True)
    print('打包完成：dist/DOISH/DOISH.exe。分发时请保留整个 DOISH 文件夹。')
    print('模型未打包；可在界面下载，或放到 dist/DOISH/models/isnetis.onnx。')


if __name__ == '__main__':
    main()
