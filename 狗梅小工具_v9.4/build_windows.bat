@echo off
chcp 65001 >nul
echo ================================================
echo  狗梅小工具 - Windows 打包脚本
echo ================================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] 安装依赖...
pip install gradio>=6.0.0 numpy opencv-python Pillow pyinstaller --quiet
if errorlevel 1 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)

echo [2/3] 开始打包...
pyinstaller --noconfirm --clean ^
    --name "狗梅小工具" ^
    --add-data "app.py;." ^
    --add-data "color_dejitter;color_dejitter" ^
    --collect-data gradio ^
    --collect-data gradio_client ^
    --collect-submodules gradio ^
    --hidden-import cv2 ^
    --hidden-import PIL ^
    --hidden-import numpy ^
    --hidden-import uvicorn ^
    --hidden-import fastapi ^
    --hidden-import httpx ^
    --hidden-import color_dejitter ^
    --hidden-import color_dejitter.core ^
    --console ^
    launcher.py

if errorlevel 1 (
    echo [错误] 打包失败，请查看上方错误信息
    pause
    exit /b 1
)

echo.
echo [3/3] 打包完成！
echo.
echo exe 文件位置：dist\狗梅小工具\狗梅小工具.exe
echo.
echo 使用方法：
echo   双击 dist\狗梅小工具\狗梅小工具.exe 即可运行
echo   首次启动约需 10-20 秒，会自动打开浏览器
echo.
pause
