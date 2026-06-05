#!/bin/bash
# 狗梅小工具 — 一键启动脚本
# 运行前请确保已安装依赖：pip install -r requirements.txt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🐶 启动狗梅小工具..."
echo "📌 访问地址：http://localhost:7860"
echo "   按 Ctrl+C 停止"
echo ""

python3 app.py
