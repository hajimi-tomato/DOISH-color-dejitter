[README.md](https://github.com/user-attachments/files/28632497/README.md)
# 🐶 狗梅小工具 (Goumei Tool) v9.4

> AI 绘画色彩去抖动工具 — 通过 NL-Means + AI 前景分割消除 AI 生图中的色彩噪声，让画面更干净、色彩更统一。

## 这是什么？

AI 生成的图像在同一色彩区域内常常存在色相级别的随机抖动——肉眼看可能不明显，但放大观察、提高饱和度、做后期调色或打印时就会暴露。狗梅小工具通过智能算法消除这些色彩噪声。

## 核心功能

- **NL-Means 色度降噪** — 在 Lab 色空间只对色度通道（a/b）做非局部均值降噪，亮度通道（L）不动，只消除色彩噪声不损失亮度细节
- **保边滤波色度平滑** — 平坦区域强力平滑、边缘处保持锐利
- **AI 前景/背景分割** — 使用 `anime-segmentation`（ISNet）模型分割前景背景；纯色背景统一为均值色，复杂背景只做降噪

## 快速开始

```bash
# 1. 安装依赖（推荐 Python 3.9~3.12）
pip install -r requirements.txt

# 2. 启动
python app.py
# 或
python launcher.py   # 自动开浏览器
```

启动后访问 **http://localhost:7860**。

详细启动指南见 [`如何启动.md`](./如何启动.md)，功能说明见 [`使用说明.md`](./使用说明.md)。

## 文件结构

```
狗梅小工具_v9.4/
├── app.py                 # 主程序（Gradio Web UI）
├── launcher.py            # 启动器（自动打开浏览器）
├── 启动.sh                # macOS/Linux 一键启动
├── build_windows.bat      # Windows 打包脚本
├── requirements.txt       # Python 依赖
├── color_dejitter/        # 核心算法包
│   ├── core.py            # 色彩去抖动核心算法 v9.4
│   └── segment.py         # anime-segmentation 分割脚本
├── models/
│   └── isnetis.onnx       # AI 分割模型（167MB，需单独下载）※
├── 使用说明.md            # 功能和参数说明
└── 如何启动.md            # 安装与启动指南
```

> ※ 模型文件较大，未纳入仓库。请从 release 或 [anime-segmentation 项目](https://github.com/SkyTNT/anime-segmentation) 下载并放入 `models/` 目录。

## 已知问题

参见 [`ISSUE_v9.4_audit.md`](./ISSUE_v9.4_audit.md) — 项目首轮代码审计清单（含 V 通道死参数、跨平台路径硬编码等 9 类问题）。

## License

TBD
