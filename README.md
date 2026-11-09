# 🐶 狗梅小工具 · DOISH color de-jitter

本机运行的 AI 插画色彩清理工具。提供色度 NL-Means、保边色度平滑、可选亮度降噪和 AI 前景分割，配套 Gradio 界面、参数预设、前后对比、HSB 观察工具与 PNG 导出。

本次修复保留 `狗梅小工具_v9.4/` 目录，补齐原文档描述的应用，并修复失效参数、Windows 分割路径和背景处理问题。

## 快速开始

推荐 **64-bit Python 3.10–3.12**。从仓库根目录执行：

```bash
python -m venv .venv
# Windows：.venv\Scripts\activate
# macOS / Linux：source .venv/bin/activate
python -m pip install -r requirements.txt
python "狗梅小工具_v9.4/launcher.py"
```

启动后浏览器访问 <http://127.0.0.1:7860>。Windows 也可双击 `狗梅小工具_v9.4/启动.bat`；macOS/Linux 可运行 `bash 狗梅小工具_v9.4/启动.sh`。这两个脚本首次启动会在工具目录创建独立 `.venv` 并安装依赖。

**普通色度降噪不需要模型，也不需要独立显卡。** 需要背景纯色化时，在「高级设置与模型」点击下载，或执行：

```bash
python "狗梅小工具_v9.4/download_model.py"
```

分割模型来自 [SkyTNT/anime-segmentation](https://github.com/SkyTNT/anime-segmentation) 的 [官方 Hugging Face 权重](https://huggingface.co/skytnt/anime-seg)。下载约 176 MB，程序检查 SHA-256；模型不提交到 Git。

## 功能与边界

- **H：色度降噪。** 对 Lab 的 a/b 通道分别做 NL-Means，保留工作空间中的 L。
- **S：保边平滑。** 可选地进一步平滑色度；半径、阈值、强度及软/硬阈值均生效。
- **V：亮度与背景。** 显式启用后进行亮度降噪；还可单独关闭「同时统一近似纯色背景」。默认关闭 V。
- **整体强度。** 对前景、背景统一混合；0 严格返回载入后的原始像素。
- **背景保护。** 使用原图的全局统计及空间分区中位数，优先保留渐变、阴影、纹理；不确定时跳过纯色化。它是启发式检测，不保证所有图片都正确。
- **本机界面。** 文件上传、剪贴板粘贴、六个内置预设、自定义预设保存/删除、可缩放平移的分割对比、5 倍差异图与下载。
- **HSB 观察工具。** 同时改变左右预览，帮助观察杂色；完全不参与结果导出。
- **图像信息。** 保留 alpha 和 PNG 文本（如 AI 参数），应用 EXIF 方向。带 ICC 的输入转换为 sRGB 并为输出附上对应配置。
- **高分辨率。** 导出保留原尺寸；当前限制 3200 万像素、100 MiB。浏览器预览最长边 1600 px。处理链为 8-bit，不面向 HDR / 16-bit 专业修图。

降噪会改变像素，可能损失真实的细微色彩变化；“Lab 中保持 L”也不等于 RGB 输出逐像素亮度完全一致，色域裁切和 8-bit 舍入仍有误差。建议先用「轻度修正」并检查原图。

## 文档

- [功能与参数说明](狗梅小工具_v9.4/使用说明.md)
- [安装、模型、打包与排错](狗梅小工具_v9.4/如何启动.md)
- [审计问题与修复状态](ISSUE_v9.4_audit.md)

## 开发与测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试覆盖参数有效性、颜色/亮度行为、强度零值、渐变、分块边界、透明通道、PNG 元数据、预设与分割缓存。GitHub Actions 在 Windows / macOS / Linux、Python 3.10 / 3.12 上运行测试，并构建 Windows 目录版产物。

```bash
python "狗梅小工具_v9.4/build_windows.py"  # 在 Windows 上运行
```

分发整个 `狗梅小工具_v9.4/dist/DOISH/` 目录；不能只拷贝一个 exe。默认不捆绑模型。

## 项目结构

```text
狗梅小工具_v9.4/
├── app.py / launcher.py       # Gradio 应用与自动开浏览器入口
├── download_model.py          # 下载与校验原始模型
├── build_windows.py / .bat    # PyInstaller 目录版打包
├── 启动.bat / 启动.sh          # 独立环境启动脚本
├── color_dejitter/
│   ├── core.py                # H/S/V 算法与参数校验
│   ├── segment.py             # ONNX 会话及蒙版缓存
│   ├── images.py              # 图片、透明通道与色彩管理
│   └── presets.py             # 内置及持久化自定义预设
├── web/                      # 对比、缩放、平移、HSB 预览
├── models/                   # 首次下载创建，Git 忽略
└── runtime/                  # 用户预设及临时结果，Git 忽略
```

## License

项目代码的许可证仍为 **TBD**，本次修复不擅自更改授权方式。第三方库和模型按其各自许可证使用；anime-seg 官方模型页面标注 Apache-2.0。仓库不重新分发模型。
