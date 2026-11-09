# DOISH Image Lab

围绕 Hive 实测的本机图像处理工具：生成多种候选，记录原图与处理图的 AI 检测分数，再结合画质门槛筛选结果。原有 DOISH 色彩去抖作为一个独立功能标签和可选前处理保留。

**当前没有经真实 Hive 测试验证的通用降分方案。** 新模块提供重采样、多尺度亮度细节调整、可复现颗粒和组合处理；其有效性必须由检测结果验证，不能从滤波强度、像素变化或软件测试通过推导。未测或没有改善时，界面会明确显示。

## Image Lab 主流程

1. 上传 1–4 张图，选择处理方案，生成原文件、统一 PNG 对照与候选。
2. 下载测试包，在 Hive 网站分别检测后导入填写好的 CSV；也可使用官方 V2 项目密钥批量检测。
3. 按同批次分数下降、PSNR 和 SSIM 门槛筛选，导出入选图片与 HTML 报告；没有满足条件的结果则不推荐。
4. 在新图片上复验，避免把单张图片上的变化当成普遍效果。

实验记录保存参数、种子、SHA-256、检测来源 / 时间 / 模型版本，支持重启恢复。网站与 API 分数不混排。**[完整使用说明与限制](HIVE_LAB.md)**。

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

以下是保留在第二个标签中的 DOISH 色彩去抖功能：

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

- [Image Lab：实验、Hive 对接、分数导入与复验](HIVE_LAB.md)
- [功能与参数说明](狗梅小工具_v9.4/使用说明.md)
- [安装、模型、打包与排错](狗梅小工具_v9.4/如何启动.md)
- [审计问题与修复状态](ISSUE_v9.4_audit.md)

## 开发与测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试覆盖原有色彩流程，以及候选复现、画质指标、原文件字节保留、分数导入原子性、文件哈希校验、失败不误判为零、同批次比较、API 响应解析和 ZIP 导出。API 测试使用模拟响应，不代表线上检测效果。GitHub Actions 在 Windows / macOS / Linux、Python 3.10 / 3.12 上运行测试，并构建 Windows 目录版产物。

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
├── hive_lab/                 # 候选、实验、官方 API、结果筛选与界面
├── web/                      # 对比、缩放、平移、HSB 预览
├── models/                   # 首次下载创建，Git 忽略
└── runtime/                  # 用户预设及临时结果，Git 忽略
```

## License

项目代码的许可证仍为 **TBD**，本次修复不擅自更改授权方式。第三方库和模型按其各自许可证使用；anime-seg 官方模型页面标注 Apache-2.0。仓库不重新分发模型。
