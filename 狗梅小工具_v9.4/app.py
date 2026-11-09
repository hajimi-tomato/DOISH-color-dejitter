"""Local Gradio application for DOISH color de-jitter."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

# Set before importing Gradio: no analytics or public sharing by default.
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import gradio as gr

from color_dejitter.core import Settings, color_dejitter
from color_dejitter.images import load_image, preview_data, save_png
from color_dejitter.presets import BUILTINS, PresetStore, data_dir
from color_dejitter.segment import model_path
from download_model import download

RESOURCES = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
FIELDS = list(Settings.__dataclass_fields__)
_results = None


def result_dir():
    global _results
    if _results is None:
        data_dir().mkdir(parents=True, exist_ok=True)
        _results = tempfile.TemporaryDirectory(prefix='results-', dir=data_dir())
    return Path(_results.name)


def settings_from(values):
    return Settings(**dict(zip(FIELDS, values))).validated()


def prepare_image(path):
    if not path:
        return None, '{}', None, None, '请上传或粘贴图片。'
    try:
        source = load_image(path)
        h, w = source.bgr.shape[:2]
        return str(path), json.dumps({'original': preview_data(source.bgr, source.alpha)}), None, None, f'已载入 {w} × {h} 图片。选择预设或调整参数后，点击「开始处理」。'
    except Exception as exc:
        # Clear stale images/results on invalid input, instead of processing the previous file.
        return None, '{}', None, None, f'图片无法载入：{exc}'


def process_image(path, *values):
    if not path:
        raise gr.Error('请先上传或粘贴图片')
    try:
        st = settings_from(values)
        started = time.perf_counter()
        source = load_image(path)
        messages = []
        # Fully transparent pixels are kept unchanged and excluded from background statistics.
        valid = source.alpha > 0 if source.alpha is not None else None
        output, difference = color_dejitter(source.bgr, **asdict(st), valid_mask=valid, report=messages.append)
        folder = Path(tempfile.mkdtemp(prefix='image-', dir=result_dir()))
        stem = re.sub(r'[^\w\-. ]', '_', Path(path).stem)[:80] or 'image'
        clean_path, diff_path = folder / f'{stem}_clean.png', folder / f'{stem}_diff.png'
        save_png(clean_path, output, source)
        save_png(diff_path, difference)
        preview = json.dumps({'original': preview_data(source.bgr, source.alpha), 'result': preview_data(output, source.alpha)})
        elapsed = time.perf_counter() - started
        h, w = output.shape[:2]
        status = f'完成 · {w} × {h} · {elapsed:.2f} 秒。下载文件保留原尺寸，HSB 观察工具不会影响导出。'
        if st.overall_strength == 0:
            status += '\n整体强度为 0：像素未修正。'
        if messages:
            status += '\n' + '\n'.join(messages)
        return preview, str(diff_path), [str(clean_path), str(diff_path)], status
    except (ValueError, OSError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc


def build_color_app():
    store = PresetStore()
    try:
        names = list(store.all())
        initial_status = '默认只清理色度；开启 V 后才处理亮度。图片仅在本机处理。'
    except ValueError as exc:
        names, initial_status = list(BUILTINS), str(exc)
    with gr.Blocks(title='狗梅小工具 · DOISH', analytics_enabled=False, delete_cache=(3600, 86400)) as demo:
        gr.Markdown('# 🐶 狗梅小工具\n让色块更干净，保留你在意的细节。')
        source_path = gr.State(None)
        with gr.Row():
            with gr.Column(scale=4, min_width=300):
                with gr.Group():
                    preset = gr.Dropdown(names, value='默认', label='预设配置')
                    with gr.Row():
                        preset_name = gr.Textbox(label='自定义预设名称', placeholder='例如：柔和肤色', max_lines=1)
                        save = gr.Button('保存预设', size='sm')
                        delete = gr.Button('删除预设', size='sm')
                    preset_status = gr.Textbox(label='预设状态', interactive=False, max_lines=2)
                upload = gr.File(label='上传图片 · 拖入 PNG / JPG / WebP 等', file_types=['image'], type='filepath', elem_id='image-upload')
                with gr.Accordion('从剪贴板粘贴图片', open=False):
                    clipboard = gr.Image(label='点击此处粘贴图片', sources=['clipboard'], type='filepath', image_mode='RGBA', format='png', height=160)
                    gr.Markdown('带 ICC / AI 参数的文件请使用上面的文件上传，以保留原始信息。')
                controls = {}
                def slider(name, lo, hi, step, label):
                    controls[name] = gr.Slider(lo, hi, value=getattr(Settings(), name), step=step, label=label)
                def check(name, label):
                    controls[name] = gr.Checkbox(value=getattr(Settings(), name), label=label)
                slider('overall_strength', 0, 1, 0.01, '整体强度 · 0 = 原图，1 = 全部修正')
                with gr.Accordion('H · 色度降噪', open=True):
                    slider('h_denoise_threshold', 0, 40, 0.5, '去噪阈值 · 6–10 轻度，10–15 中度')
                    slider('h_smooth_radius', 1, 30, 1, '搜索半径 · 越大越慢')
                    slider('h_denoise_strength', 0, 1, 0.01, '色度修正比例')
                    check('h_use_soft_threshold', '软阈值 · 渐进保护较大色差')
                with gr.Accordion('S · 保边色度平滑', open=False):
                    check('s_enabled', '启用 S 平滑')
                    slider('s_smooth_radius', 1, 60, 1, '平滑半径')
                    slider('s_denoise_threshold', 0, 0.4, 0.005, '平滑色差阈值')
                    slider('s_denoise_strength', 0, 1, 0.01, '平滑修正比例')
                    check('s_use_soft_threshold', '软阈值保护')
                with gr.Accordion('V · 亮度与背景', open=False):
                    check('v_enabled', '启用 V 亮度修正')
                    slider('v_smooth_radius', 1, 30, 1, '亮度搜索 / 背景柔边半径')
                    slider('v_denoise_threshold', 0, 0.2, 0.005, '亮度去噪阈值')
                    slider('v_denoise_strength', 0, 1, 0.01, '亮度 / 背景修正比例')
                    check('v_use_soft_threshold', '软阈值保护')
                    check('background_enabled', '同时统一近似纯色背景 · 需要分割模型')
                    gr.Markdown('优先保留渐变、阴影和纹理。背景判定是启发式，请使用前后对比检查。')
                with gr.Accordion('高级设置与模型', open=False):
                    slider('edge_protection', 0, 200, 1, '边缘保护 · NL-Means 模板大小')
                    slider('low_sat_skip', 0, 1, 0.01, '低饱和度色度保护阈值')
                    model_status = gr.Textbox(value='模型文件已存在，启用时验证加载。' if model_path().is_file() else '分割模型未安装。普通色度降噪无需模型。', label='分割模型', interactive=False)
                    get_model = gr.Button('下载 / 校验分割模型 · 约 176 MB')
                run = gr.Button('开始处理', variant='primary', elem_id='process-button')
            with gr.Column(scale=7, min_width=320):
                compare = gr.HTML('{}', html_template=(RESOURCES/'web/compare.html').read_text('utf-8'), css_template=(RESOURCES/'web/compare.css').read_text('utf-8'), js_on_load=(RESOURCES/'web/compare.js').read_text('utf-8'), elem_id='comparison')
                status = gr.Textbox(value=initial_status, label='处理状态', lines=3, interactive=False)
                downloads = gr.File(label='下载 · 修正结果和 5 倍差异图', file_count='multiple', interactive=False)
                with gr.Accordion('差异图 · 已放大 5 倍', open=False):
                    difference = gr.Image(label='变化区域', interactive=False, type='filepath', format='png')
        ordered = [controls[name] for name in FIELDS]
        upload.change(prepare_image, upload, [source_path, compare, difference, downloads, status], api_name='load_image', concurrency_limit=1)
        clipboard.change(prepare_image, clipboard, [source_path, compare, difference, downloads, status], api_name=False, concurrency_limit=1)
        run.click(process_image, [source_path, *ordered], [compare, difference, downloads, status], api_name='process', concurrency_limit=1, concurrency_id='processing')

        def load_preset(name):
            try:
                selected = BUILTINS[name] if name in BUILTINS else store.all()[name]
                return [*[getattr(selected, key) for key in FIELDS], f'已载入「{name}」，点击开始处理应用。', None]
            except (ValueError, KeyError) as exc:
                raise gr.Error(str(exc)) from exc
        preset.input(load_preset, preset, [*ordered, status, downloads], api_name=False)

        def save_preset(name, *values):
            try:
                name = store.save(name, settings_from(values))
                return gr.Dropdown(choices=list(store.all()), value=name), f'已保存「{name}」。'
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
        save.click(save_preset, [preset_name, *ordered], [preset, preset_status], api_name=False)

        def delete_preset(name):
            try:
                store.delete(name)
                return [gr.Dropdown(choices=list(store.all()), value='默认'), f'已删除「{name}」，恢复默认参数。', *[getattr(Settings(), k) for k in FIELDS], None, '参数已恢复默认，请重新处理。']
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
        delete.click(delete_preset, preset, [preset, preset_status, *ordered, downloads, status], api_name=False)
        for component in ordered:
            component.input(lambda: (None, '参数已更改；当前对比为上次结果，请点击开始处理。'), outputs=[downloads, status], api_name=False, queue=False)

        def install_model(progress=gr.Progress()):
            try:
                download(lambda message: progress(None, desc=message))
                return '模型已下载，SHA-256 校验通过；下次启用背景纯色化时生效。'
            except Exception as exc:
                return f'下载失败：{exc}。可以稍后重试，或按启动指南手动下载。'
        get_model.click(install_model, outputs=model_status, api_name=False, concurrency_limit=1)
        gr.Markdown('本机处理 · H/S 在 Lab 中保留亮度；色彩转换和 8-bit 导出仍可能有微小误差。')
    demo.queue(default_concurrency_limit=1, max_size=16)
    return demo


def build_app():
    from hive_lab.ui import build_lab
    # Construct outside the parent context: nested Blocks otherwise auto-render
    # once on exit and a second time on the explicit render() call.
    lab_ui = build_lab()
    color_ui = build_color_app()
    with gr.Blocks(title='DOISH Image Lab', analytics_enabled=False, delete_cache=(3600, 86400)) as demo:
        gr.Markdown('# DOISH Image Lab\n以实测结果选择处理方案。')
        with gr.Tab('Hive 实测与处理'):
            lab_ui.render()
        with gr.Tab('DOISH 色彩去抖'):
            color_ui.render()
    demo.queue(default_concurrency_limit=1, max_size=16)
    return demo


def main(open_browser=False):
    parser = argparse.ArgumentParser(description='DOISH Image Lab · 图像处理与 Hive 实测')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('端口必须在 1–65535 之间')
    demo = build_app()
    try:
        demo.launch(server_name=args.host, server_port=args.port, inbrowser=open_browser and not args.no_browser, share=False,
                    max_file_size='100mb', theme=gr.themes.Soft(primary_hue='green', neutral_hue='stone'),
                    css='.gradio-container {max-width:1400px!important} footer {display:none!important}')
    except OSError as exc:
        raise SystemExit(f'启动失败：{exc}\n如果端口被占用，请使用 --port 7861；无需结束其他程序。') from exc


if __name__ == '__main__':
    main()
