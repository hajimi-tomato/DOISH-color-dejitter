"""Local UI for generation, manual/API measurements and fidelity-constrained selection."""
import json
from pathlib import Path
import sys

import gradio as gr

from color_dejitter.images import load_image, preview_data
from . import experiments as lab
from .pipeline import RECIPES

HEADERS = ['编号', '方案', 'Hive AI %', '下降百分点', 'PSNR dB', 'SSIM ≤1024', '结论']


def show(experiment_id, psnr=32, ssim=0.95, drop=1):
    if not experiment_id:
        raise gr.Error('请先生成或打开实验')
    try:
        folder, manifest = lab.read_manifest(experiment_id)
        rows, winners = lab.evaluate(manifest, psnr, ssim, drop)
        downloads = [str(p) for p in lab.export(experiment_id, psnr, ssim, drop)]
        downloads += [str(folder / r['file']) for r in winners.values()]
        choices = [(f'{r["id"]} · {r["label"]}', r['id']) for r in manifest['rows']]
        measured = sum(r['score'] is not None for r in manifest['rows'])
        text = f'实验 {experiment_id[:8]} · 已测 {measured}/{len(rows)} · {len(winners)} 张原图找到通过门槛的改善候选。'
        if not winners:
            text += '\n目前没有已验证且通过门槛的改善结果。'
        text += '\n实验已保存在本机，重启后可从历史列表恢复。'
        if manifest['errors']:
            text += '\n' + manifest['errors'][-1]['message']
        return rows, downloads, text, gr.Dropdown(choices=choices, value=None), '{}'
    except (ValueError, OSError, KeyError) as exc:
        raise gr.Error(str(exc)) from exc


def compare(experiment_id, row_id):
    if not experiment_id or not row_id:
        return '{}'
    folder, manifest = lab.read_manifest(experiment_id)
    row = next((r for r in manifest['rows'] if r['id'] == row_id), None)
    if row is None:
        raise gr.Error('没有找到此候选图片')
    base = next(r for r in manifest['rows'] if r['sample'] == row['sample'] and r['variant'] == 'control')
    original = load_image(lab.checked_file(folder, base))
    result = load_image(lab.checked_file(folder, row))
    return json.dumps({'original': preview_data(original.bgr, original.alpha),
                       'result': preview_data(result.bgr, result.alpha)})


def build_lab():
    resources = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown('## 图像处理与 Hive 实测\n生成候选 → 检测原图和候选 → 按实测改善与画质筛选。未测的方案不会被标记为有效。')
        with gr.Row():
            history = gr.Dropdown(choices=lab.list_experiments(), label='当前实验 / 历史记录', value=None, scale=5)
            refresh = gr.Button('刷新历史', scale=1)
        with gr.Row():
            with gr.Column(scale=4):
                files = gr.File(file_count='multiple', file_types=['image'], type='filepath', label='输入图片 · 最多 4 张，总计 3200 万像素')
                recipes = gr.CheckboxGroup([(r.label, key) for key, r in RECIPES.items()],
                                          value=['deep_grain', 'deep_texture', 'deep_hive'], label='候选处理方案')
                extra = gr.Checkbox(False, label='组合前额外执行 DOISH 色彩去抖')
                seed = gr.Number(value=2026, precision=0, minimum=0, maximum=2**32 - 1, label='颗粒随机种子 · 相同种子可复现')
                create = gr.Button('生成实验与测试包', variant='primary')
                gr.Markdown('深度候选需先在主标签下载 DRUNet 模型，单张最多 1677 万像素。每张输入额外包含上传原文件、统一 PNG 对照；较强处理可能损失细节。')
                with gr.Accordion('录入 Hive 网站结果', open=True):
                    gr.Markdown('下载测试包 → 在 [Hive 网站](https://hivedetect.ai/) 分别检测 → 填写 scores.csv 的 AI 百分比 → 导入。请在同一批次测完对应对照。')
                    csv = gr.File(label='填好的 scores.csv', file_types=['.csv'], type='filepath')
                    batch = gr.Textbox(label='检测日期 / 批次', placeholder='例如 2026-11-09 晚间 Hive 网站')
                    ingest = gr.Button('导入分数并筛选')
                with gr.Accordion('Hive 官方 API · 可选', open=False):
                    gr.Markdown('使用 AI 图像检测项目的 **V2 项目密钥**。每次点击会检测当前实验的所有文件，包括对照；网站与 API 分数分别记录，不混排。')
                    key = gr.Textbox(type='password', label='Hive API Key', placeholder='仅本次请求使用，不写入实验或导出文件')
                    consent = gr.Checkbox(False, label='将当前实验全部图片上传给 Hive；我了解可能产生 API 费用')
                    detect = gr.Button('上传全组并检测')
                with gr.Accordion('筛选门槛', open=False):
                    psnr = gr.Slider(20, 50, value=32, step=0.5, label='最低 PSNR · dB')
                    ssim = gr.Slider(0.8, 1, value=0.95, step=0.005, label='最低 SSIM · 最长边 1024 计算')
                    drop = gr.Slider(0.1, 30, value=1, step=0.1, label='至少下降多少个百分点')
                    rerank = gr.Button('按当前门槛重新筛选')
            with gr.Column(scale=7):
                status = gr.Textbox(value='从左侧添加图片开始。现有色彩工具位于「DOISH 色彩去抖」标签。', label='实验状态', lines=3, interactive=False)
                table = gr.Dataframe(headers=HEADERS, datatype=['str', 'str', 'number', 'number', 'number', 'number', 'str'], interactive=False, label='实测结果 · 空白 = 未测 / 不可比较', wrap=True)
                selection = gr.Dropdown(label='查看候选与 PNG 对照', choices=[], value=None)
                viewer = gr.HTML('{}', html_template=(resources/'web/compare.html').read_text('utf-8'),
                                 css_template=(resources/'web/compare.css').read_text('utf-8'),
                                 js_on_load=(resources/'web/compare.js').read_text('utf-8'))
                downloads = gr.File(label='下载测试包 / 报告 / 分数表 / 入选图片', file_count='multiple', interactive=False)
                gr.Markdown('**结果边界**：分数降低只代表本图本次检测的变化；需在新图片复验。画质指标不能代替目视检查。此工具与 Hive 官方无隶属关系。')
        outputs = [table, downloads, status, selection, viewer]
        limits = [psnr, ssim, drop]

        def create_run(paths, chosen, random_seed, doish, p, s, d, progress=gr.Progress()):
            try:
                if random_seed != int(random_seed):
                    raise ValueError('种子必须是整数')
                experiment_id = lab.generate(paths, chosen, int(random_seed), doish,
                                             lambda message: progress(None, desc=message))
                return [gr.Dropdown(choices=lab.list_experiments(), value=experiment_id), *show(experiment_id, p, s, d)]
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from exc

        def import_run(experiment_id, path, batch_name, p, s, d):
            if not experiment_id or not path:
                raise gr.Error('请先选择实验和填好的 CSV')
            try:
                lab.import_scores(experiment_id, path, batch_name)
                return show(experiment_id, p, s, d)
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from exc

        def api_run(experiment_id, token, permitted, p, s, d, progress=gr.Progress()):
            if not experiment_id:
                raise gr.Error('请先选择实验')
            try:
                message = lab.detect_all(experiment_id, token, permitted, progress=lambda text: progress(None, desc=text))
                result = list(show(experiment_id, p, s, d))
                result[2] = message + '\n' + result[2]
                return [*result, '']
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from exc

        options = {'concurrency_limit': 1, 'concurrency_id': 'deep-processing'}
        create.click(create_run, [files, recipes, seed, extra, *limits], [history, *outputs], api_name='lab_generate', **options)
        history.input(show, [history, *limits], outputs, api_name='lab_open', **options)
        refresh.click(lambda: gr.Dropdown(choices=lab.list_experiments()), outputs=history, api_name=False, **options)
        ingest.click(import_run, [history, csv, batch, *limits], outputs, api_name='lab_import_scores', **options)
        # Secret-bearing endpoint is not exposed via Gradio's public API schema.
        detect.click(api_run, [history, key, consent, *limits], [*outputs, key], api_name=False, **options)
        rerank.click(show, [history, *limits], outputs, api_name='lab_evaluate', **options)
        selection.input(compare, [history, selection], viewer, api_name='lab_compare', **options)
    return demo
