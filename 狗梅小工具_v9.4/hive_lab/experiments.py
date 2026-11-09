"""Persistent experiments, exact file hashes, score imports and honest ranking."""
from copy import deepcopy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import shutil
import uuid
import zipfile

from PIL import Image

from color_dejitter.images import load_image, save_png
from color_dejitter.presets import data_dir
from . import VERSION
from .pipeline import RECIPES, process, quality

CSV_FIELDS = ['id', 'file', 'sha256', 'hive_ai_percent']


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
        return h.hexdigest()


def root_dir():
    root = data_dir() / 'experiments'
    root.mkdir(parents=True, exist_ok=True)
    return root


def folder_for(experiment_id):
    if not isinstance(experiment_id, str) or len(experiment_id) != 32 or any(c not in '0123456789abcdef' for c in experiment_id):
        raise ValueError('无效实验编号')
    return root_dir() / experiment_id


def write_manifest(folder, manifest):
    temporary = folder / 'manifest.part'
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(folder / 'manifest.json')


def read_manifest(experiment_id):
    folder = folder_for(experiment_id)
    return folder, json.loads((folder / 'manifest.json').read_text('utf-8'))


def checked_file(folder, row):
    path = (folder / row['file']).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file() or digest(path) != row['sha256']:
        raise ValueError('测试图片已变更或丢失，请重新生成实验')
    return path


def generate(paths, recipes, seed=2026, add_doish=False, progress=None):
    if not paths or len(paths) > 4:
        raise ValueError('每个实验请选择 1–4 张图片')
    if not recipes or len(set(recipes)) != len(recipes) or any(r not in RECIPES for r in recipes):
        raise ValueError('请选择至少一个有效且不重复的处理方案')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError('随机种子必须是 0–4294967295 的整数')
    # Decode one image at a time; preflight dimensions without retaining a batch in RAM.
    pixels = 0
    for path in paths:
        with Image.open(path) as image:
            pixels += image.width * image.height
    if pixels > 32_000_000:
        raise ValueError('单次实验总像素不能超过 3200 万，请分批处理')
    experiment_id = uuid.uuid4().hex
    folder = folder_for(experiment_id)
    folder.mkdir()
    manifest = {'schema': 1, 'version': VERSION, 'id': experiment_id, 'created_at': now(),
                'seed': seed, 'add_doish': bool(add_doish), 'recipes': {r: asdict(RECIPES[r]) for r in recipes},
                'rows': [], 'score_history': [], 'errors': []}
    try:
        for index, path in enumerate(paths, 1):
            source = load_image(path)
            sample = f'{index:02d}'
            subdir = folder / sample
            subdir.mkdir()
            # Actual uploaded bytes are a separate provenance/encoding control.
            suffix = Path(path).suffix.lower()
            uploaded = subdir / ('uploaded' + suffix)
            shutil.copyfile(path, uploaded)
            variants = ['uploaded', 'control', *recipes]
            for variant in variants:
                if progress:
                    progress(f'图片 {index}/{len(paths)} · {variant}')
                out = source.bgr if variant in ('uploaded', 'control') else process(source, variant, seed, add_doish)
                target = uploaded if variant == 'uploaded' else subdir / f'{variant}.png'
                if variant != 'uploaded':
                    save_png(target, out, source)
                manifest['rows'].append({
                    'id': f'{sample}_{variant}', 'sample': sample, 'source_name': Path(path).name,
                    'variant': variant, 'label': {'uploaded': '上传原文件', 'control': '统一 PNG 对照'}.get(variant, RECIPES[variant].label if variant in RECIPES else variant),
                    'file': target.relative_to(folder).as_posix(), 'sha256': digest(target),
                    'width': out.shape[1], 'height': out.shape[0],
                    'quality': quality(source.bgr, out, source.alpha), 'score': None})
            # Source is released before decoding the next image.
        write_manifest(folder, manifest)
    except Exception:
        # Keep partial output inspectable, but do not list it as a completed experiment.
        (folder / 'FAILED.txt').write_text('实验生成未完成，请重新生成。', encoding='utf-8')
        raise
    return experiment_id


def score_csv(manifest):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in manifest['rows']:
        writer.writerow({**{k: row[k] for k in CSV_FIELDS[:-1]},
                         'hive_ai_percent': '' if row['score'] is None else row['score']['percent']})
    return stream.getvalue()


def import_scores(experiment_id, csv_path, batch_label):
    if not batch_label or not batch_label.strip():
        raise ValueError('请注明网站检测日期 / 批次，例如 2026-11-09 晚间 Hive 网站')
    if Path(csv_path).stat().st_size > 1024 * 1024:
        raise ValueError('分数表不能超过 1 MiB')
    folder, saved = read_manifest(experiment_id)
    manifest = deepcopy(saved)
    lookup = {row['id']: row for row in manifest['rows']}
    seen, count = set(), 0
    with open(csv_path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError('分数表列名或顺序不匹配，请填写本实验导出的 scores.csv')
        for entry in reader:
            row_id = entry.get('id')
            if row_id in seen or row_id not in lookup:
                raise ValueError('分数表包含重复或不属于本实验的编号')
            seen.add(row_id)
            row = lookup[row_id]
            if entry.get('sha256') != row['sha256'] or entry.get('file') != row['file']:
                raise ValueError('文件哈希 / 文件名不匹配，不能将分数套用到另一张图')
            checked_file(folder, row)
            value = entry.get('hive_ai_percent')
            if value is None:
                raise ValueError('分数表行不完整')
            if not value.strip():
                continue
            percent = float(value.strip().removesuffix('%'))
            if not math.isfinite(percent) or not 0 <= percent <= 100:
                raise ValueError('Hive 分数必须在 0–100 之间；空白表示未测，不能用 0 代替')
            row['score'] = {'percent': percent, 'channel': 'manual', 'batch': batch_label.strip()[:160],
                            'model': 'hivedetect.ai (manual)', 'model_version': 'unknown', 'at': now()}
            count += 1
    if count == 0:
        raise ValueError('没有可导入的分数')
    manifest['score_history'].append({'at': now(), 'rows': [{ 'id': r['id'], 'score': r['score']} for r in saved['rows'] if r['score']]})
    write_manifest(folder, manifest)
    return count


def detect_all(experiment_id, key, consent, detector=None, progress=None):
    if not consent:
        raise ValueError('请先勾选上传许可：图片将发送到 Hive，可能产生 API 费用')
    if not key or not key.strip():
        raise ValueError('请填写 Hive AI 图像检测项目密钥')
    from .hive import detect
    detector = detector or detect
    folder, manifest = read_manifest(experiment_id)
    # Preflight every artifact before sending any request.
    for row in manifest['rows']:
        checked_file(folder, row)
    batch = uuid.uuid4().hex
    manifest['score_history'].append({'at': now(), 'rows': [{'id': r['id'], 'score': r['score']} for r in manifest['rows'] if r['score']]})
    manifest['errors'] = []
    for index, row in enumerate(manifest['rows']):
        if progress:
            progress(f'Hive {index + 1}/{len(manifest["rows"])} · {row["id"]}')
        try:
            result = detector(folder / row['file'], key)
            row['score'] = {**result, 'channel': 'api', 'batch': batch, 'at': now()}
        except ValueError as exc:
            manifest['errors'].append({'id': row['id'], 'message': str(exc), 'at': now()})
            write_manifest(folder, manifest)
            return f'检测在 {row["id"]} 停止：{exc}。之前完成的分数已保存；再次点击会重新检测全组。'
        write_manifest(folder, manifest)
    return f'已完成 {len(manifest["rows"])} 次 Hive 检测。'


def comparable(a, b):
    return a is not None and b is not None and all(a.get(k) == b.get(k) for k in ('channel', 'batch', 'model', 'model_version'))


def evaluate(manifest, min_psnr=32, min_ssim=0.95, min_drop=1):
    values = (min_psnr, min_ssim, min_drop)
    if not all(math.isfinite(float(v)) for v in values) or not 0 <= min_psnr <= 100 or not 0 <= min_ssim <= 1 or not 0 <= min_drop <= 100:
        raise ValueError('画质 / 分数门槛无效')
    baselines = {r['sample']: r for r in manifest['rows'] if r['variant'] == 'control'}
    winners, rows = {}, []
    for row in manifest['rows']:
        q, score = row['quality'], row['score']
        baseline = baselines[row['sample']]['score']
        drop = baseline['percent'] - score['percent'] if comparable(score, baseline) else None
        quality_ok = (q['psnr_db'] is None or q['psnr_db'] >= min_psnr) and q['ssim_1024'] >= min_ssim
        control = row['variant'] in ('uploaded', 'control')
        if control:
            verdict = '对照'
        elif score is None:
            verdict = '待检测'
        elif drop is None:
            verdict = '缺同批次对照'
        elif not quality_ok:
            verdict = '画质门槛未通过'
        elif drop <= 0 or drop < min_drop:
            verdict = '未达到改善门槛'
        else:
            verdict = '本图有改善 · 待复验'
            previous = winners.get(row['sample'])
            if previous is None or score['percent'] < previous['score']['percent']:
                winners[row['sample']] = row
        rows.append([row['id'], row['label'], None if score is None else round(score['percent'], 6),
                     None if drop is None else round(drop, 6), q['psnr_db'], q['ssim_1024'], verdict])
    return rows, winners


def export(experiment_id, min_psnr=32, min_ssim=0.95, min_drop=1):
    folder, manifest = read_manifest(experiment_id)
    rows, winners = evaluate(manifest, min_psnr, min_ssim, min_drop)
    for row in manifest['rows']:
        checked_file(folder, row)
    (folder / 'scores.csv').write_text(score_csv(manifest), encoding='utf-8-sig')
    (folder / '使用说明.txt').write_text(
        'DOISH Image Lab\n先分别上传每张图片到 https://hivedetect.ai/，不要上传拼图。\n'
        'uploaded 是上传原文件；control 是统一 PNG 对照。候选处理效果与 control 比较。\n'
        '在 scores.csv 的 hive_ai_percent 列填入网站 AI 百分比（0–100），未测留空。\n'
        '不要改编号、文件名、哈希；回到本工具选择对应实验，导入 CSV 并注明同一检测批次。\n'
        '网站与 API 结果不混排；同一批次请使用同一网站/版本。每次 API 检测重新计费全组。\n'
        'PNG 对照与候选保留可读取的 PNG 文本和 ICC；重新编码不能保留原文件所有 EXIF/C2PA 数据。\n'
        '这不是已验证的 Hive 绕过工具。分数下降不代表来源改变，画质指标也不能代替目视检查。\n'
        '先在探索样本筛选方案，再在新图片复验；不要把单图结果当成普遍效果。\n', encoding='utf-8')
    heading = ['编号', '方案', 'Hive AI %', '下降百分点', 'PSNR dB', 'SSIM ≤1024', '结论']
    table = '<tr>' + ''.join(f'<th>{h}</th>' for h in heading) + '</tr>'
    for row in rows:
        table += '<tr>' + ''.join(f'<td>{html.escape(str(v)) if v is not None else "—"}</td>' for v in row) + '</tr>'
    selected = '；'.join(f'{sample}: {r["id"]}' for sample, r in winners.items()) or '没有已测且通过门槛的改善候选'
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DOISH Image Lab 报告</title>
<style>body{{font:16px system-ui;margin:40px;color:#182b38}}table{{border-collapse:collapse}}td,th{{padding:10px;border:1px solid #ccd6df}}th{{background:#edf4f7}}p{{max-width:1000px}}</style>
<h1>DOISH Image Lab</h1><p>实验 {manifest['id']} · {html.escape(manifest['created_at'])}</p>
<p>门槛：PSNR ≥ {min_psnr} dB，SSIM ≥ {min_ssim}，下降 ≥ {min_drop} 个百分点。下降值与同批次统一 PNG 对照比较。</p>
<p>{html.escape(selected)}</p><table>{table}</table><p>空白表示未测。PSNR 无差异时显示 —（无穷大）。SSIM 在最长边 1024 像素计算。入选仅代表本图本次实测改善，需要新样本复验和目视检查。</p></html>'''
    (folder / 'report.html').write_text(report, encoding='utf-8')
    archive = folder / f'test-pack-{uuid.uuid4().hex[:8]}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for name in ('manifest.json', 'scores.csv', 'report.html', '使用说明.txt'):
            zipped.write(folder / name, name)
        for row in manifest['rows']:
            zipped.write(folder / row['file'], row['file'])
    return archive, folder / 'report.html', folder / 'scores.csv'


def list_experiments():
    choices = []
    for path in sorted(root_dir().glob('*/manifest.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text('utf-8'))
            choices.append((f'{data["created_at"][:19]} · {data["id"][:8]} · {len(data["rows"])} 张', data['id']))
        except (ValueError, KeyError, OSError):
            continue
    return choices
