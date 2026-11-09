import csv
import io
import json
from pathlib import Path
import zipfile

import httpx
import numpy as np
from PIL import Image, PngImagePlugin
import pytest

from color_dejitter.images import load_image
from hive_lab import experiments as lab
from hive_lab.hive import detect, parse_response
from hive_lab.pipeline import RECIPES, process, quality


@pytest.fixture
def sample(tmp_path, monkeypatch):
    monkeypatch.setenv('DOISH_DATA_DIR', str(tmp_path / 'data'))
    image = np.random.default_rng(27).integers(40, 210, (32, 48, 4), dtype=np.uint8)
    image[..., 3] = 255
    image[:2, :, 3] = 0
    path = tmp_path / '图像.png'
    info = PngImagePlugin.PngInfo()
    info.add_itxt('parameters', 'test provenance')
    Image.fromarray(image).save(path, pnginfo=info)
    return path


@pytest.mark.parametrize('recipe', list(RECIPES))
def test_recipe_determinism_dimensions_alpha_and_effect(sample, recipe):
    source = load_image(sample)
    a, b = process(source, recipe, 9), process(source, recipe, 9)
    np.testing.assert_array_equal(a, b)
    assert a.shape == source.bgr.shape and a.dtype == np.uint8
    np.testing.assert_array_equal(a[:2], source.bgr[:2])
    assert np.any(a[2:] != source.bgr[2:])


def test_seed_only_affects_random_component(sample):
    source = load_image(sample)
    np.testing.assert_array_equal(process(source, 'resample', 1), process(source, 'resample', 2))
    assert np.any(process(source, 'texture', 1) != process(source, 'texture', 2))
    with pytest.raises(ValueError):
        process(source, 'texture', -1)


def test_quality_identity_and_damage(sample):
    source = load_image(sample)
    same = quality(source.bgr, source.bgr, source.alpha)
    assert same == {'mae_rgb': 0.0, 'psnr_db': None, 'ssim_1024': 1.0}
    damaged = quality(source.bgr, np.zeros_like(source.bgr), source.alpha)
    assert damaged['psnr_db'] < 20 and damaged['ssim_1024'] < 0.5


def make_experiment(sample):
    return lab.generate([sample], ['texture', 'balanced'])


def write_scores(folder, manifest, values):
    stream = io.StringIO(lab.score_csv(manifest))
    rows = list(csv.DictReader(stream))
    path = folder / 'filled.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=lab.CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            row['hive_ai_percent'] = values.get(row['id'], '')
            writer.writerow(row)
    return path


def test_bundle_original_bytes_metadata_and_blank_scores(sample):
    run = make_experiment(sample)
    folder, manifest = lab.read_manifest(run)
    assert lab.digest(folder / manifest['rows'][0]['file']) == lab.digest(sample)
    for row in manifest['rows'][1:]:
        with Image.open(folder / row['file']) as image:
            assert image.text['parameters'] == 'test provenance'
            np.testing.assert_array_equal(np.asarray(image)[..., 3], np.asarray(Image.open(sample))[..., 3])
    archive, report, score_file = lab.export(run)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert len([n for n in zipped.namelist() if n.endswith('.png')]) == 4
        assert json.loads(zipped.read('manifest.json'))['id'] == run
    assert '没有已测' in report.read_text('utf-8')
    assert all(r['hive_ai_percent'] == '' for r in csv.DictReader(score_file.open(encoding='utf-8-sig')))


def test_import_rank_and_quality_gate(sample):
    run = make_experiment(sample)
    folder, manifest = lab.read_manifest(run)
    path = write_scores(folder, manifest, {'01_control': '99%', '01_texture': '20', '01_balanced': '99'})
    assert lab.import_scores(run, path, 'same website batch') == 3
    _, saved = lab.read_manifest(run)
    rows, winners = lab.evaluate(saved, 0, 0, 1)
    assert winners['01']['variant'] == 'texture'
    assert rows[2][3] == 79
    assert not lab.evaluate(saved, 100, 1, 1)[1]
    # A missing baseline or a different provider must never produce a winner.
    saved['rows'][1]['score']['channel'] = 'api'
    assert not lab.evaluate(saved, 0, 0, 1)[1]


@pytest.mark.parametrize('value', ['nan', '-1', '101', 'inf'])
def test_invalid_scores_are_atomic(sample, value):
    run = make_experiment(sample)
    folder, manifest = lab.read_manifest(run)
    before = (folder / 'manifest.json').read_bytes()
    path = write_scores(folder, manifest, {'01_control': '90', '01_texture': value})
    with pytest.raises(ValueError):
        lab.import_scores(run, path, 'batch')
    assert (folder / 'manifest.json').read_bytes() == before


def test_zero_is_valid_missing_is_not_zero_and_no_improvement(sample):
    run = make_experiment(sample)
    folder, manifest = lab.read_manifest(run)
    path = write_scores(folder, manifest, {'01_control': 0, '01_texture': 0})
    lab.import_scores(run, path, 'batch')
    _, saved = lab.read_manifest(run)
    assert saved['rows'][1]['score']['percent'] == 0
    assert saved['rows'][3]['score'] is None
    assert not lab.evaluate(saved, 0, 0, 0)[1]


def test_tampered_image_rejected_before_upload(sample):
    run = make_experiment(sample)
    folder, manifest = lab.read_manifest(run)
    (folder / manifest['rows'][2]['file']).write_bytes(b'changed')
    calls = []
    with pytest.raises(ValueError, match='变更'):
        lab.detect_all(run, 'key', True, detector=lambda *args: calls.append(args))
    assert not calls


def payload(score=0.97):
    return {'status': [{'status': {'code': '0'}, 'response': {
        'input': {'model': 'AI art and deepfake detection', 'model_version': 1},
        'output': [{'classes': [{'class': 'deepfake', 'score': 0.1}, {'class': 'ai_generated', 'score': score}]}]}}]}


def test_official_adapter_exact_field_and_no_redirect(sample):
    def handler(request):
        assert request.url == 'https://api.thehive.ai/api/v2/task/sync'
        assert request.headers['authorization'] == 'Token secret'
        assert sample.read_bytes() in request.read()
        return httpx.Response(200, json=payload())
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = detect(sample, 'secret', client=client)
    assert result['percent'] == 97
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(302, headers={'Location': 'https://example.com'}))) as client:
        with pytest.raises(ValueError, match='302'):
            detect(sample, 'secret', client=client)


@pytest.mark.parametrize('body', [{}, {'status': [{'status': {'code': '1'}}]}, payload(float('nan')), payload(1.1)])
def test_api_failure_never_turns_into_zero(body):
    with pytest.raises(ValueError):
        parse_response(body)


def test_partial_api_failure_preserves_scores_and_no_key(sample):
    run = make_experiment(sample)
    calls = []
    def detector(path, key):
        calls.append(path)
        if len(calls) == 3:
            raise ValueError('service unavailable')
        return {'percent': 99, 'model': 'test', 'model_version': 1}
    with pytest.raises(ValueError):
        lab.detect_all(run, 'SUPER_SECRET', False, detector)
    assert not calls
    assert '停止' in lab.detect_all(run, 'SUPER_SECRET', True, detector)
    folder, manifest = lab.read_manifest(run)
    assert sum(r['score'] is not None for r in manifest['rows']) == 2
    assert 'SUPER_SECRET' not in (folder / 'manifest.json').read_text('utf-8')
    assert not lab.evaluate(manifest, 0, 0, 1)[1]


def test_path_traversal_and_invalid_recipe_rejected(sample):
    with pytest.raises(ValueError):
        lab.read_manifest('../elsewhere')
    with pytest.raises(ValueError):
        lab.generate([sample], ['unknown'])


def test_lab_build_and_persistent_ui(sample):
    from hive_lab.ui import build_lab, compare, show
    run = make_experiment(sample)
    demo = build_lab()
    labels = {getattr(b, 'label', '') for b in demo.blocks.values()}
    assert '当前实验 / 历史记录' in labels and 'Hive API Key' in labels
    result = show(run)
    assert len(result[0]) == 4 and len(result[1]) == 3
    assert 'original' in json.loads(compare(run, '01_texture'))
    assert any(value == run for _, value in lab.list_experiments())
