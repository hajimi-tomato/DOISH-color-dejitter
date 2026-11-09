"""Application callbacks: same data path used by Gradio, including exports."""
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def application(tmp_path, monkeypatch):
    monkeypatch.setenv('DOISH_DATA_DIR', str(tmp_path/'data'))
    import app
    yield app
    if app._results is not None:
        app._results.cleanup()
        app._results = None


def test_load_process_and_download_zero_strength(application, tmp_path):
    from color_dejitter.core import Settings
    image = np.random.default_rng(12).integers(0,256,(32,48,4),dtype=np.uint8)
    path = tmp_path/'input.png'
    Image.fromarray(image).save(path)
    source, preview, _, _, status = application.prepare_image(path)
    assert source and json.loads(preview)['original'].startswith('data:image/png;')
    settings = Settings(overall_strength=0)
    before, difference, downloads, message = application.process_image(source,*asdict(settings).values())
    assert '像素未修正' in message and len(downloads)==2
    assert all(Path(p).is_file() for p in downloads)
    with Image.open(downloads[0]) as result:
        np.testing.assert_array_equal(np.asarray(result),image)
    with Image.open(difference) as result:
        assert not np.asarray(result).any()
    assert json.loads(before)['result']


def test_invalid_upload_clears_previous_state(application,tmp_path):
    path=tmp_path/'invalid.png';path.write_text('not an image')
    source, preview, difference, downloads, status=application.prepare_image(path)
    assert source is None and preview=='{}' and difference is None and downloads is None
    assert '无法载入' in status


def test_app_builds_with_all_documented_controls(application):
    demo=application.build_app()
    labels={getattr(block,'label',None) for block in demo.blocks.values()}
    assert {'预设配置','处理状态','色度修正比例','亮度 / 背景修正比例','分割模型'} <= labels
    def layout_ids(node):
        return [node['id']] + [item for child in node.get('children', []) for item in layout_ids(child)]
    ids = layout_ids(demo.config['layout'])
    assert len(ids) == len(set(ids)), 'Tabs must not render nested applications twice'
