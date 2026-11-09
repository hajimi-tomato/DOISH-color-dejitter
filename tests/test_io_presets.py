from dataclasses import replace
import numpy as np
from PIL import Image, ImageCms, PngImagePlugin
import pytest

from color_dejitter.core import Settings
from color_dejitter.images import load_image, save_png
from color_dejitter.presets import PresetStore


def test_rgba_metadata_icc_round_trip(tmp_path):
    pixels = np.zeros((32,32,4), np.uint8)
    pixels[:] = [70,130,180,255]
    pixels[:8,:,3] = 0
    info = PngImagePlugin.PngInfo()
    text = '测试 AI 参数 ' * 160000  # exceeds Pillow's default 1 MiB decompression cap
    info.add_itxt('parameters', text, zip=True)
    path = tmp_path/'中文输入.png'
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.fromarray(pixels).save(path,pnginfo=info,icc_profile=profile)
    source = load_image(path)
    saved = tmp_path/'result.png'
    save_png(saved, source.bgr, source)
    with Image.open(saved) as image:
        image.load()
        np.testing.assert_array_equal(np.array(image)[...,3], pixels[...,3])
        assert image.text['parameters'] == text
        assert image.info['icc_profile']


def test_exif_rotation_applied(tmp_path):
    image = Image.new('RGB',(40,20),'red')
    exif = Image.Exif(); exif[274]=6
    path=tmp_path/'rotated.jpg';image.save(path,exif=exif)
    assert load_image(path).bgr.shape[:2] == (40,20)


def test_presets_persist_and_builtins_protected(tmp_path):
    store = PresetStore(tmp_path/'presets.json')
    settings = replace(Settings(),h_denoise_strength=0.3,v_enabled=True)
    store.save('我的预设',settings)
    assert PresetStore(store.path).all()['我的预设'] == settings
    for name in ['默认',' 默认 ']:
        with pytest.raises(ValueError): store.save(name,settings)
    with pytest.raises(ValueError): store.delete('默认')
    store.delete('我的预设')
    assert '我的预设' not in store.all()


def test_corrupt_preset_file_is_not_overwritten(tmp_path):
    path=tmp_path/'presets.json';path.write_text('{bad',encoding='utf-8')
    with pytest.raises(ValueError): PresetStore(path).save('test',Settings())
    assert path.read_text('utf-8') == '{bad'


def test_high_bit_depth_rejected(tmp_path):
    path=tmp_path/'16bit.png'
    Image.fromarray(np.full((10,10),1000,np.uint16)).save(path)
    with pytest.raises(ValueError,match='8-bit'): load_image(path)
