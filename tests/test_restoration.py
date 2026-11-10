from dataclasses import replace

import cv2
import numpy as np
import pytest

from color_dejitter.images import SourceImage
from restoration.core import Settings, fill_transparent, restore
from restoration.keepcolor_numpy import denoise, l0_smooth, tiled
from restoration.ui import FIELDS, mask_from_editor, process_image


def predictor(tensor):
    image = tensor[0,:3].transpose(1,2,0)
    return cv2.GaussianBlur(image,(3,3),0.8).transpose(2,0,1)[None]


@pytest.mark.parametrize('shape',[(1,1),(3,7),(129,131),(271,287)])
def test_pixel_rearrangement_identity_and_tiling(shape):
    value=np.random.default_rng(3).random((1,3,*shape),dtype=np.float32)
    result=denoise(lambda x:x[:,:3],value,12,'pd2',refine=3,tile=128)
    # Multiple float32 overlap-add passes can accumulate several ULPs.
    np.testing.assert_allclose(result,value,atol=1e-6)
    np.testing.assert_allclose(tiled(lambda x:x,value,128),value,atol=1e-6)


def test_bypass_never_loads_model():
    source=SourceImage(np.random.default_rng(2).integers(0,256,(24,25,3),dtype=np.uint8))
    def fail(*args):
        raise AssertionError('model should not run')
    for settings,mask in [(Settings(amount=0),None),(Settings(sigma=0,texture=0),None),(Settings(),np.ones((24,25),dtype=np.float32))]:
        np.testing.assert_array_equal(restore(source,settings,mask,predictor=fail),source.bgr)


def test_mask_and_alpha_exact_protection():
    source=SourceImage(np.random.default_rng(2).integers(0,256,(32,40,3),dtype=np.uint8),np.full((32,40),255,np.uint8))
    source.alpha[:4]=0
    mask=np.zeros((32,40),np.float32); mask[:,20:]=1
    output=restore(source,Settings(texture=0),mask,predictor=predictor)
    np.testing.assert_array_equal(output[:4],source.bgr[:4])
    np.testing.assert_array_equal(output[:,20:],source.bgr[:,20:])
    assert np.any(output[4:,:20]!=source.bgr[4:,:20])


def test_hidden_rgb_does_not_contaminate_visible_pixels():
    first=np.full((32,32,3),.5,np.float32)
    second=first.copy(); second[:8]=1
    alpha=np.full((32,32),255,np.uint8);alpha[:8]=0
    np.testing.assert_array_equal(fill_transparent(first,alpha),fill_transparent(second,alpha))


def test_l0_reduces_small_flat_variation_and_preserves_constant():
    constant=np.full((32,32,3),.4,np.float32)
    np.testing.assert_allclose(l0_smooth(constant),constant,atol=1e-6)
    noisy=constant+np.random.default_rng(8).normal(0,.005,constant.shape).astype(np.float32)
    smooth=l0_smooth(noisy)
    assert np.std(smooth)<np.std(noisy)/2


@pytest.mark.parametrize('settings',[Settings(sigma=float('nan')),Settings(amount=-1),Settings(mode='bad'),Settings(tile=0),Settings(lam=.5)])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):settings.validated()


def test_editor_uses_layers_not_background():
    background=np.full((16,16,4),255,np.uint8)
    assert mask_from_editor({'background':background,'layers':[]},(32,32)) is None
    layer=np.zeros((16,16,4),np.uint8);layer[:8,:,3]=255
    mask=mask_from_editor({'background':background,'layers':[layer]},(32,32))
    assert np.all(mask[:14]==1) and np.all(mask[18:]==0)


def test_real_callback_exports_metadata_and_zero_identity(tmp_path,monkeypatch):
    from PIL import Image,PngImagePlugin
    from dataclasses import asdict
    monkeypatch.setenv('DOISH_DATA_DIR',str(tmp_path/'data'))
    path=tmp_path/'source.png'
    info=PngImagePlugin.PngInfo();info.add_itxt('parameters','keep me')
    original=np.random.default_rng(4).integers(0,256,(24,31,4),dtype=np.uint8)
    Image.fromarray(original).save(path,pnginfo=info)
    settings=Settings(amount=0)
    _,files,_=process_image(path,None,*[getattr(settings,k) for k in FIELDS])
    with Image.open(files[0]) as image:
        np.testing.assert_array_equal(np.asarray(image),original)
        assert image.text['parameters']=='keep me'
