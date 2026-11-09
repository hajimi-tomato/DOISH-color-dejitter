from types import SimpleNamespace
import numpy as np
import pytest
from color_dejitter import segment


class FakeSession:
    def get_inputs(self):
        return [SimpleNamespace(name='image')]
    def run(self, _, inputs):
        image = inputs['image']
        assert image.dtype == np.float32 and image.flags.c_contiguous
        return [np.ones((1,1,1024,1024),np.float32)]


@pytest.mark.parametrize('shape',[(1,3000,3),(3000,1,3),(91,53,3)])
def test_extreme_aspect_ratio_and_output_size(shape):
    mask=segment.get_mask(np.zeros(shape,np.uint8),FakeSession())
    assert mask.shape == shape[:2]
    assert np.all(mask==255)


def test_missing_model_is_explicit(tmp_path):
    with pytest.raises(segment.SegmentationUnavailable,match='未找到'):
        segment._get_session(tmp_path/'missing.onnx')


def test_mask_cache_reuses_and_copies(monkeypatch):
    segment._masks.clear()
    calls=[]
    monkeypatch.setattr(segment,'_get_session',lambda _: FakeSession())
    def predict(image,session):
        calls.append(1)
        return np.ones(image.shape[:2],np.uint8)*123
    monkeypatch.setattr(segment,'get_mask',predict)
    image=np.zeros((20,20,3),np.uint8)
    first=segment.foreground_mask(image);first[:]=0
    second=segment.foreground_mask(image)
    assert len(calls)==1 and np.all(second==123)
