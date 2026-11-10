"""NumPy/ONNX adaptation of keepcolor-denoise's tiled DRUNet-PD and L0 stages.

Copyright (c) 2026 juanhanyu. MIT; see third_party/keepcolor-LICENSE.txt.
Source: juanhanyu/keepcolor-denoise, commit 63ecb40b014f6073eafc7072a4cc3c49714452a0.
Changes: NCHW NumPy backend, configurable tiles/refinement, sigma-zero identity,
and reflected borders for L0 to reduce the original FFT wraparound interaction.
"""
import numpy as np


def feather(n, overlap, lead, tail):
    weight = np.ones(n, dtype=np.float32)
    k = min(overlap, n)
    if lead:
        weight[:k] = np.linspace(0, 1, k + 2, dtype=np.float32)[1:-1]
    if tail:
        weight[-k:] = np.minimum(weight[-k:], np.linspace(1, 0, k + 2, dtype=np.float32)[1:-1])
    return weight


def tiled(fn, image, tile=256, overlap=64, report=None):
    if tile <= overlap or overlap < 0:
        raise ValueError('分块大小必须大于重叠范围')
    _, _, h, w = image.shape
    if h <= tile and w <= tile:
        return fn(image)
    ys = sorted({*range(0, max(h - tile, 0) + 1, tile-overlap), max(h-tile, 0)})
    xs = sorted({*range(0, max(w - tile, 0) + 1, tile-overlap), max(w-tile, 0)})
    output = np.zeros_like(image)
    weight = np.zeros((1, 1, h, w), dtype=np.float32)
    for index, (y, x) in enumerate((y, x) for y in ys for x in xs):
        if report:
            report(f'分块 {index + 1}/{len(ys)*len(xs)}')
        y1, x1 = min(y+tile, h), min(x+tile, w)
        mask = feather(y1-y, overlap, y>0, y1<h)[:, None] * feather(x1-x, overlap, x>0, x1<w)[None, :]
        output[..., y:y1, x:x1] += fn(image[..., y:y1, x:x1]) * mask
        weight[..., y:y1, x:x1] += mask
    return output / weight


def drunet(predict, image, sigma):
    if sigma == 0:
        return image.copy()
    h, w = image.shape[-2:]
    padded = np.pad(image, ((0,0),(0,0),(0,-h%8),(0,-w%8)), mode='edge')
    noise = np.full_like(padded[:, :1], sigma/255)
    return predict(np.concatenate((padded, noise), axis=1))[..., :h, :w]


def denoise(predict, image, sigma, mode='pd2', refine=4, tile=256, report=None):
    if sigma == 0:
        return image.copy()
    run = lambda value, level: tiled(lambda part: drunet(predict, part, level), value, tile, report=report)
    if mode == 'standard':
        return np.clip(run(image, sigma), 0, 1)
    if mode != 'pd2':
        raise ValueError('未知降噪模式')
    _, channels, h, w = image.shape
    padded = np.pad(image, ((0,0),(0,0),(0,h%2),(0,w%2)), mode='edge')
    result = np.empty_like(padded)
    for y in range(2):
        for x in range(2):
            if report:
                report(f'像素重排子图 {y*2+x+1}/4')
            result[..., y::2, x::2] = run(padded[..., y::2, x::2], sigma)
    result = result[..., :h, :w]
    if refine > 0:
        if report:
            report('全尺寸轻度修整')
        result = run(result, refine)
    return np.clip(result, 0, 1)


def l0_smooth(image, lam=0.0003):
    """L0 gradient minimisation port of keepcolor; RGB float32 HWC, 0..1."""
    if lam == 0:
        return image.copy()
    h, w = image.shape[:2]
    pad = min(16, h, w)
    # Reflection moves the periodic FFT boundary away from the actual image edge.
    source = np.pad(image, ((pad,pad),(pad,pad),(0,0)), mode='reflect').astype(np.float64)
    height, width = source.shape[:2]
    fy, fx = np.fft.fftfreq(height)[:, None], np.fft.fftfreq(width)[None, :]
    lap = ((2-2*np.cos(2*np.pi*fx)) + (2-2*np.cos(2*np.pi*fy)))[..., None]
    frequency = np.fft.fft2(source, axes=(0,1))
    result, beta = source.copy(), 2*lam
    while beta < 1e5:
        gx = np.roll(result, -1, axis=1)-result
        gy = np.roll(result, -1, axis=0)-result
        flat = np.sum(gx*gx+gy*gy, axis=2) <= lam/beta
        gx[flat], gy[flat] = 0, 0
        divergence = np.roll(gx, 1, axis=1)-gx + np.roll(gy, 1, axis=0)-gy
        result = np.fft.ifft2((frequency+beta*np.fft.fft2(divergence, axes=(0,1))) / (1+beta*lap), axes=(0,1)).real
        beta *= 2
    return np.clip(result[pad:pad+h, pad:pad+w], 0, 1).astype(np.float32)
