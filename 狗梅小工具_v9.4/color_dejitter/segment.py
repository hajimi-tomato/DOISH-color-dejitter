#!/usr/bin/env python3
"""
anime-segmentation 前景/背景分割脚本
使用系统 Python 3.9 + onnxruntime 运行

用法: /usr/bin/python3 segment.py <input_image_path> <output_mask_path>
输出: 灰度 mask 图片，255=前景（人物），0=背景

预处理按照官方 inference.py：等比缩放 + 黑色 padding 居中填充到 1024x1024
后处理：裁掉 padding + resize 回原尺寸
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image


def get_mask(input_img: np.ndarray, onnx_session, s: int = 1024) -> np.ndarray:
    """
    运行 anime-seg 模型，返回前景 mask (0~255 uint8)。
    严格按照官方 inference.py 的预处理/后处理流程。
    """
    # 归一化到 [0, 1]
    input_img = (input_img / 255.0).astype(np.float32)
    h0, w0 = input_img.shape[:2]

    # 等比缩放：长边缩到 s，短边按比例
    if h0 > w0:
        h, w = s, int(s * w0 / h0)
    else:
        h, w = int(s * h0 / w0), s

    # padding 量
    ph, pw = s - h, s - w

    # 创建黑色画布，居中放置缩放后的图像
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_resized = cv2.resize(input_img, (w, h))
    img_input[ph // 2: ph // 2 + h, pw // 2: pw // 2 + w] = img_resized

    # HWC -> CHW -> NCHW
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]

    # 推理
    input_name = onnx_session.get_inputs()[0].name
    output = onnx_session.run(None, {input_name: img_input})

    # 后处理：和官方一致
    pred = output[0][0]  # shape: (1, s, s) or (s, s)
    if pred.ndim == 3:
        pred = np.transpose(pred, (1, 2, 0))  # CHW -> HWC
    else:
        pred = pred[:, :, np.newaxis]  # (s,s) -> (s,s,1)

    # 裁掉 padding，恢复有效区域
    pred = pred[ph // 2: ph // 2 + h, pw // 2: pw // 2 + w]

    # resize 回原尺寸
    pred = cv2.resize(pred, (w0, h0))

    # 确保是 2D
    if pred.ndim == 3:
        pred = pred[:, :, 0]

    # clamp 到 [0, 1]
    pred = np.clip(pred, 0, 1)

    # 转为 uint8
    return (pred * 255).astype(np.uint8)


def main():
    if len(sys.argv) != 3:
        print("Usage: /usr/bin/python3 segment.py <input_image> <output_mask>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # 加载模型
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    model_path = os.path.join(model_dir, "isnetis.onnx")

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    import onnxruntime as ort
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # 加载图片 (BGR -> RGB)
    img = cv2.imread(input_path)
    if img is None:
        print(f"ERROR: Cannot read image {input_path}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 获取 mask
    mask = get_mask(img_rgb, session)

    # 保存 mask
    cv2.imwrite(output_path, mask)
    print("OK")


if __name__ == "__main__":
    main()
