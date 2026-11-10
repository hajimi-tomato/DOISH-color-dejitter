"""Reproduce the published ONNX from a hash-verified original DRUNet PTH.

Export-only dependencies: torch, spandrel, onnx, onnxruntime, numpy.
Tested exporter: torch 2.7.0, legacy ONNX export, opset 17.
Different exporter versions can produce different file hashes; compare numeric
outputs and do not replace the application checksum without a reviewed release.
"""
import argparse
import hashlib
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    expected='479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4'
    if hashlib.sha256(args.weights.read_bytes()).hexdigest()!=expected:
        raise ValueError('Original weight SHA-256 mismatch')
    import numpy as np
    import torch
    import spandrel
    import onnx
    import onnxruntime as ort
    model=spandrel.ModelLoader().load_from_file(str(args.weights)).model.eval().cpu()
    x=torch.rand(1,4,64,80)
    torch.onnx.export(model,x,str(args.output),input_names=['input'],output_names=['output'],
                      dynamic_axes={'input':{2:'height',3:'width'},'output':{2:'height',3:'width'}},
                      opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(str(args.output)))
    session=ort.InferenceSession(str(args.output),providers=['CPUExecutionProvider'])
    with torch.no_grad():
        error=float(np.max(np.abs(model(x).numpy()-session.run(None,{'input':x.numpy()})[0])))
    if error>=1e-4:
        raise RuntimeError(f'Numeric parity failed: {error}')
    print(f'Max error: {error}; SHA-256: {hashlib.sha256(args.output.read_bytes()).hexdigest()}')


if __name__=='__main__':
    main()
