# Third-party attribution

`restoration/keepcolor_numpy.py` adapts the DRUNet pixel-rearrangement, overlapping
tile blending and L0 gradient-minimization implementation from
[juanhanyu/keepcolor-denoise](https://github.com/juanhanyu/keepcolor-denoise),
commit `63ecb40b014f6073eafc7072a4cc3c49714452a0`.
Copyright (c) 2026 juanhanyu; MIT license in `keepcolor-LICENSE.txt`.
This adaptation uses NumPy and ONNX Runtime, configurable refinement/tile sizes,
exact disabled-stage bypasses and reflected L0 borders. The primary application,
edge/chroma/protection blending, model packaging and measurement workflows are
separate changes. No APISR implementation or weights are included.

DRUNet was introduced by Kai Zhang et al. in *Plug-and-Play Image Restoration with
Deep Denoiser Prior*, IEEE TPAMI 2021. Original implementation and pretrained
model: [cszn/DPIR](https://github.com/cszn/DPIR), MIT (`DPIR-LICENSE.txt`).
The model distributed as `drunet_color.onnx` is an ONNX conversion of
`https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth`, not a
newly trained model. The state dict was loaded using spandrel; spandrel and
PyTorch are export-time dependencies and are not needed for inference.

- Original PTH SHA-256: `479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4`
- ONNX SHA-256: `39e42853791f9255551cb1dc534c8e778dcbd9316bd07fff8097ab38e0408d70`
- Opset: 17; float32; input `1×4×H×W` RGB plus sigma/255; output `1×3×H×W`.
- Export validation max absolute error against PyTorch on a 64×80 tensor: `6.86e-7`.

L0 algorithm reference: Li Xu, Cewu Lu, Yi Xu, Jiaya Jia, *Image Smoothing via
L0 Gradient Minimization*, ACM SIGGRAPH Asia 2011.

These licenses cover their respective third-party components, not an assignment
of the existing repository's overall license (which remains TBD).
