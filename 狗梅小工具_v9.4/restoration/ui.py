"""Primary image cleanup screen; the benchmark lab is a secondary workflow."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from color_dejitter.images import as_pil, load_image, preview_data, save_png
from color_dejitter.presets import data_dir
from color_dejitter.segment import app_dir
from hive_lab.pipeline import quality
from .core import PRESETS, Settings, restore
from .model import MODEL_SHA256, download, model_path

FIELDS = ['mode','sigma','refine','amount','texture','lam','edges','color','doish','device','tile']


def prepare(path):
    if not path:
        return None, '{}', None, '请上传原图。'
    source = load_image(path)
    image = as_pil(source.bgr, source.alpha)
    image.thumbnail((1600,1600), Image.Resampling.LANCZOS)
    return {'background':np.asarray(image.convert('RGBA')), 'layers':[], 'composite':np.asarray(image.convert('RGBA'))}, json.dumps({'original':preview_data(source.bgr,source.alpha)}), None, '已载入。可直接处理，或在保护画板涂白保留局部。'


def mask_from_editor(editor, shape):
    if not editor or not editor.get('layers'):
        return None
    combined = None
    for layer in editor['layers']:
        array = np.asarray(layer)
        if array.ndim != 3 or array.shape[2] != 4:
            raise ValueError('保护画板格式错误')
        alpha = array[...,3].astype(np.float32)/255
        alpha = cv2.resize(alpha, (shape[1],shape[0]), interpolation=cv2.INTER_LINEAR)
        combined = alpha if combined is None else np.maximum(combined, alpha)
    return combined


def process_image(path, editor, *values, report=None):
    if not path:
        raise ValueError('请先上传原图')
    settings = Settings(**dict(zip(FIELDS,values))).validated()
    source = load_image(path)
    protect = mask_from_editor(editor,source.bgr.shape[:2])
    started = time.perf_counter()
    notes=[]
    def notify(text):
        if text.startswith('推理设备') or '失败' in text:
            notes.append(text)
        if report:
            report(text)
    output = restore(source,settings,protect,notify)
    root = data_dir()/'restoration'
    root.mkdir(parents=True,exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='clean-',dir=root))
    target = folder/'clean.png'
    save_png(target,output,source)
    difference = np.clip(np.abs(output.astype(np.int16)-source.bgr.astype(np.int16))*5,0,255).astype(np.uint8)
    save_png(folder/'difference_x5.png',difference)
    if protect is not None:
        Image.fromarray(np.rint(protect*255).astype(np.uint8)).save(folder/'protection.png')
    metrics=quality(source.bgr,output,source.alpha)
    record={'settings':asdict(settings),'model_sha256':MODEL_SHA256,'fidelity':metrics,
            'seconds':round(time.perf_counter()-started,3),'notes':notes,'hive_score':None,
            'protection':'protection.png' if protect is not None else None}
    (folder/'parameters.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    view=json.dumps({'original':preview_data(source.bgr,source.alpha),'result':preview_data(output,source.alpha)})
    status=f'完成 · {output.shape[1]} × {output.shape[0]} · {record["seconds"]:.1f} 秒 · SSIM {metrics["ssim_1024"]:.4f}。\n'+'；'.join(notes)
    status+='\n请检查文字、毛发、皮肤与渐变。画质差异指标不代表 Hive 检测结果。'
    return view,[str(target),str(folder/'difference_x5.png'),str(folder/'parameters.json')]+([str(folder/'protection.png')] if protect is not None else []),status


def build_restoration():
    resources=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[1]))
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown('## 深度降噪与去纹理\nDRUNet 清理细颗粒和相关噪声，L0 压制色块中的细碎纹理。适用于生成图后期；具体模型出图仍需实图验证。')
        with gr.Row():
            with gr.Column(scale=4):
                upload=gr.File(label='原图 · 保留原尺寸，最多 1677 万像素',file_types=['image'],type='filepath')
                examples=sorted((app_dir()/'examples').glob('*.png'))[:4]
                if examples:
                    gr.Examples([[str(path)] for path in examples],inputs=[upload],label='本机示例图片')
                preset=gr.Dropdown([(label,key) for key,(label,_) in PRESETS.items()],value='texture',label='清理方案')
                controls={}
                controls['amount']=gr.Slider(0,1,value=.9,step=.01,label='整体处理强度 · 0 为原图')
                controls['sigma']=gr.Slider(0,50,value=12,step=.5,label='DRUNet 降噪强度')
                controls['texture']=gr.Slider(0,1,value=.15,step=.01,label='额外去纹理 · L0')
                controls['edges']=gr.Slider(0,1,value=.65,step=.01,label='边缘与细线保护')
                controls['color']=gr.Slider(0,1,value=.8,step=.01,label='低频色彩偏移修正')
                with gr.Accordion('高级算法参数',open=False):
                    controls['mode']=gr.Dropdown([('普通 DRUNet · 细颗粒','standard'),('2×2 像素重排 · 相关纹理','pd2')],value='pd2',label='降噪路径')
                    controls['refine']=gr.Slider(0,10,value=3,step=.5,label='重排后的全尺寸修整')
                    controls['lam']=gr.Slider(.00001,.01,value=.0003,step=.00001,label='L0 纹理阈值 · 越高越容易抹掉渐变')
                    controls['doish']=gr.Checkbox(False,label='附加原 DOISH 色彩去抖')
                    controls['device']=gr.Dropdown([('自动 · Windows 优先 DirectML','auto'),('CPU','cpu')],value='auto',label='计算设备')
                    controls['tile']=gr.Dropdown([128,256,512],value=256,label='推理分块 · 显存不足可用 128')
                with gr.Accordion('局部保护画板 · 涂白区域保留原图',open=False):
                    editor=gr.ImageEditor(label='保护文字、眼睛、毛发或需要保留的纹理',type='numpy',image_mode='RGBA',
                                          brush=gr.Brush(colors=['#ffffff'],color_mode='fixed'),sources=[],transforms=(),height=360)
                    gr.Markdown('画板最长边 1600 像素，保护蒙版映射回原尺寸。橡皮擦可移除保护。')
                run=gr.Button('开始深度清理',variant='primary')
                with gr.Accordion('降噪模型',open=False):
                    model_status=gr.Textbox(value='模型文件已存在，首次处理时校验。' if model_path().is_file() else '需要下载 DRUNet 模型，约 125 MiB。',label='DRUNet 模型状态',interactive=False)
                    get_model=gr.Button('下载 / 校验降噪模型')
                    gr.Markdown('核心流程移植自 MIT 许可的 [keepcolor-denoise](https://github.com/juanhanyu/keepcolor-denoise)，模型来自 [DRUNet / DPIR](https://github.com/cszn/DPIR)。')
            with gr.Column(scale=7):
                viewer=gr.HTML('{}',html_template=(resources/'web/compare.html').read_text('utf-8'),css_template=(resources/'web/compare.css').read_text('utf-8'),js_on_load=(resources/'web/compare.js').read_text('utf-8'))
                status=gr.Textbox(label='深度清理状态',value='请上传图片。首次使用需要安装降噪模型。',lines=3,interactive=False)
                downloads=gr.File(label='结果 / 差异图 / 参数 / 保护蒙版',file_count='multiple',interactive=False)
                gr.Markdown('Hive 候选沿用同一深度处理内核；请在「Hive 验证」标签生成对照并测分。此处没有内置或模拟的 Hive 分数。')
        ordered=[controls[k] for k in FIELDS]
        def load(path):
            try:
                return prepare(path)
            except (ValueError,OSError) as exc:
                return None,'{}',None,f'无法载入：{exc}'
        upload.change(load,upload,[editor,viewer,downloads,status],api_name='restoration_load')
        def apply(path,mask,*values,progress=gr.Progress()):
            try:
                return process_image(path,mask,*values,report=lambda text:progress(None,desc=text))
            except (ValueError,OSError,RuntimeError) as exc:
                raise gr.Error(str(exc)) from exc
        run.click(apply,[upload,editor,*ordered],[viewer,downloads,status],api_name='restoration_process',concurrency_limit=1,concurrency_id='deep-processing')
        preset.input(lambda key:[*[getattr(PRESETS[key][1],f) for f in FIELDS],None,'方案已载入，请重新处理。'],preset,[*ordered,downloads,status],api_name=False)
        for component in [*ordered,editor]:
            component.input(lambda:(None,'参数 / 蒙版已改变，请重新处理。'),outputs=[downloads,status],api_name=False,queue=False)
        def install(progress=gr.Progress()):
            try:
                download(lambda text:progress(None,desc=text))
                return 'DRUNet 模型已就绪，SHA-256 校验通过。'
            except Exception as exc:
                return f'下载失败：{exc}'
        get_model.click(install,outputs=model_status,api_name=False,concurrency_limit=1,concurrency_id='deep-processing')
    return demo
