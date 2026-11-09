"""Real upload/process/download smoke test; optionally start the packaged exe.

python tests/smoke_app.py --url http://127.0.0.1:7860
python tests/smoke_app.py --exe 狗梅小工具_v9.4/dist/DOISH/DOISH.exe
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
from PIL import Image, PngImagePlugin
from gradio_client import Client, handle_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'狗梅小工具_v9.4'))
from color_dejitter.core import Settings


def exercise(url, folder, use_model=False):
    image=np.random.default_rng(17).integers(90,180,(96,128,4),dtype=np.uint8)
    image[...,3]=255
    image[:10,:,3]=0
    path=folder/'中文测试图片.png'
    info=PngImagePlugin.PngInfo();info.add_itxt('parameters','DOISH API smoke test')
    Image.fromarray(image).save(path,pnginfo=info)
    client=Client(url,verbose=False,download_files=str(folder/'downloads'))
    loaded=client.predict(handle_file(str(path)),api_name='/load_image')
    assert 'original' in json.loads(loaded[0])
    zero=client.predict(*asdict(Settings(overall_strength=0)).values(),api_name='/process')
    with Image.open(zero[2][0]) as png:
        np.testing.assert_array_equal(np.asarray(png),image)
        assert png.text['parameters']=='DOISH API smoke test'
    with Image.open(zero[1]) as png:
        assert not np.asarray(png).any()
    processed=client.predict(*asdict(Settings(s_enabled=True,v_enabled=True,background_enabled=use_model)).values(),api_name='/process')
    with Image.open(processed[2][0]) as png:
        out=np.asarray(png)
        assert out.shape==image.shape
        np.testing.assert_array_equal(out[...,3],image[...,3])
        assert np.any(out[:,:,:3]!=image[:,:,:3])
    assert all(Path(p).is_file() for p in processed[2])
    if use_model:
        assert '已跳过' not in processed[3], processed[3]
    print('PASS: upload, zero identity, H/S/V processing, alpha, PNG metadata and downloads',flush=True)
    print(processed[3],flush=True)
    # Exercise the new primary workflow through the actual Gradio API too.
    generated = client.predict([handle_file(str(path))], ['texture', 'balanced'], 2026, False,
                               32, 0.95, 1, api_name='/lab_generate')
    experiment_id = generated[0]['value']
    bundle = Path(generated[2][0])
    import csv
    import io
    import zipfile
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert len(manifest['rows']) == 4
        entries = list(csv.DictReader(io.StringIO(archive.read('scores.csv').decode('utf-8-sig'))))
    scores = folder / 'scores.csv'
    with scores.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        for entry in entries:
            entry['hive_ai_percent'] = '99' if entry['id'].endswith(('uploaded', 'control')) else '20'
            writer.writerow(entry)
    # Synthetic scores verify plumbing only; these are not measurements from Hive.
    imported = client.predict(experiment_id, handle_file(str(scores)), 'synthetic smoke test',
                              20, 0.8, 1, api_name='/lab_import_scores')
    assert '1 张原图' in imported[2]
    preview = client.predict(experiment_id, '01_texture', api_name='/lab_compare')
    assert 'result' in json.loads(preview)
    print('PASS: lab generation, ZIP/CSV export, synthetic score import, ranking and comparison', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    target=parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--url')
    target.add_argument('--exe',type=Path)
    target.add_argument('--source',type=Path,help='Start a Python launcher with the active interpreter')
    parser.add_argument('--model',type=Path,help='Optional real model smoke test')
    args=parser.parse_args()
    process=None
    # Keep test files inside the workspace when running from CI or the repo.
    with tempfile.TemporaryDirectory(prefix='doish-smoke-',dir=Path.cwd()) as name:
        folder=Path(name)
        with open(folder/'server.log','w',encoding='utf-8') as log:
            try:
                if args.exe or args.source:
                    with socket.socket() as sock:
                        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
                    url=f'http://127.0.0.1:{port}'
                    env=dict(os.environ,PYTHONUTF8='1',GRADIO_ANALYTICS_ENABLED='False',DOISH_DATA_DIR=str(folder/'data'))
                    if args.model: env['DOISH_MODEL_PATH']=str(args.model.resolve())
                    command = [str(args.exe.resolve())] if args.exe else [sys.executable, str(args.source.resolve())]
                    process=subprocess.Popen([*command,'--no-browser','--port',str(port)],stdout=log,stderr=log,env=env,
                                             creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    deadline=time.monotonic()+60
                    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError('Packaged server exited before startup')
                        try:
                            with opener.open(url+'/config',timeout=1) as response:
                                assert response.status==200
                            break
                        except (OSError,AssertionError):
                            if time.monotonic()>deadline: raise TimeoutError('Packaged server did not start in 60 seconds')
                            time.sleep(0.5)
                else:
                    url=args.url
                exercise(url,folder,use_model=bool(args.model))
            except Exception:
                log.flush()
                print((folder/'server.log').read_text(encoding='utf-8',errors='replace'),file=sys.stderr)
                raise
            finally:
                if process and process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait(timeout=5)


if __name__=='__main__':
    main()
