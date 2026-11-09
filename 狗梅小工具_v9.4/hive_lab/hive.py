"""Official, project-key V2 API adapter. No browser/private website endpoints."""
import math
from pathlib import Path

import httpx

ENDPOINT = 'https://api.thehive.ai/api/v2/task/sync'
DOCS = 'https://docs.thehive.ai/reference/ai-generated-image-and-video-detection-1'


def parse_response(payload):
    matches = []
    for item in payload.get('status', []):
        if str(item.get('status', {}).get('code')) != '0':
            raise ValueError('Hive 返回任务失败；没有将失败视作低分')
        response = item.get('response', {})
        for frame in response.get('output', []):
            classes = frame.get('classes', [])
            scores = [c.get('score') for c in classes if c.get('class') == 'ai_generated']
            if scores:
                if len(scores) != 1 or isinstance(scores[0], bool):
                    raise ValueError('Hive 的 ai_generated 字段不唯一或类型错误')
                score = float(scores[0])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Hive 返回非法分数')
                info = response.get('input', {})
                matches.append({'percent': score * 100, 'model': info.get('model', 'unknown'),
                                'model_version': info.get('model_version', 'unknown'),
                                'classes': classes})
    if len(matches) != 1:
        raise ValueError('未找到唯一的 ai_generated 图像分数；请使用 AI 图像检测项目的 V2 密钥')
    return matches[0]


def detect(path, key, client=None):
    key = key.strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError('请填写有效的 Hive AI 图像检测项目密钥')
    def request(connection):
        with Path(path).open('rb') as media:
            response = connection.post(ENDPOINT, headers={'Authorization': f'Token {key}'},
                                       files={'media': (Path(path).name, media)}, timeout=60,
                                       follow_redirects=False)
        if response.status_code != 200:
            # Never echo request headers / response body (may contain credentials).
            raise ValueError(f'Hive HTTP {response.status_code}；检测未完成，不自动重试计费请求')
        try:
            return parse_response(response.json())
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError('Hive 响应格式不兼容') from exc
    try:
        if client is not None:
            return request(client)
        with httpx.Client() as connection:
            return request(connection)
    except httpx.RequestError as exc:
        raise ValueError('Hive 网络请求失败或超时；已停止，已完成的分数会保留') from exc
