import base64
from io import BytesIO
import json
from pathlib import Path
from PIL import Image
import numpy as np
import cv2


def image_uri(a):
    stream = BytesIO()
    Image.fromarray(a).save(stream, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()


def example(name, before, after, result, truth=None, metrics=None):
    item = dict(name=name, before=image_uri(before), after=image_uri(after),
                result=result.summary(), metrics=metrics)
    if result.aligned is not None:
        overlay = result.aligned.copy()
        overlay[~result.valid] = (30, 36, 48)
        overlay[result.mask] = (.35*overlay[result.mask]+.65*np.array([255, 92, 96])).astype(np.uint8)
        for r in result.regions:
            cv2.rectangle(overlay, (r['x'], r['y']), (r['x']+r['width'], r['y']+r['height']), (255, 190, 100), 1)
        item.update(aligned=image_uri(result.aligned), overlay=image_uri(overlay))
    if truth is not None:
        gt = before.copy()
        gt[truth] = (.3*gt[truth]+.7*np.array([85, 240, 190])).astype(np.uint8)
        item['truth'] = image_uri(gt)
    return item


def write_report(output, payload):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Prevent file names or metadata from terminating the JSON script element.
    serialized = json.dumps(payload, allow_nan=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('template.html').read_text(encoding='utf-8')
    (output/'report.html').write_text(template.replace('__PAYLOAD__', serialized), encoding='utf-8')
    (output/'metrics.json').write_text(json.dumps({k:v for k,v in payload.items() if k != 'examples'}, indent=2, allow_nan=False), encoding='utf-8')
