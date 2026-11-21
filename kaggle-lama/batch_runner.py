"""Finite Kaggle GPU job. This file is also the shared input/output contract."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
from importlib.metadata import version
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time
import warnings
import zipfile

from PIL import Image

SCHEMA = 1
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 30_000_000
OUTPUT_LIMIT = 18_000_000_000  # Leave room under Kaggle's documented 20 GB output limit.
PARAMETERS = {'mask_dilate_size', 'mask_box_expand_ratio', 'extra_boxes'}
WORKER_REQUIREMENTS = ('ultralytics==8.4.181', 'shapely>=2,<3', 'pyclipper>=1.3,<2',
    'networkx>=3,<4', 'einops>=0.8,<1', 'safetensors>=0.5,<1', 'kornia==0.8.2',
    'omegaconf>=2.3,<3', 'litelama==0.1.7')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def safe_name(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name or
            name.startswith('/') or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError('Expected a safe relative POSIX path')
    return name


def validate_options(options, size=None):
    if not isinstance(options, dict) or set(options)-PARAMETERS:
        raise ValueError('Unknown repair option')
    dilate = options.get('mask_dilate_size', 10)
    expand = options.get('mask_box_expand_ratio', 20.0)
    boxes = options.get('extra_boxes', [])
    if type(dilate) is not int or not 0 <= dilate <= 64:
        raise ValueError('mask_dilate_size must be an integer in 0..64')
    if (type(expand) not in (int, float) or not math.isfinite(expand) or not 0 <= expand <= 100):
        raise ValueError('mask_box_expand_ratio must be finite in 0..100')
    if not isinstance(boxes, list) or len(boxes) > 64:
        raise ValueError('extra_boxes must be an array of at most 64 rectangles')
    clipped = []
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError('Each extra box must contain exactly four integers')
        x1, y1, x2, y2 = box
        if size:
            w, h = size
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Extra box has no area after clipping')
        clipped.append((x1, y1, x2, y2))
    return dict(mask_dilate_size=dilate, mask_box_expand_ratio=float(expand), extra_boxes=tuple(clipped))


def decode_image(data):
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Each image must contain 1 byte..20 MiB')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ('PNG', 'JPEG') or getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Only non-animated PNG/JPEG images are accepted')
            if image.width * image.height > MAX_PIXELS:
                raise ValueError('Image exceeds 30 megapixels')
            image.load()
            return image.convert('RGB')


def validate_request(request):
    if (not isinstance(request, dict) or request.get('schema_version') != SCHEMA or
            set(request)-{'schema_version','job_id','defaults','pages'}):
        raise ValueError('Invalid request schema')
    if not re.fullmatch(r'lama-[a-z0-9-]{8,55}', str(request.get('job_id', ''))):
        raise ValueError('Invalid job_id')
    defaults = request.get('defaults', {})
    validate_options(defaults)
    pages = request.get('pages')
    if not isinstance(pages, list) or not pages or len(pages) > 5000:
        raise ValueError('A task needs 1..5000 pages; split larger inputs into tasks')
    names = set()
    for page in pages:
        if not isinstance(page, dict) or set(page)-{'id','file','source_name','sha256','options'}:
            raise ValueError('Invalid page entry')
        if not re.fullmatch(r'p[0-9]{6}', str(page.get('id', ''))) or page['id'] in names:
            raise ValueError('Page IDs must be unique p000001-style values')
        names.add(page['id'])
        safe_name(page['file'])
        if not isinstance(page.get('source_name'), str) or not page['source_name']:
            raise ValueError('source_name is required')
        if not re.fullmatch('[0-9a-f]{64}', str(page.get('sha256', ''))):
            raise ValueError('Page SHA256 is required')
        if not isinstance(page.get('options', {}), dict):
            raise ValueError('Page options must be an object')
        validate_options({**defaults, **page.get('options', {})})
    return request


def read_task(archive, expected_job=None, expected_request_sha=None):
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate archive members')
    for name in names:
        safe_name(name)
    info = archive.getinfo('request.json')
    if info.file_size > 8 * 1024 * 1024:
        raise ValueError('Request manifest exceeds 8 MiB')
    raw = archive.read(info)
    if expected_request_sha and digest(raw) != expected_request_sha:
        raise ValueError('Request manifest checksum mismatch')
    request = validate_request(json.loads(raw))
    if expected_job and request['job_id'] != expected_job:
        raise ValueError('Task identity mismatch')
    expected_files = {'request.json'} | {page['file'] for page in request['pages']}
    if set(names) != expected_files:
        raise ValueError('Unexpected or missing input files')
    for page in request['pages']:
        if archive.getinfo(page['file']).file_size > MAX_IMAGE_BYTES:
            raise ValueError('Image exceeds 20 MiB')
    return request


def run_batch(archive, request, engine, output_dir, provenance=None, output_limit=OUTPUT_LIMIT):
    """Write PNG files and JSON records into one result ZIP; keep failures explicit."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(schema_version=SCHEMA, job_id=request['job_id'], status='running',
                  started_at=datetime.now(timezone.utc).isoformat(), pages=[], provenance=provenance or {})
    zip_path = output_dir/'results.zip'
    if zip_path.exists():
        raise FileExistsError('Refusing to overwrite an existing result archive')
    total_bytes = 0
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as output:
        for page in request['pages']:
            record = dict(id=page['id'], source_name=page['source_name'], input_sha256=page['sha256'])
            image = None
            try:
                data = archive.read(page['file'])
                if digest(data) != page['sha256']:
                    raise ValueError('Image checksum mismatch')
                image = decode_image(data)
                options = validate_options({**request.get('defaults', {}), **page.get('options', {})}, image.size)
                start = time.perf_counter()
                response = engine.process(image, **options)
                record.update(width=response['width'], height=response['height'], boxes=response['boxes'],
                              seconds=round(time.perf_counter()-start, 3), options=options, status='ok', files={})
                buffers = []
                for key, suffix in [('inpainted_image_b64','cleaned.png'), ('mask_b64','mask.png'),
                                    ('precise_mask_b64','precise-mask.png')]:
                    name = page['id']+'/'+suffix
                    content = base64.b64decode(response[key], validate=True)
                    buffers.append((name, content))
                    record['files'][key] = dict(path=name, sha256=digest(content), bytes=len(content))
                if total_bytes + sum(len(data) for _, data in buffers) > output_limit:
                    raise ValueError('Task output limit reached; split this batch into smaller tasks')
                for name, content in buffers:
                    output.writestr(name, content)
                    total_bytes += len(content)
            except Exception as error:
                logging.exception('Page %s failed', page['id'])
                record = dict(id=page['id'], source_name=page['source_name'], input_sha256=page['sha256'],
                              status='error', error=type(error).__name__+': '+str(error)[:2000])
            finally:
                if image is not None:
                    image.close()
            output.writestr(page['id']+'/result.json', json_bytes(record))
            result['pages'].append(record)
            print('PAGE', page['id'], record['status'], flush=True)
        ok = sum(page['status'] == 'ok' for page in result['pages'])
        result.update(status='completed' if ok == len(result['pages']) else 'partial' if ok else 'failed',
                      succeeded=ok, failed=len(result['pages'])-ok,
                      finished_at=datetime.now(timezone.utc).isoformat())
        output.writestr('job-result.json', json_bytes(result))
    result['archive'] = dict(path='results.zip', sha256=file_digest(zip_path), bytes=zip_path.stat().st_size)
    (output_dir/'job-result.json').write_bytes(json_bytes(result))
    return result


def unpack_runtime(source, destination):
    destination = Path(destination)
    with zipfile.ZipFile(source) as archive:
        if sum(info.file_size for info in archive.infolist()) > 3_000_000_000:
            raise ValueError('Unexpectedly large runtime')
        names = set()
        for info in archive.infolist():
            name = safe_name(info.filename)
            if name in names or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Duplicate or symlink runtime member')
            names.add(name)
        archive.extractall(destination)
    manifest = json.loads((destination/'runtime-manifest.json').read_text(encoding='utf-8'))
    if file_digest(destination/'lama_engine.py') != manifest['engine_sha256']:
        raise ValueError('Engine checksum mismatch')
    for name, meta in manifest['models'].items():
        if file_digest(destination/'Saber/models'/safe_name(name)) != meta['sha256']:
            raise ValueError('Model checksum mismatch: '+name)
    return manifest


def run_kaggle(config):
    """Notebook entry point: install missing inference packages, run, then exit."""
    os.environ.update(YOLO_AUTOINSTALL='False', YOLO_OFFLINE='True', YOLO_CONFIG_DIR='/tmp/lama-ultralytics',
                      ULTRALYTICS_CONFIG_DIR='/tmp/lama-ultralytics', MPLCONFIGDIR='/tmp/lama-matplotlib')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    output = Path('/kaggle/working')
    engine = None
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU is unavailable; this job does not fall back to CPU')
        print('GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__, flush=True)
        inputs = Path('/kaggle/input')
        runtime = next((p for p in inputs.rglob('runtime.bin')
                        if file_digest(p) == config['runtime_sha256']), None)
        task = next((p for p in inputs.rglob('task.bin')
                     if file_digest(p) == config['task_sha256']), None)
        if runtime is None or task is None:
            raise FileNotFoundError('Attached private runtime/task dataset is missing or mismatched')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                        '--no-cache-dir', *WORKER_REQUIREMENTS], check=True)
        with tempfile.TemporaryDirectory(prefix='lama-runtime-', dir='/tmp') as work:
            manifest = unpack_runtime(runtime, work)
            sys.path.insert(0, work)
            from lama_engine import SaberEngine
            engine = SaberEngine(Path(work)/'Saber')
            engine.load()
            provenance = dict(saber_commit=manifest['saber_commit'], engine_sha256=manifest['engine_sha256'],
                runtime_sha256=config['runtime_sha256'], torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), python=sys.version,
                packages={name: version(name) for name in ('ultralytics', 'numpy', 'opencv-python',
                    'Pillow', 'kornia', 'litelama', 'shapely', 'pyclipper', 'omegaconf', 'torchvision')})
            with zipfile.ZipFile(task) as archive:
                request = read_task(archive, config['job_id'], config['request_sha256'])
                result = run_batch(archive, request, engine, output, provenance)
            print('JOB_RESULT', result['status'], 'pages:', len(result['pages']), flush=True)
    except Exception as error:
        logging.exception('Batch initialization failed')
        output.mkdir(parents=True, exist_ok=True)
        (output/'job-result.json').write_bytes(json_bytes(dict(schema_version=SCHEMA,
            job_id=config['job_id'], status='failed', error=type(error).__name__+': '+str(error)[:4000])))
    finally:
        if engine is not None:
            engine.close()


if __name__ == '__main__':
    if 'KAGGLE_JOB' not in globals():
        raise SystemExit('Use client.py submit to generate a private Kaggle job')
    run_kaggle(KAGGLE_JOB)
