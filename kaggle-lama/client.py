"""Submit private finite Kaggle LaMa jobs and verify their saved results."""
import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import secrets
import sys
import time
import zipfile

from batch_runner import (SCHEMA, OUTPUT_LIMIT, MAX_IMAGE_BYTES, decode_image, digest,
    file_digest, json_bytes, read_task, safe_name, validate_options, validate_request)

HERE = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_bytes(json_bytes(data))
    temp.replace(path)


def authenticate(key_file=None):
    # Read only on the caller's machine; no credentials enter a dataset or script.
    if key_file:
        raw = Path(key_file).read_text(encoding='utf-8-sig').strip()
        if raw.startswith('{'):
            key = json.loads(raw)
            os.environ.update(KAGGLE_USERNAME=key['username'], KAGGLE_KEY=key['key'])
        else:
            os.environ['KAGGLE_API_TOKEN'] = raw
        from kaggle import api
        if not getattr(api, '_authenticated', False):
            api.authenticate()
        return api
    # No key file: fall back to the Kaggle SDK's own configuration
    # (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars).
    # The SDK raises a clear error when no credentials are configured.
    from kaggle import api
    api.authenticate()
    return api


def input_pages(source):
    """Yield original image bytes; ignore non-image archive entries without extracting."""
    source = Path(source)
    if source.is_dir():
        files = sorted((p for p in source.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES),
                       key=lambda p: p.relative_to(source).as_posix())
        for path in files:
            if path.is_symlink() or not path.is_file():
                raise ValueError('Input images must be regular files')
            if path.stat().st_size > MAX_IMAGE_BYTES:
                raise ValueError('Image exceeds 20 MiB: '+path.name)
            yield safe_name(path.relative_to(source).as_posix()), path.read_bytes()
    elif source.suffix.lower() in {'.zip', '.cbz'}:
        with zipfile.ZipFile(source) as archive:
            seen = set()
            for info in sorted(archive.infolist(), key=lambda i: i.filename):
                if info.is_dir():
                    safe_name(info.filename.rstrip('/'))
                    continue
                name = safe_name(info.filename)
                if name in seen or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Duplicate or symlink archive member: '+name)
                seen.add(name)
                if Path(name).suffix.lower() in IMAGE_SUFFIXES:
                    if info.file_size > MAX_IMAGE_BYTES:
                        raise ValueError('Image exceeds 20 MiB: '+name)
                    yield name, archive.read(info)
    else:
        raise ValueError('Input must be an image folder or ZIP/CBZ archive')


def prepare(source, output, options=None, job_id=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    options = {} if options is None else options
    if not isinstance(options, dict) or set(options)-{'defaults', 'pages'}:
        raise ValueError('Options must contain defaults and/or pages')
    defaults, overrides = options.get('defaults', {}), options.get('pages', {})
    validate_options(defaults)
    if not isinstance(overrides, dict):
        raise ValueError('pages overrides must be an object keyed by source_name')
    for override in overrides.values():
        validate_options(override)
    job_id = job_id or 'lama-'+datetime.now(timezone.utc).strftime('%Y%m%d')+'-'+secrets.token_hex(6)
    request = dict(schema_version=SCHEMA, job_id=job_id, defaults=defaults, pages=[])
    estimated_bytes, seen = 0, set()
    with zipfile.ZipFile(output/'task.bin', 'w', compression=zipfile.ZIP_STORED) as archive:
        for index, (name, data) in enumerate(input_pages(source), 1):
            if index > 5000:
                raise ValueError('Split tasks into at most 5000 pages')
            with decode_image(data) as image:
                validate_options({**defaults, **overrides.get(name, {})}, image.size)
                # RGB clean image + two grayscale masks, with a conservative PNG allowance.
                estimated_bytes += image.width*image.height*6 + 131072
            if estimated_bytes > OUTPUT_LIMIT:
                raise ValueError('Estimated outputs exceed 18 GB; split the input')
            page_id = f'p{index:06d}'
            staged = 'pages/'+page_id+('.png' if data.startswith(b'\x89PNG') else '.jpg')
            entry = dict(id=page_id, file=staged, source_name=name, sha256=digest(data))
            if name in overrides:
                entry['options'] = overrides[name]
            request['pages'].append(entry)
            archive.writestr(staged, data)
            seen.add(name)
        if set(overrides)-seen:
            raise ValueError('Options refer to unknown source names: '+str(sorted(set(overrides)-seen)))
        validate_request(request)
        raw = json_bytes(request)
        archive.writestr('request.json', raw)
    save_json(output/'request.json', request)
    state = dict(schema_version=SCHEMA, job_id=job_id, phase='prepared',
                 pages=len(request['pages']), task_sha256=file_digest(output/'task.bin'),
                 request_sha256=digest(raw), estimated_output_bytes=estimated_bytes)
    save_json(output/'job.json', state)
    return state


def wait_dataset(api, ref, seconds=900):
    deadline = time.monotonic()+seconds
    previous = None
    while time.monotonic() < deadline:
        status = str(api.dataset_status(ref)).lower()
        if status != previous:
            print('DATASET', ref, status, flush=True)
            previous = status
        if status == 'ready':
            return
        if 'error' in status or 'fail' in status:
            raise RuntimeError('Dataset processing failed: '+status)
        time.sleep(15)
    raise TimeoutError('Dataset is still processing; retry submit with the same job.json')


def create_private_dataset(api, folder, ref):
    save_json(Path(folder)/'dataset-metadata.json', dict(id=ref, title=ref.split('/')[1],
        licenses=[dict(name='other')], description='Private processing input. Original authors retain all rights.'))
    response = api.dataset_create_new(str(folder), public=False, quiet=True,
                                     convert_to_csv=False, dir_mode='skip')
    if str(response.status).lower() != 'ok':
        raise RuntimeError('Private dataset creation failed: '+str(response.error))


def publish_runtime(api, bundle, output):
    bundle = Path(bundle).resolve()
    allowed = {'runtime.bin', 'runtime-manifest.json', 'dataset-metadata.json'}
    if {p.name for p in bundle.iterdir()}-allowed:
        raise ValueError('Runtime upload folder must contain only the bundle and its manifest')
    manifest = load_json(bundle/'runtime-manifest.json')
    if file_digest(bundle/'runtime.bin') != manifest['runtime_sha256']:
        raise ValueError('Runtime archive checksum mismatch')
    with zipfile.ZipFile(bundle/'runtime.bin') as archive:
        embedded = json.loads(archive.read('runtime-manifest.json'))
        if any(embedded.get(key) != manifest.get(key) for key in embedded):
            raise ValueError('Runtime manifests disagree')
    owner = api.config_values['username']
    ref = owner+'/saber-lama-runtime-'+manifest['runtime_sha256'][:16]
    print('PRIVATE RUNTIME', ref, flush=True)
    create_private_dataset(api, bundle, ref)
    wait_dataset(api, ref)
    deployment = dict(schema_version=SCHEMA, runtime_dataset=ref,
        runtime_sha256=manifest['runtime_sha256'], saber_commit=manifest['saber_commit'],
        engine_sha256=manifest['engine_sha256'], accelerator='NvidiaTeslaT4')
    save_json(output, deployment)
    return deployment


def make_script(deployment, state):
    config = {key: state[key] for key in ('job_id','task_sha256','request_sha256')}
    config['runtime_sha256'] = deployment['runtime_sha256']
    return 'KAGGLE_JOB = '+repr(config)+'\n'+(HERE/'batch_runner.py').read_text(encoding='utf-8')


def submit(api, job_path, deployment):
    job_path = Path(job_path).resolve()
    folder, state = job_path.parent, load_json(job_path)
    if state.get('phase') == 'submitted':
        raise ValueError('This job was already submitted; use status/wait/fetch')
    owner = api.config_values['username']
    if state.get('owner', owner) != owner:
        raise ValueError('This task belongs to a different Kaggle account')
    if file_digest(folder/'task.bin') != state['task_sha256']:
        raise ValueError('Task archive changed after prepare')
    with zipfile.ZipFile(folder/'task.bin') as archive:
        read_task(archive, state['job_id'], state['request_sha256'])
    state.update(owner=owner, input_dataset=owner+'/'+state['job_id']+'-input',
                 notebook=owner+'/'+state['job_id'], runtime_dataset=deployment['runtime_dataset'],
                 runtime_sha256=deployment['runtime_sha256'])
    save_json(job_path, state)  # Resource names survive an interrupted upload/push.
    upload = folder/'input-upload'
    upload.mkdir(exist_ok=True)
    archive_path = upload/'task.bin'
    if not archive_path.exists():
        try:
            os.link(folder/'task.bin', archive_path)
        except OSError:
            import shutil
            shutil.copyfile(folder/'task.bin', archive_path)
    if state['phase'] == 'prepared':
        # Persist before creating: in an ambiguous network failure, check the same unique ID.
        state['phase'] = 'creating-input'
        save_json(job_path, state)
        create_private_dataset(api, upload, state['input_dataset'])
        state['phase'] = 'input-uploaded'
        save_json(job_path, state)
    wait_dataset(api, state['input_dataset'])
    kernel = folder/'notebook'
    kernel.mkdir(exist_ok=True)
    (kernel/'run.py').write_text(make_script(deployment, state), encoding='utf-8')
    save_json(kernel/'kernel-metadata.json', dict(id=state['notebook'], title=state['job_id'],
        code_file='run.py', language='python', kernel_type='script', is_private=True,
        enable_gpu=True, enable_internet=True, machine_shape=deployment['accelerator'],
        dataset_sources=[state['runtime_dataset'], state['input_dataset']],
        competition_sources=[], kernel_sources=[]))
    if state['phase'] == 'submitting':
        raise ValueError('Previous push had an uncertain outcome. Check status before resubmitting; do not duplicate GPU runs.')
    state['phase'] = 'submitting'
    save_json(job_path, state)
    response = api.kernels_push(str(kernel), acc=deployment['accelerator'])
    error = getattr(response, 'error', None)
    invalid = [getattr(response, field, None) for field in
               ('invalidDatasetSources','invalidCompetitionSources','invalidKernelSources',
                'invalid_dataset_sources','invalid_competition_sources','invalid_kernel_sources')]
    if error or any(invalid):
        state['phase'] = 'push-rejected'
        save_json(job_path, state)
        raise RuntimeError('Notebook rejected: '+str(error or invalid))
    state.update(phase='submitted', version=getattr(response, 'versionNumber', None) or
                 getattr(response, 'version_number', None), url=getattr(response, 'url', None))
    save_json(job_path, state)
    print('SUBMITTED', state['notebook'], 'version', state['version'], flush=True)
    return state


def kernel_status(api, state):
    response = api.kernels_status(state['notebook'])
    value = response.status
    status = str(getattr(value, 'name', value)).lower()
    if status == 'cancel_acknowledged':
        status = 'cancelled'
    return status, getattr(response, 'failure_message', None)


def wait_job(api, state, timeout=43200, poll=30):
    deadline, previous = time.monotonic()+timeout, None
    while time.monotonic() < deadline:
        status, error = kernel_status(api, state)
        if status != previous:
            print('NOTEBOOK', status, error or '', flush=True)
            previous = status
        if status in ('complete','completed'):
            return
        if status in ('error','failed','cancelled','canceled'):
            raise RuntimeError('Notebook failed: '+str(error or status))
        time.sleep(max(15, poll))
    raise TimeoutError('Job still pending/running; poll later with the same job.json')


def verify_output(folder, state):
    folder = Path(folder)
    result = load_json(folder/'job-result.json')
    if result.get('job_id') != state['job_id'] or result.get('schema_version') != SCHEMA:
        raise ValueError('Downloaded result identity mismatch')
    if result.get('status') == 'failed' and 'archive' not in result:
        raise RuntimeError('Worker initialization failed: '+result.get('error', 'unknown'))
    if result.get('provenance', {}).get('runtime_sha256') != state['runtime_sha256']:
        raise ValueError('Downloaded runtime identity mismatch')
    if len(result.get('pages', [])) != state['pages']:
        raise ValueError('Downloaded page count mismatch')
    ok = sum(page.get('status') == 'ok' for page in result['pages'])
    expected_status = 'completed' if ok == state['pages'] else 'partial' if ok else 'failed'
    if (result.get('succeeded') != ok or result.get('failed') != state['pages']-ok or
            result.get('status') != expected_status):
        raise ValueError('Downloaded completion summary disagrees with page records')
    archive_path = folder/'results.zip'
    meta = result['archive']
    if archive_path.stat().st_size != meta['bytes'] or file_digest(archive_path) != meta['sha256']:
        raise ValueError('Result archive checksum mismatch')
    expected = {'job-result.json'}
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate output archive entries')
        internal = json.loads(archive.read('job-result.json'))
        if internal != {key: value for key, value in result.items() if key != 'archive'}:
            raise ValueError('Embedded and standalone reports disagree')
        for page in result['pages']:
            record_path = safe_name(page['id']+'/result.json')
            expected.add(record_path)
            if json.loads(archive.read(record_path)) != page:
                raise ValueError('Per-page record mismatch')
            for file in page.get('files', {}).values():
                name = safe_name(file['path'])
                expected.add(name)
                content = archive.read(name)
                if len(content) != file['bytes'] or digest(content) != file['sha256']:
                    raise ValueError('Output file checksum mismatch: '+name)
        if set(archive.namelist()) != expected:
            raise ValueError('Unexpected or missing output archive entries')
    return result


def fetch(api, state, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ref = state['notebook']+('/'+str(state['version']) if state.get('version') else '')
    # Explicit pagination avoids the SDK's locale-encoded automatic .log write on Windows.
    token = ''
    while True:
        _, token = api.kernels_output(ref, path=str(output),
            file_pattern=r'^(results\.zip|job-result\.json)$', quiet=True, page_token=token)
        if not token:
            break
    (output/'notebook.log').write_text(api.kernels_logs(state['notebook']), encoding='utf-8')
    result = verify_output(output, state)
    print('RESULT', result['status'], 'succeeded', result['succeeded'], 'failed', result['failed'])
    return result


def cleanup(api, state, yes=False):
    owner = api.config_values['username']
    expected_input, expected_kernel = owner+'/'+state['job_id']+'-input', owner+'/'+state['job_id']
    if (state.get('input_dataset') != expected_input or state.get('notebook') != expected_kernel or
            expected_input == state.get('runtime_dataset')):
        raise ValueError('Cleanup IDs must match this account and this unique task')
    print('DELETE ONLY:', expected_kernel, 'and', expected_input)
    if not yes:
        print('Preview only. Add --yes after downloading and verifying the outputs.')
        return
    status, _ = kernel_status(api, state)
    if status not in ('complete','completed','error','failed','cancelled','canceled'):
        raise ValueError('Refusing to delete a queued/running task')
    api.kernels_delete(expected_kernel, no_confirm=True)
    if not api.dataset_delete(owner, expected_input.split('/')[1], no_confirm=True):
        raise RuntimeError('Notebook deleted but input dataset deletion failed; retry manually')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', help='Caller-only raw Kaggle token or legacy kaggle.json; otherwise use Kaggle environment/config')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('account')
    p = sub.add_parser('publish-runtime', help='One-time private deployment after build_runtime.py')
    p.add_argument('--bundle', required=True)
    p.add_argument('--output', required=True, help='Deployment JSON for subsequent submit calls')
    p = sub.add_parser('prepare')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--options', help='JSON with defaults and source-name-keyed pages overrides')
    p = sub.add_parser('submit')
    p.add_argument('--job', required=True)
    p.add_argument('--deployment', default=str(HERE/'deployment.json'))
    for name in ('status','wait','logs','fetch','cleanup'):
        p = sub.add_parser(name)
        p.add_argument('--job', required=True)
        if name == 'wait':
            p.add_argument('--timeout', type=int, default=43200)
        if name == 'fetch':
            p.add_argument('--output', required=True)
        if name == 'cleanup':
            p.add_argument('--yes', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.input, args.output, load_json(args.options) if args.options else None)
        print('PREPARED', result['job_id'], result['pages'], 'pages')
        return 0
    api = authenticate(args.key_file)
    if args.command == 'publish-runtime':
        publish_runtime(api, args.bundle, args.output)
        return 0
    if args.command == 'account':
        quota = api.quota_view()
        print('ACCOUNT', api.config_values['username'], 'GPU used', quota.gpu_quota.time_used,
              'of', quota.gpu_quota.total_time_allowed)
        return 0
    state = load_json(args.job)
    if args.command == 'submit':
        submit(api, args.job, load_json(args.deployment))
    elif args.command == 'status':
        print(*kernel_status(api, state))
    elif args.command == 'wait':
        wait_job(api, state, args.timeout)
    elif args.command == 'logs':
        print(api.kernels_logs(state['notebook']))
    elif args.command == 'fetch':
        result = fetch(api, state, args.output)
        return 0 if result['status'] == 'completed' else 2
    elif args.command == 'cleanup':
        cleanup(api, state, args.yes)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, KeyError, zipfile.BadZipFile) as error:
        print(type(error).__name__+': '+str(error), file=sys.stderr)
        sys.exit(1)
