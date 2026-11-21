"""Build a private, pinned inference runtime. No credentials enter this bundle."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

MODELS = ('saber_yolo/saber_yolo.pt', 'default/detect-20241225.ckpt',
          'ctd/comictextdetector.pt', 'lama-manga/lama-manga.safetensors')
SABER_COMMIT = '13491d83c0f0cb974ed62a6ba662d164f721f24e'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(saber_root, engine, output):
    saber_root, engine, output = Path(saber_root), Path(engine), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=saber_root,text=True).strip()
    if revision != SABER_COMMIT:
        raise ValueError('This runtime requires the validated Saber commit '+SABER_COMMIT)
    archive = subprocess.check_output(['git','archive','--format=zip',revision,'src','LICENSE'],cwd=saber_root)
    manifest = dict(schema_version=1,saber_commit=revision,engine_sha256=sha256(engine),models={})
    with zipfile.ZipFile(output/'runtime.bin','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as target:
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            for item in source.infolist():
                if not item.is_dir():
                    target.writestr('Saber/'+item.filename,source.read(item))
        target.write(engine,'lama_engine.py')
        for model in MODELS:
            path = saber_root/'models'/model
            manifest['models'][model] = dict(sha256=sha256(path),bytes=path.stat().st_size)
            target.write(path,'Saber/models/'+model,compress_type=zipfile.ZIP_STORED)
        target.writestr('runtime-manifest.json',json.dumps(manifest,indent=2))
    manifest['runtime_sha256']=sha256(output/'runtime.bin')
    manifest['runtime_bytes']=(output/'runtime.bin').stat().st_size
    (output/'runtime-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saber-root',required=True)
    parser.add_argument('--engine',default=str(Path(__file__).resolve().parent.parent/'engine/lama_engine.py'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    build(args.saber_root,args.engine,args.output)
