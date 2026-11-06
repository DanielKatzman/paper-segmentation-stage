"""Headless v2 release interface; the frozen predictor remains SegFormer v3."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def metadata():
    return json.loads(Path(__file__).with_name('compatibility.json').read_text(encoding='utf-8'))


def verify_models(models):
    from .inference.release import load_release
    release = load_release(release_directory=models)
    expected = metadata()['model']
    if release.model_sha256 != expected['sha256'] or sha(release.release_path) != expected['release_manifest_sha256']:
        raise ValueError('Models/configuration differ from this distribution')
    return release


def verify_output(root):
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    root = Path(root).resolve()
    rows = [json.loads(line) for line in (root/'segmentation_manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = set()
    by_source = {}
    for row in rows:
        capture_id = row['capture_id']
        if capture_id in ids:
            raise ValueError('Duplicate capture ID')
        ids.add(capture_id)
        def artifact(key):
            relative = Path(row[key])
            path = (root/relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Invalid artifact path: '+key)
            return path
        full = np.asarray(Image.open(artifact('mask_path')).convert('L'))
        crop = np.asarray(Image.open(artifact('piece_mask_path')).convert('L'))
        rgba = np.asarray(Image.open(artifact('piece_path')).convert('RGBA'))
        artifact('overlay_path')
        x, y, w, h = row['bbox_xywh']
        if (min(x, y) < 0 or min(w, h) < 1 or full.shape != tuple(row['original_size_hw'])
                or x+w > full.shape[1] or y+h > full.shape[0]
                or crop.shape != (h, w) or rgba.shape != (h, w, 4)):
            raise ValueError('Invalid native crop geometry')
        if not np.isin(full, [0, 255]).all() or not np.array_equal(full[y:y+h, x:x+w], crop) or not np.array_equal(rgba[:, :, 3], crop):
            raise ValueError('Binary full/crop masks or alpha differ')
        if np.count_nonzero(full) != np.count_nonzero(crop):
            raise ValueError('Foreground extends outside its declared crop')
        if row['accepted'] and ndimage.label(full > 0, np.ones((3, 3), bool))[1] != 1:
            raise ValueError('Accepted mask must be one nonempty eight-connected component')
        if row['accepted'] and row['review_status'] == 'needs_review':
            raise ValueError('Review requirement contradicts acceptance')
        parent = row.get('source_capture_id', capture_id)
        for prior, previous in by_source.setdefault(parent, []):
            px,py,pw,ph = prior['bbox_xywh']
            left,top,right,bottom = max(x,px),max(y,py),min(x+w,px+pw),min(y+h,py+ph)
            if right>left and bottom>top and np.any((crop[top-y:bottom-y,left-x:right-x]>0)
                    & (previous[top-py:bottom-py,left-px:right-px]>0)):
                raise ValueError('Piece masks overlap within a source capture')
        by_source[parent].append((row,crop))
    run = json.loads((root/'segmentation_run.json').read_text(encoding='utf-8'))
    sources_ready = True
    if run.get('capture_mode') == 'multi':
        sources = run.get('sources', [])
        source_ids = [s['source_capture_id'] for s in sources]
        observations = [ident for s in sources for ident in s['observation_ids']]
        if (len(source_ids)!=len(set(source_ids)) or len(observations)!=len(set(observations))
                or set(observations)!=ids or run.get('source_count')!=len(sources)):
            raise ValueError('Source inventory and per-piece manifest disagree')
        for source in sources:
            parent = source['source_capture_id']
            if parent != 'capture_'+source['source_sha256']:
                raise ValueError('Source identity differs from its content hash')
            for row,_ in by_source.get(parent, []):
                if (row['capture_id'] not in source['observation_ids']
                        or row['source_sha256']!=source['source_sha256']):
                    raise ValueError('Piece refers to a different source capture')
            if not source['observation_ids'] and not source.get('empty_reviewed'):
                sources_ready = False
        if set(by_source)-set(source_ids):
            raise ValueError('Piece refers to an unknown source capture')
        if run['complete'] and len(sources)!=run['total_discovered']:
            raise ValueError('Complete publication omitted a source capture')
    return {'status': 'ready' if rows and run['complete'] and sources_ready and all(r['accepted'] for r in rows) else 'review_required',
            'pieces': sum(not r.get('empty_proposal',False) for r in rows),
            'observations': len(rows), 'sources': run.get('source_count', len(rows)),
            'accepted': sum(bool(r['accepted']) for r in rows), 'complete': run['complete']}


def run(args):
    from .pipeline.runner import run_pipeline
    from .pipeline.multi import run_multi_pipeline
    release = verify_models(args.models)
    source, output = args.input.resolve(), args.output.resolve()
    if not source.is_dir() or source == output or source in output.parents or output in source.parents:
        raise ValueError('Input must exist and output must not overlap it')
    request = {'input': str(source), 'source_files': files(source), 'review': args.review,
               'pdf_dpi': args.pdf_dpi, 'cpu_threads': args.cpu_threads,
               'model_sha256': release.model_sha256, 'release_sha256': sha(release.release_path),
               'interface_sha256': sha(Path(__file__)), 'runtime': sys.version,
               'pipeline_sha256': sha(Path(__file__).with_name('pipeline')/'runner.py'),
               'multi_pipeline_sha256': sha(Path(__file__).with_name('pipeline')/'multi.py'),
               'piece_mode': args.piece_mode, 'minimum_pixels': args.minimum_pixels}
    package_root = Path(__file__).parent
    request['producer_version'] = metadata()['release_version']
    from importlib.metadata import version
    request['dependency_versions'] = {name: version(name) for name in
        ('numpy', 'Pillow', 'scipy', 'onnxruntime', 'PyMuPDF')}
    request['package_sources_sha256'] = hashlib.sha256(json.dumps({
        p.relative_to(package_root).as_posix(): sha(p) for p in sorted(package_root.rglob('*.py'))
        if '__pycache__' not in p.parts}, sort_keys=True).encode()).hexdigest()
    if args.capture_profile:
        profile = json.loads(args.capture_profile.read_text(encoding='utf-8'))
        required = {'schema_version': 'stage1.capture-profile.v1', 'motion_model': 'rotation_translation',
                    'shared_native_pixel_scale': True, 'relative_piece_scale': 1,
                    'per_piece_resize': False, 'perspective': False}
        if any(profile.get(k) != v for k, v in required.items()) or profile.get('stage1_masks') not in ('segformer_v3_predicted', 'predicted_reviewed'):
            raise ValueError('Capture declaration must describe shared-scale predicted/reviewed masks')
        request['capture_profile'] = profile
    if not request['source_files']:
        raise ValueError('Input folder is empty')
    if output.exists() and not args.resume:
        raise ValueError('Use a new output directory or --resume for the identical request')
    output.mkdir(parents=True, exist_ok=True)
    lock = output/'.paper-stage1.lock'
    # A stale lock requires explicit operator recovery after checking the process.
    with lock.open('x', encoding='utf-8') as handle:
        import os
        handle.write(str(os.getpid()))
    try:
        context = output/'integration_request.json'
        if context.exists():
            if json.loads(context.read_text(encoding='utf-8')) != request:
                raise ValueError('Resume request identity changed; choose a new output')
        elif args.resume:
            raise ValueError('Resume requires an authenticated prior request')
        else:
            context.write_text(json.dumps(request, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        receipt = output/'integration_receipt.json'
        if receipt.exists():
            previous = json.loads(receipt.read_text(encoding='utf-8'))
            current = {p: h for p, h in files(output).items() if p not in (lock.name, receipt.name)}
            if current != previous['artifacts']:
                raise ValueError('Published Stage 1 artifacts changed; use a new output')
            return previous['result']
        if args.capture_profile:
            (output/'capture_profile.json').write_text(json.dumps(profile, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        # Recover partially materialized output using authenticated prediction caches;
        # regenerate final rows instead of trusting an interrupted final manifest.
        with redirect_stdout(sys.stderr):
            pipeline = run_multi_pipeline if args.piece_mode == 'multi' else run_pipeline
            options = {'minimum_pixels': args.minimum_pixels} if args.piece_mode == 'multi' else {}
            pipeline(source, output, review=args.review, cpu_threads=args.cpu_threads,
                         pdf_dpi=args.pdf_dpi, headless=True, reprocess=args.resume,
                         release_directory=args.models, **options)
        if files(source) != request['source_files']:
            raise ValueError('Input changed during segmentation; choose a new output')
        result = verify_output(output)
        artifacts = {p: h for p, h in files(output).items() if p not in (lock.name, receipt.name)}
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps({'result': result, 'artifacts': artifacts}, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        temporary.replace(receipt)
        return result
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    v = sub.add_parser('verify'); v.add_argument('--models', type=Path, required=True)
    i = sub.add_parser('inspect'); i.add_argument('--input', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--input', type=Path, required=True); r.add_argument('--output', type=Path, required=True)
    r.add_argument('--models', type=Path, required=True)
    r.add_argument('--review', choices=('none', 'flagged', 'all'), default='flagged')
    r.add_argument('--cpu-threads', type=int, default=4); r.add_argument('--pdf-dpi', type=int, default=300)
    r.add_argument('--resume', action='store_true')
    r.add_argument('--piece-mode', choices=('multi', 'single'), default='multi')
    r.add_argument('--minimum-pixels', type=int, default=4)
    r.add_argument('--capture-profile', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'verify':
            release = verify_models(args.models)
            result = {'status': 'verified', 'release_version': metadata()['release_version'], 'model_sha256': release.model_sha256, 'device': 'cpu'}
        elif args.command == 'inspect':
            result = verify_output(args.input)
        else:
            if args.cpu_threads < 1 or args.pdf_dpi < 1 or args.minimum_pixels < 1:
                raise ValueError('Thread count and PDF DPI must be positive')
            result = run(args)
        print(json.dumps(result, sort_keys=True))
        return 3 if result['status'] == 'review_required' else 0
    except KeyboardInterrupt:
        print(json.dumps({'status': 'interrupted'})); return 130
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        print(json.dumps({'status': 'failed', 'error': str(exc)})); return 2


if __name__ == '__main__':
    raise SystemExit(main())
