"""Prepare a separate paired dataset; never reassign the source split."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from handgesture.preprocessing import HandPreprocessor, PreprocessingConfig, sha256
from handgesture.runtime import progress, validate_workers

GESTURES = ['fist','one','peace','three','four','palm']
MAPPING = [{'class_id': i, 'gesture': g} for i,g in enumerate(GESTURES)]
LM_COLUMNS = [f'lm_{i}' for i in range(63)]
SPLITS = ['train','validation','test']


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Invalid or missing path: {relative}')
    return path


def _extract_record(row,source_root,stage,processor):
    item = {'sample_id':row['sample_id'],'split':row['split'],
            'class_id':int(row['class_id']),'gesture':row['gesture']}
    path = safe_path(source_root,row['image_path'])
    if row.get('image_sha256') and sha256(path) != row['image_sha256']:
        raise ValueError(f'Source image hash mismatch: {path}')
    try:
        with Image.open(path) as image:
            rgb = np.array(ImageOps.exif_transpose(image).convert('RGB'))
        result = processor.process(rgb)
        item['status'] = result.status
        item['reason'] = result.reason
        if result.status == 'ok':
            # ID-derived names avoid collisions and path traversal.
            name = __import__('hashlib').sha256(row['sample_id'].encode()).hexdigest()+'.png'
            relative = f"images/{row['split']}/{row['gesture']}/{name}"
            destination = stage/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(result.roi).save(destination)
            item.update(image_path=relative,image_sha256=sha256(destination),
                        width=result.roi.shape[1],height=result.roi.shape[0],
                        source_image_path=row['image_path'],source_image_sha256=sha256(path),
                        source_label=row.get('source_label','train_val_'+row['gesture']),
                        source_filepath=row.get('source_filepath',row['image_path']),
                        roi_bbox=json.dumps(result.bbox),
                        **dict(zip(LM_COLUMNS,map(float,result.features))))
    except (OSError,ValueError) as exc:
        item.update(status='invalid',reason=f'{type(exc).__name__}: {exc}')
    return item


_WORKER_CONTEXT = None


def _initialize_worker(config,model_path,source_root,stage,processor):
    global _WORKER_CONTEXT
    detector=processor or HandPreprocessor(config,model_path,mode='IMAGE')
    _WORKER_CONTEXT=(source_root,stage,detector)


def _worker_record(row):
    return _extract_record(row,*_WORKER_CONTEXT)


def prepare_dataset(source_root, output_root, model_path=None, processor=None, limit=None, workers=0):
    validate_workers(workers)
    progress(stage="preprocessing", status="running", completed=0, total=None, model=None)
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if source_root == output_root or output_root.is_relative_to(source_root):
        raise ValueError('Derived data must be outside source dataset')
    manifest = source_root/'metadata/manifest.csv'
    frame = pd.read_csv(manifest,float_precision='round_trip')
    if not frame.sample_id.is_unique:
        raise ValueError('Duplicate source sample IDs')
    if limit is not None:
        if limit < 1: raise ValueError('limit must be positive')
        frame = frame.groupby(['split','class_id'],sort=True).head(limit).copy()
    owned = processor is None
    config = processor.config if processor else PreprocessingConfig(sha256(model_path))
    identity = {'source_manifest_sha256':sha256(manifest),
                'preprocessing_fingerprint':config.fingerprint,'limit_per_class_split':limit}
    if output_root.exists():
        existing, report = load_dataset(output_root)
        if report['identity'] != identity:
            raise ValueError('Output already contains a different dataset')
        progress(stage='preprocessing', status='complete', completed=len(frame), total=len(frame), reused=True)
        return existing, report
    stage = output_root.with_name(output_root.name+'.staging')
    stage.mkdir(parents=True,exist_ok=True)
    identity_path = stage/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Staging identity differs; choose another output root')
    write_json(identity_path,identity)
    journal = stage/'progress.jsonl'
    completed = {}
    if journal.exists():
        lines=journal.read_bytes().splitlines(keepends=True)
        valid_bytes=0
        for index,line in enumerate(lines):
            try:item=json.loads(line)
            except (json.JSONDecodeError,UnicodeDecodeError):
                if index!=len(lines)-1:raise ValueError('Corrupt preprocessing journal before last record')
                # A crash can interrupt the final append. Retry only that record.
                with journal.open('r+b') as stream:stream.truncate(valid_bytes)
                break
            completed[item['sample_id']] = item
            valid_bytes+=len(line)
        if completed and journal.read_bytes()[-1:]!=b'\n':
            with journal.open('ab') as stream:stream.write(b'\n')
    if workers < 0:raise ValueError('workers must be nonnegative')
    pool=None
    if workers>1:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=get_context('fork'),
             initializer=_initialize_worker,initargs=(config,model_path,source_root,stage,processor))
    elif processor is None:
        processor = HandPreprocessor(config,model_path,mode='IMAGE')
    pending=[r for r in frame.to_dict('records') if r['sample_id'] not in completed]
    initial=len(completed)
    progress(stage="preprocessing", completed=initial, total=len(frame), resumed=bool(initial))
    try:
        items=(pool.map(_worker_record,pending,chunksize=8) if pool else
               (_extract_record(row,source_root,stage,processor) for row in pending))
        with journal.open('a') as log:
            for index,item in enumerate(items,initial+1):
                log.write(json.dumps(item,allow_nan=False)+'\n');log.flush()
                completed[item['sample_id']] = item
                if index == initial+1 or index % 100 == 0 or index == len(frame):
                    progress(stage='preprocessing', completed=index, total=len(frame))
    finally:
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        elif owned:processor.close()
    records = [completed[s] for s in frame.sample_id]
    # Cropping may make previously different files identical. Reject all copies
    # rather than let duplicate crops leak across splits or retain label conflicts.
    hashes = {}
    for record in records:
        if record['status']=='ok': hashes.setdefault(record['image_sha256'],[]).append(record)
    for group in hashes.values():
        if len(group)>1:
            for record in group:
                record.update(status='invalid',reason='duplicate_derived_image')
                (stage/record['image_path']).unlink(missing_ok=True)
    accepted = pd.DataFrame([r for r in records if r['status']=='ok']).drop(columns=['status','reason'],errors='ignore')
    rejected = pd.DataFrame([r for r in records if r['status']!='ok'],
                            columns=['sample_id','split','class_id','gesture','status','reason'])
    if accepted.empty:
        raise ValueError('No accepted samples; inspect staging progress.jsonl')
    metadata = stage/'metadata';metadata.mkdir(exist_ok=True)
    accepted.sort_values('sample_id').to_csv(metadata/'manifest.csv',index=False)
    rejected.to_csv(metadata/'rejections.csv',index=False)
    report = {'format_version':2,'identity':identity,'preprocessing':asdict(config),
              'preprocessing_fingerprint':config.fingerprint,'class_mapping':MAPPING,
              'smoke_test':limit is not None,
              'counts':{'source':len(frame),'accepted':len(accepted),'rejected':len(rejected)},
              'source_split_counts':frame.split.value_counts().to_dict(),
              'split_counts':accepted.split.value_counts().to_dict(),
              'class_split_counts':accepted.groupby(['split','gesture']).size().unstack(fill_value=0).to_dict(),
              'rejection_reasons':rejected.reason.value_counts().to_dict(),
              'artifact_sha256':{p.name:sha256(p) for p in [metadata/'manifest.csv',metadata/'rejections.csv']},
              'limits':['source random split retained; not subject-wise',
                        'source images were previously cropped; webcam domain differs']}
    write_json(metadata/'preparation_report.json',report)
    # Validate complete staged artifact before publishing it.
    load_dataset(stage)
    output_root.parent.mkdir(parents=True,exist_ok=True)
    os.rename(stage,output_root)
    return load_dataset(output_root)


def load_dataset(root, verify_images=True):
    root = Path(root)
    metadata = root/'metadata'
    report = json.loads((metadata/'preparation_report.json').read_text())
    for name,digest in report['artifact_sha256'].items():
        if sha256(metadata/name) != digest:
            raise ValueError(f'Metadata hash mismatch: {name}')
    config = PreprocessingConfig(**report['preprocessing'])
    config.assert_matches(report['preprocessing_fingerprint'])
    frame = pd.read_csv(metadata/'manifest.csv',float_precision='round_trip')
    required = ['sample_id','split','class_id','gesture','image_path','image_sha256',*LM_COLUMNS]
    if set(required)-set(frame) or frame[required].isna().any().any():
        raise ValueError('Incomplete manifest')
    if not np.isfinite(frame[LM_COLUMNS].to_numpy()).all():
        raise ValueError('Nonfinite features')
    for col in ['sample_id','image_path','image_sha256']:
        if not frame[col].is_unique: raise ValueError(f'Duplicate {col}')
    if (frame.split.value_counts().to_dict() != report['split_counts']
            or set(frame.split) != set(SPLITS) or report['class_mapping'] != MAPPING):
        raise ValueError('Split counts or mapping mismatch')
    for split in SPLITS:
        if set(frame.loc[frame.split.eq(split),'class_id']) != set(range(6)):
            raise ValueError(f'Split {split} is missing classes')
    for c,g in enumerate(GESTURES):
        if not frame.loc[frame.class_id.eq(c),'gesture'].eq(g).all():
            raise ValueError('Invalid class mapping')
    if not set(frame.class_id).issubset(range(6)):
        raise ValueError('Invalid class ID')
    if verify_images:
        for row in frame.itertuples():
            if sha256(safe_path(root,row.image_path)) != row.image_sha256:
                raise ValueError(f'Image hash mismatch: {row.image_path}')
    return frame.sort_values('sample_id').reset_index(drop=True), report
