#!/usr/bin/env python3
"""Verify, stage and migrate prepared source data to split/class image folders."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd
from prepare_data import (CLASS_MAP, ARTIFACT_NAMES, PreparationError, load_manifest,
                          original_basename, read_input, read_canonical, sha256_file,
                          check_manifest_images, write_canonical)


def verify_model(data_dir: Path, assets_dir: Path, expected_hash=None):
    source = data_dir / 'hand_landmarker.task'
    destination = assets_dir / 'hand_landmarker.task'
    if source.is_file():
        model_hash = sha256_file(source)
        if destination.exists() and sha256_file(destination) != model_hash:
            raise PreparationError('Model đích đã tồn tại và không khớp; không ghi đè.')
    elif destination.is_file():
        model_hash = sha256_file(destination)
    else:
        raise PreparationError('Không tìm thấy hand_landmarker.task ở nguồn hoặc assets.')
    if expected_hash and model_hash != expected_hash:
        raise PreparationError('Model không khớp hash đã ghi nhận.')
    return model_hash



def legacy_fingerprints(data_dir: Path):
    """Fingerprint every managed legacy item, refusing symlinked deletion roots."""
    items = ['processed', 'landmarks.csv', 'no_hand_log.csv', 'hand_landmarker.task',
             *[f'prepared/{name}' for name in ARTIFACT_NAMES]]
    if (data_dir / 'prepared').is_symlink():
        raise PreparationError('Không chuyển đổi/xóa prepared/ dạng symlink.')
    result = {}
    for name in items:
        path = data_dir / name
        if path.is_symlink():
            raise PreparationError(f'Không xóa nguồn dạng symlink: {name}')
        if not path.exists():
            continue
        if path.is_dir():
            digest = hashlib.sha256()
            for member in sorted(path.rglob('*')):
                if member.is_symlink():
                    raise PreparationError(f'Không xóa nguồn chứa symlink: {member}')
                if member.is_file():
                    digest.update(member.relative_to(path).as_posix().encode('utf-8'))
                    digest.update(bytes.fromhex(sha256_file(member)))
            result[name] = digest.hexdigest()
        else:
            result[name] = sha256_file(path)
    return result


def verify_legacy_fingerprints(data_dir: Path, expected: dict):
    present = legacy_fingerprints(data_dir)
    for name, digest in present.items():
        if expected.get(name) != digest:
            raise PreparationError(f'Nguồn cũ thay đổi/không khớp; không xóa: {name}')

def cleanup_source(data_dir: Path):
    """Remove only the legacy dataset paths authorized in the migration plan."""
    if (data_dir / 'prepared').is_symlink():
        raise PreparationError('Không xóa prepared/ dạng symlink.')
    processed = data_dir / 'processed'
    if processed.exists():
        if processed.is_symlink():
            raise PreparationError('Không xóa processed/ dạng symlink.')
        shutil.rmtree(processed)
    for name in ('landmarks.csv', 'no_hand_log.csv', 'hand_landmarker.task'):
        (data_dir / name).unlink(missing_ok=True)
    prepared = data_dir / 'prepared'
    if prepared.is_dir():
        for name in ARTIFACT_NAMES:
            (prepared / name).unlink(missing_ok=True)
        if not any(prepared.iterdir()):
            prepared.rmdir()


def reorganize(data_dir: Path, assets_dir: Path, apply=False):
    data_dir = Path(data_dir).resolve()
    assets_dir = Path(assets_dir).resolve()
    if assets_dir == data_dir or data_dir in assets_dir.parents:
        raise PreparationError('assets-dir phải nằm ngoài data/.')
    if (data_dir / 'images').exists() or (data_dir / 'metadata').exists():
        manifest, no_hand, report = read_canonical(data_dir)
        model_hash = verify_model(data_dir, assets_dir, report['migration']['model_sha256'])
        if not (assets_dir / 'hand_landmarker.task').is_file():
            raise PreparationError('Bộ đích tồn tại nhưng model chưa được chuyển tới assets.')
        if not report['migration']['source_full_data_removed']:
            verify_legacy_fingerprints(data_dir, report['migration'].get('legacy_fingerprints', {}))
            if apply:
                cleanup_source(data_dir)
                report['migration']['source_full_data_removed'] = True
                report['migration']['status'] = 'complete'
                write_canonical(data_dir / 'metadata', manifest, no_hand, report)
            print('Bộ đích hợp lệ; nguồn cũ còn chờ dọn.' if not apply else 'Đã hoàn tất dọn nguồn cũ.')
        else:
            print('Bộ dữ liệu đã ở cấu trúc mới và được kiểm chứng; không thay đổi.')
        return report

    baseline_legacy = legacy_fingerprints(data_dir)
    prepared = data_dir / 'prepared'
    manifest = load_manifest(prepared / 'manifest.csv')
    old_report = json.loads((prepared / 'preparation_report.json').read_text(encoding='utf-8'))
    if sha256_file(prepared / 'manifest.csv') != old_report['artifact_sha256']['manifest.csv']:
        raise PreparationError('Manifest nguồn không khớp báo cáo.')
    for name, digest in old_report['input_sha256'].items():
        if name not in {'landmarks.csv', 'no_hand_log.csv'} or sha256_file(data_dir / name) != digest:
            raise PreparationError(f'CSV nguồn không khớp báo cáo: {name}')
    if len(manifest) != old_report['counts']['accepted_samples']:
        raise PreparationError('Số mẫu manifest không khớp báo cáo.')
    check_manifest_images(data_dir, manifest)
    expected_splits = {split: int(manifest['split'].eq(split).sum()) for split in ('train', 'validation', 'test')}
    if expected_splits != old_report['split_counts']:
        raise PreparationError('Split nguồn không khớp báo cáo.')
    source_files = [p for p in (data_dir / 'processed').rglob('*') if p.is_file()]
    if (data_dir / 'processed').is_symlink() or any(p.is_symlink() for p in (data_dir / 'processed').rglob('*')):
        raise PreparationError('Không chuyển đổi/xóa nguồn processed/ chứa symlink.')
    if any(p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'} for p in source_files):
        raise PreparationError('processed/ có file ngoài danh sách ảnh; không tự xóa.')
    if any(not str(row.image_path).startswith('processed/') for row in manifest.itertuples()):
        raise PreparationError('Manifest nguồn không trỏ vào processed/.')
    no_hand_all = read_input(data_dir / 'no_hand_log.csv', ['filepath', 'label'])
    no_hand = no_hand_all.loc[no_hand_all['label'].isin(CLASS_MAP), ['filepath', 'label']].copy()
    if len(no_hand) != old_report['counts']['selected_no_hand_rows']:
        raise PreparationError('Log no-hand không khớp số lượng đã kiểm chứng.')
    model_hash = verify_model(data_dir, assets_dir)
    new_manifest = manifest.copy()
    new_manifest['image_path'] = [f'images/{r.split}/{r.gesture}/{original_basename(r.source_filepath)}'
                                  for r in manifest.itertuples(index=False)]
    if not new_manifest['image_path'].is_unique:
        raise PreparationError('Đường dẫn đích trùng; không chuyển đổi.')
    print(f'Giữ {len(manifest):,} ảnh; xóa {len(source_files)-len(manifest):,} ảnh ngoài bộ được chọn.')
    print(f'Splits: {expected_splits}; giữ {len(no_hand):,} bản ghi no-hand.')
    print(f'Model đích: {assets_dir / "hand_landmarker.task"}')
    if not apply:
        print('Dry run: chưa di chuyển hay xóa file. Dùng --apply để thực hiện.')
        return old_report

    source_snapshot = {name: sha256_file(data_dir / name) for name in ('landmarks.csv', 'no_hand_log.csv')}
    source_snapshot['prepared/manifest.csv'] = sha256_file(prepared / 'manifest.csv')
    report = copy.deepcopy(old_report)
    report['format_version'] = 2
    report['config']['layout'] = 'images/<split>/<gesture>/<original_basename>'
    report['config']['image_check'] = 'canonical_all_images_EXIF_orientation_RGB_full_decode'
    report['migration'] = {
        'source_full_data_removed': False, 'status': 'published_cleanup_pending',
        'source_input_sha256': copy.deepcopy(old_report['input_sha256']),
        'legacy_fingerprints': baseline_legacy,
        'source_manifest_sha256': source_snapshot['prepared/manifest.csv'],
        'source_report_sha256': sha256_file(prepared / 'preparation_report.json'),
        'source_image_count': len(source_files), 'kept_images': len(manifest),
        'removed_unused_images': len(source_files)-len(manifest),
        'source_no_hand_rows': len(no_hand_all), 'kept_no_hand_rows': len(no_hand),
        'model_sha256': model_hash,
        'history_note': 'input_sha256 and all_* row counts describe deleted original source files.',
    }
    with tempfile.TemporaryDirectory(prefix='.data-reorganization-', dir=data_dir.parent) as staging_name:
        staging = Path(staging_name)
        for i, (old, new) in enumerate(zip(manifest.itertuples(index=False), new_manifest.itertuples(index=False)), 1):
            target = staging / new.image_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data_dir / old.image_path, target)
            if i % 5000 == 0:
                print(f'Đã staging {i:,}/{len(manifest):,} ảnh...', file=sys.stderr, flush=True)
        write_canonical(staging / 'metadata', new_manifest, no_hand, report)
        staged_manifest, staged_log, staged_report = read_canonical(staging)
        pd.testing.assert_frame_equal(manifest.drop(columns='image_path'), staged_manifest.drop(columns='image_path'))
        # Re-check sources before publishing and deleting; this also catches edits during staging.
        for name, digest in source_snapshot.items():
            if sha256_file(data_dir / name) != digest:
                raise PreparationError(f'Nguồn thay đổi trong khi staging: {name}')
        check_manifest_images(data_dir, manifest)
        if legacy_fingerprints(data_dir) != baseline_legacy:
            raise PreparationError('Nguồn cũ thay đổi trong khi staging; không xóa.')
        if verify_model(data_dir, assets_dir) != model_hash:
            raise PreparationError('Model thay đổi trong khi staging.')
        assets_dir.mkdir(parents=True, exist_ok=True)
        model_destination = assets_dir / 'hand_landmarker.task'
        if not model_destination.exists():
            shutil.copy2(data_dir / 'hand_landmarker.task', model_destination)
        if sha256_file(model_destination) != model_hash:
            raise PreparationError('Model sao chép không khớp; không xóa nguồn.')
        (staging / 'images').rename(data_dir / 'images')
        try:
            (staging / 'metadata').rename(data_dir / 'metadata')
        except OSError:
            (data_dir / 'images').rename(staging / 'images')
            raise
        cleanup_source(data_dir)
        staged_report['migration']['source_full_data_removed'] = True
        staged_report['migration']['status'] = 'complete'
        write_canonical(data_dir / 'metadata', staged_manifest, staged_log, staged_report)
    print('Chuyển đổi hoàn tất; bộ đích được kiểm chứng trước khi xóa nguồn cũ.')
    return staged_report


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Làm gọn dữ liệu 6 lớp; mặc định chỉ kiểm tra, --apply để chuyển đổi và xóa nguồn dư.')
    parser.add_argument('--data-dir', type=Path, default=root / 'data')
    parser.add_argument('--assets-dir', type=Path, default=root / 'assets/mediapipe')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    try:
        reorganize(args.data_dir.expanduser(), args.assets_dir.expanduser(), args.apply)
    except (PreparationError, OSError, ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
