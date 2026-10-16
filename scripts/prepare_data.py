#!/usr/bin/env python3
"""Prepare one shared six-class manifest for landmark and RGB classifiers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

CLASS_MAP = {
    'train_val_fist': {'class_id': 0, 'gesture': 'fist', 'application': 'Số 0 / búa'},
    'train_val_one': {'class_id': 1, 'gesture': 'one', 'application': 'Số 1'},
    'train_val_peace': {'class_id': 2, 'gesture': 'peace', 'application': 'Số 2 / kéo'},
    'train_val_three': {'class_id': 3, 'gesture': 'three', 'application': 'Số 3'},
    'train_val_four': {'class_id': 4, 'gesture': 'four', 'application': 'Số 4'},
    'train_val_palm': {'class_id': 5, 'gesture': 'palm', 'application': 'Số 5 / bao'},
}
LM_COLUMNS = [f'lm_{i}' for i in range(63)]
MANIFEST_COLUMNS = ['sample_id', 'source_label', 'class_id', 'gesture', 'source_filepath',
                    'image_path', 'image_sha256', 'width', 'height', 'split', *LM_COLUMNS]
EXCLUDED_COLUMNS = ['sample_id', 'source_label', 'source_filepath', 'image_path',
                    'image_sha256', 'reason', 'detail']
ARTIFACT_NAMES = ['manifest.csv', 'excluded_samples.csv', 'preparation_report.json']


class PreparationError(ValueError):
    """Input does not support a valid shared experiment protocol."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_input(path: Path, required: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={'filepath': 'string', 'label': 'string'})
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise PreparationError(f'{path.name}: thiếu cột bắt buộc: {", ".join(missing)}')
    return frame


def original_basename(value) -> str | None:
    if pd.isna(value) or not str(value).strip():
        return None
    return str(value).replace('\\', '/').rsplit('/', 1)[-1] or None


def index_images(data_dir: Path, frame: pd.DataFrame) -> dict:
    """Match full original basenames, including names containing underscores."""
    wanted = {label: set(group['original_name'].dropna()) for label, group in frame.groupby('label')}
    index = defaultdict(list)
    for label, names in wanted.items():
        folder = data_dir / 'processed' / label
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue
            suffixes = [path.name, *(path.name[i+1:] for i, char in enumerate(path.name) if char == '_')]
            matching_names = set(suffixes).intersection(names)
            if matching_names:
                # Prefer the longest original basename, avoiding nested suffix collisions.
                name = max(matching_names, key=len)
                index[(label, name)].append(path)
    return index


def split_samples(manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = manifest.groupby('class_id').size().reindex(range(6), fill_value=0)
    try:
        if counts.min() < 2:
            raise ValueError('mỗi lớp cần ít nhất hai mẫu trước khi chia')
        train_idx, remainder = train_test_split(
            manifest.index.to_numpy(), test_size=.30, random_state=seed,
            stratify=manifest['class_id'])
        validation_idx, test_idx = train_test_split(
            remainder, test_size=.50, random_state=seed,
            stratify=manifest.loc[remainder, 'class_id'])
    except ValueError as exc:
        raise PreparationError(f'Không đủ mẫu để stratify 70/15/15. Số lượng lớp: {counts.to_dict()}. {exc}') from exc
    result = manifest.copy()
    result['split'] = pd.Series(pd.NA, index=result.index, dtype='string')
    result.loc[train_idx, 'split'] = 'train'
    result.loc[validation_idx, 'split'] = 'validation'
    result.loc[test_idx, 'split'] = 'test'
    for split in ('train', 'validation', 'test'):
        if set(result.loc[result['split'].eq(split), 'class_id']) != set(range(6)):
            raise PreparationError(f'Split {split} không giữ đủ 6 lớp; cần bổ sung mẫu.')
    if (not result['sample_id'].is_unique or not result['image_sha256'].is_unique
            or result['split'].isna().any()):
        raise PreparationError('Manifest không bảo đảm ID/hash duy nhất và split đầy đủ.')
    return result


def prepare(data_dir: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    input_hashes = {name: sha256_file(data_dir / name) for name in ('landmarks.csv', 'no_hand_log.csv')}
    raw = read_input(data_dir / 'landmarks.csv', ['filepath', 'label', *LM_COLUMNS])
    raw_no_hand = read_input(data_dir / 'no_hand_log.csv', ['filepath', 'label'])
    selected = raw.loc[raw['label'].isin(CLASS_MAP)].copy()
    no_hand = raw_no_hand.loc[raw_no_hand['label'].isin(CLASS_MAP)].copy()
    if selected.empty:
        raise PreparationError('Không có mẫu landmark nào thuộc 6 lớp đã chọn.')
    selected['original_name'] = selected['filepath'].map(original_basename)
    selected['sample_id'] = selected['label'].astype(str) + '/' + selected['original_name'].fillna('<missing>')
    selected[LM_COLUMNS] = selected[LM_COLUMNS].apply(pd.to_numeric, errors='coerce')
    selected = selected.sort_values('sample_id', kind='stable').reset_index(drop=True)
    selected['finite'] = np.isfinite(selected[LM_COLUMNS].to_numpy(dtype=np.float64)).all(axis=1)
    selected['duplicate_id'] = selected['sample_id'].duplicated(keep=False)
    image_index = index_images(data_dir, selected)
    excluded_records, image_records = [], []
    decode_attempts = 0
    hash_failures = 0
    decode_failures = 0
    unmatched_paths = set(path for paths in image_index.values() for path in paths)
    available_local = sum(1 for label in CLASS_MAP
                          for path in (data_dir / 'processed' / label).glob('*')
                          if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})

    def exclude(record, reason, detail=''):
        excluded_records.append({
            key: record.get(key, '') for key in EXCLUDED_COLUMNS if key not in {'reason', 'detail'}
        } | {'reason': reason, 'detail': detail})

    for i, row in enumerate(selected.to_dict('records'), 1):
        label = str(row['label'])
        record = {'sample_id': row['sample_id'], 'source_label': label,
                  'source_filepath': '' if pd.isna(row['filepath']) else str(row['filepath']),
                  **CLASS_MAP[label], **{column: row[column] for column in LM_COLUMNS}}
        matches = image_index.get((label, row['original_name']), [])
        record['image_path'] = matches[0].relative_to(data_dir).as_posix() if len(matches) == 1 else ''
        if row['original_name'] is None:
            exclude(record, 'missing_source_filepath')
        elif len(matches) == 0:
            exclude(record, 'missing_image')
        elif len(matches) > 1:
            exclude(record, 'ambiguous_image', '; '.join(p.relative_to(data_dir).as_posix() for p in matches))
        else:
            # Decode every uniquely paired image, including rows with bad landmarks.
            # This lets conflicts on image content be audited independently of features.
            path = matches[0]
            if row['duplicate_id']:
                record['invalid_reason'] = 'duplicate_sample_id'
                record['invalid_detail'] = 'Không thể xác định duy nhất một bản ghi nguồn cho ID này.'
            try:
                record['image_sha256'] = sha256_file(path)
            except OSError as exc:
                hash_failures += 1
                exclude(record, 'unreadable_image', f'{type(exc).__name__}: {exc}')
                continue
            decode_attempts += 1
            try:
                with Image.open(path) as source:
                    rgb = ImageOps.exif_transpose(source).convert('RGB')
                    rgb.load()
                    record['width'], record['height'] = rgb.size
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                decode_failures += 1
                record['invalid_reason'] = 'unreadable_image'
                record['invalid_detail'] = f'{type(exc).__name__}: {exc}'
            if not row['finite'] and 'invalid_reason' not in record:
                record['invalid_reason'] = 'nonfinite_landmarks'
            image_records.append(record)
        if i % 2000 == 0 or i == len(selected):
            print(f'Đã kiểm tra {i:,}/{len(selected):,} bản ghi...', file=sys.stderr, flush=True)

    # Audit conflicts across all successfully hashed unique pairs, even if a vector is invalid.
    hash_groups = defaultdict(list)
    for record in image_records:
        hash_groups[record['image_sha256']].append(record)
    kept = []
    same_class_groups = cross_label_groups = 0
    for records in hash_groups.values():
        if len({r['class_id'] for r in records}) > 1:
            cross_label_groups += 1
            labels = ', '.join(sorted({r['source_label'] for r in records}))
            for record in records:
                exclude(record, 'image_label_conflict', labels)
            continue
        if len(records) > 1:
            same_class_groups += 1
        valid = []
        for record in records:
            if 'invalid_reason' in record:
                exclude(record, record['invalid_reason'], record.get('invalid_detail', ''))
            else:
                valid.append(record)
        if valid:
            valid.sort(key=lambda r: r['sample_id'])
            kept.append(valid[0])
            for record in valid[1:]:
                exclude(record, 'duplicate_image_same_class', f"Giữ {valid[0]['sample_id']}")

    manifest = pd.DataFrame(kept).reindex(columns=MANIFEST_COLUMNS)
    manifest = manifest.sort_values('sample_id', kind='stable').reset_index(drop=True)
    manifest = split_samples(manifest, seed)
    excluded = pd.DataFrame(excluded_records, columns=EXCLUDED_COLUMNS).sort_values(
        ['sample_id', 'reason'], kind='stable').reset_index(drop=True)
    if len(manifest) + len(excluded) != len(selected):
        raise PreparationError('Số mẫu chấp nhận + loại không khớp số bản ghi nguồn.')

    class_counts = []
    for label, mapping in CLASS_MAP.items():
        successful = int(selected['label'].eq(label).sum())
        failed = int(no_hand['label'].eq(label).sum())
        final = manifest.loc[manifest['source_label'].eq(label)]
        class_counts.append({
            'source_label': label, **mapping, 'selected_success_rows': successful,
            'selected_no_hand_rows': failed,
            'logged_failure_pct': 100 * failed / (successful + failed) if successful + failed else None,
            'accepted_samples': len(final),
            'excluded_rows': int(excluded['source_label'].eq(label).sum()),
            'split_counts': {split: int(final['split'].eq(split).sum())
                             for split in ('train', 'validation', 'test')},
        })
    split_counts = {split: int(manifest['split'].eq(split).sum())
                    for split in ('train', 'validation', 'test')}
    overlap = selected[['filepath', 'label']].merge(
        no_hand[['filepath', 'label']].drop_duplicates(), on=['filepath', 'label'], how='inner')
    report = {
        'format_version': 1,
        'class_mapping': CLASS_MAP,
        'config': {'seed': seed, 'requested_ratios': {'train': .70, 'validation': .15, 'test': .15},
                   'method': 'two_stage_stratified_random_split',
                   'sample_order': 'source_label/original_basename ascending',
                   'image_path_base': 'data_dir', 'landmark_columns': LM_COLUMNS,
                   'image_check': 'all_unique_pairs_EXIF_orientation_RGB_full_decode',
                   'duplicate_policy': 'cross_label_exclude_all; same_class_keep_smallest_valid_sample_id'},
        'environment': {'python_executable': sys.executable, 'python_version': platform.python_version(),
                        'packages': {name: importlib.metadata.version(name)
                                     for name in ('pandas', 'numpy', 'Pillow', 'scikit-learn')}},
        'input_sha256': input_hashes,
        'counts': {'all_success_rows': len(raw), 'all_no_hand_rows': len(raw_no_hand),
                   'selected_success_rows': len(selected), 'selected_no_hand_rows': len(no_hand),
                   'nonfinite_landmark_rows': int((~selected['finite']).sum()),
                   'duplicate_sample_id_rows': int(selected['duplicate_id'].sum()),
                   'uniquely_paired_rows': sum(len(image_index.get((label, name), [])) == 1
                                              for label, name in zip(selected['label'], selected['original_name'])),
                   'decode_attempts': decode_attempts, 'decode_failures': decode_failures,
                   'hash_read_failures': hash_failures,
                   'local_images_without_selected_source_match': available_local - len(unmatched_paths),
                   'accepted_samples': len(manifest), 'excluded_rows': len(excluded)},
        'exclusion_reason_counts': dict(Counter(excluded['reason'])),
        'duplicate_image_groups': {'same_class': same_class_groups, 'cross_label': cross_label_groups},
        'split_counts': split_counts,
        'actual_split_ratios': {split: count / len(manifest) for split, count in split_counts.items()},
        'per_class': class_counts,
        'no_hand_audit': {'duplicate_filepath_rows': int(no_hand['filepath'].duplicated().sum()),
                          'missing_filepath_rows': int(no_hand['filepath'].isna().sum()),
                          'success_no_hand_overlap_rows': len(overlap)},
        'limitations': [
            'Random stratification does not separate subjects or capture sessions; metadata unavailable.',
            'Landmark normalization and image crop provenance not verified from source code.',
            'Image SHA-256 detects identical file bytes only, not visually similar/re-encoded images.',
            'Classifier comparison uses accepted paired samples, not webcam end-to-end accuracy.',
            'No-hand counts describe provided logs; no-hand images are unavailable.',
        ],
    }
    if {name: sha256_file(data_dir / name) for name in input_hashes} != input_hashes:
        raise PreparationError('CSV nguồn thay đổi trong lúc chạy; không ghi đầu ra.')
    return manifest, excluded, report


def write_outputs(output_dir: Path, manifest: pd.DataFrame, excluded: pd.DataFrame, report: dict) -> None:
    # All validation precedes writes. Stage files on the same filesystem, then replace only owned names.
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=output_dir) as staging:
        staging = Path(staging)
        manifest.to_csv(staging / 'manifest.csv', index=False, float_format='%.17g')
        excluded.to_csv(staging / 'excluded_samples.csv', index=False)
        (staging / 'preparation_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        report['artifact_sha256'] = {
            name: sha256_file(staging / name) for name in ('manifest.csv', 'excluded_samples.csv')}
        (staging / 'preparation_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        for name in ARTIFACT_NAMES:
            os.replace(staging / name, output_dir / name)



def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision='round_trip')
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise PreparationError(f'Manifest thiếu cột: {sorted(missing)}')
    frame = frame[MANIFEST_COLUMNS]
    if frame.empty or frame[MANIFEST_COLUMNS[:10]].isna().any().any():
        raise PreparationError('Manifest rỗng hoặc có metadata thiếu.')
    if not frame['sample_id'].is_unique or not frame['image_sha256'].is_unique:
        raise PreparationError('Manifest có ID hoặc SHA-256 trùng.')
    if not np.isfinite(frame[LM_COLUMNS].to_numpy(dtype=float)).all():
        raise PreparationError('Manifest có landmark không hữu hạn.')
    for row in frame.itertuples(index=False):
        mapping = CLASS_MAP.get(row.source_label)
        basename = original_basename(row.source_filepath)
        if (mapping is None or row.class_id != mapping['class_id'] or row.gesture != mapping['gesture']
                or not basename or row.sample_id != f'{row.source_label}/{basename}'):
            raise PreparationError(f'ID/nhãn không nhất quán: {row.sample_id}')
        if row.split not in {'train', 'validation', 'test'}:
            raise PreparationError(f'Split không hợp lệ: {row.sample_id}')
        if len(str(row.image_sha256)) != 64 or any(c not in '0123456789abcdef' for c in str(row.image_sha256)):
            raise PreparationError(f'Hash không hợp lệ: {row.sample_id}')
    for split in ('train', 'validation', 'test'):
        if set(frame.loc[frame['split'].eq(split), 'class_id']) != set(range(6)):
            raise PreparationError(f'Split {split} không có đủ 6 lớp.')
    return frame


def check_manifest_images(data_dir: Path, frame: pd.DataFrame, canonical=False) -> None:
    expected_paths = set()
    for i, row in enumerate(frame.itertuples(index=False), 1):
        relative = Path(row.image_path)
        if relative.is_absolute() or '..' in relative.parts:
            raise PreparationError(f'Đường dẫn ảnh vượt ra ngoài data/: {row.image_path}')
        path = data_dir / relative
        try:
            path.resolve().relative_to(data_dir.resolve())
        except ValueError as exc:
            raise PreparationError(f'Ảnh nằm ngoài data/: {row.image_path}') from exc
        if canonical:
            expected = f'images/{row.split}/{row.gesture}/{row.sample_id.split("/", 1)[1]}'
            if row.image_path != expected:
                raise PreparationError(f'Đường dẫn canonical không đúng: {row.image_path}')
        if row.image_path in expected_paths:
            raise PreparationError(f'Đường dẫn ảnh trùng: {row.image_path}')
        expected_paths.add(row.image_path)
        if sha256_file(path) != row.image_sha256:
            raise PreparationError(f'SHA-256 ảnh không khớp: {row.image_path}')
        with Image.open(path) as source:
            rgb = ImageOps.exif_transpose(source).convert('RGB')
            rgb.load()
            if rgb.size != (row.width, row.height):
                raise PreparationError(f'Kích thước ảnh không khớp: {row.image_path}')
        if i % 5000 == 0:
            print(f'Đã xác minh {i:,}/{len(frame):,} ảnh...', file=sys.stderr, flush=True)
    if canonical:
        actual = {p.relative_to(data_dir).as_posix() for p in (data_dir / 'images').rglob('*') if p.is_file()}
        if actual != expected_paths:
            raise PreparationError('Thư mục images/ có file dư hoặc thiếu so với manifest.')


def read_canonical(data_dir: Path, seed=None):
    metadata = data_dir / 'metadata'
    report = json.loads((metadata / 'preparation_report.json').read_text(encoding='utf-8'))
    if report.get('format_version') != 2:
        raise PreparationError('Metadata chưa phải cấu trúc canonical phiên bản 2.')
    if seed is not None and seed != report['config']['seed']:
        raise PreparationError('Seed không khớp split cố định; muốn chia lại cần một tác vụ riêng.')
    for name in ('manifest.csv', 'no_hand_log.csv'):
        expected_hash = report.get('artifact_sha256', {}).get(name)
        if not expected_hash or sha256_file(metadata / name) != expected_hash:
            raise PreparationError(f'Hash artifact không khớp: {name}')
    manifest = load_manifest(metadata / 'manifest.csv')
    no_hand = read_input(metadata / 'no_hand_log.csv', ['filepath', 'label'])
    if not no_hand['label'].isin(CLASS_MAP).all():
        raise PreparationError('Log no-hand canonical chứa nhãn ngoài 6 lớp.')
    splits = {split: int(manifest['split'].eq(split).sum()) for split in ('train', 'validation', 'test')}
    if splits != report['split_counts'] or len(manifest) != report['counts']['accepted_samples']:
        raise PreparationError('Số lượng mẫu/split không khớp báo cáo.')
    if len(no_hand) != report['counts']['selected_no_hand_rows']:
        raise PreparationError('Số lượng no-hand không khớp báo cáo.')
    check_manifest_images(data_dir, manifest, canonical=True)
    return manifest, no_hand, report


def write_canonical(output_dir: Path, manifest: pd.DataFrame, no_hand: pd.DataFrame, report: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.metadata-', dir=output_dir) as staging:
        staging = Path(staging)
        manifest.to_csv(staging / 'manifest.csv', index=False, float_format='%.17g')
        no_hand.to_csv(staging / 'no_hand_log.csv', index=False)
        report['artifact_sha256'] = {name: sha256_file(staging / name)
                                     for name in ('manifest.csv', 'no_hand_log.csv')}
        (staging / 'preparation_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        for name in ('manifest.csv', 'no_hand_log.csv', 'preparation_report.json'):
            os.replace(staging / name, output_dir / name)

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Kiểm tra toàn bộ ảnh và tạo manifest 6 lớp + split 70/15/15.')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--output-dir', type=Path, help='Mặc định: metadata/ với canonical, prepared/ với nguồn cũ')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error('--seed phải trong [0, 2**32).')
    data_dir = args.data_dir.expanduser().resolve()
    canonical = (data_dir / 'metadata').exists() or (data_dir / 'images').exists()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else data_dir / ('metadata' if canonical else 'prepared'))
    try:
        if canonical:
            manifest, no_hand, report = read_canonical(data_dir, args.seed)
            excluded = pd.DataFrame(columns=EXCLUDED_COLUMNS)
            write_canonical(output_dir, manifest, no_hand, report)
        else:
            manifest, excluded, report = prepare(data_dir, args.seed)
            write_outputs(output_dir, manifest, excluded, report)
    except (PreparationError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    print(f'Thành công: {len(manifest):,} mẫu; loại {len(excluded):,} bản ghi.')
    print(f'Splits: {report["split_counts"]}')
    print(f'Đầu ra: {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
