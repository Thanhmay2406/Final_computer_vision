"""Integration contracts for the shared six-class preparation CLI."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_data.py'
LABELS = ['fist', 'one', 'peace', 'three', 'four', 'palm']
LM = [f'lm_{i}' for i in range(63)]


def make_data(folder, per_class=20):
    folder.mkdir()
    rows = []
    rng = np.random.default_rng(19)
    for class_id, gesture in enumerate(LABELS):
        label = f'train_val_{gesture}'
        images = folder / 'processed' / label
        images.mkdir(parents=True)
        for i in range(per_class):
            name = f'original_{i:03d}.png'
            Image.fromarray(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)).save(images / f'export_{name}')
            row = {'filepath': f'/kaggle/source/{label}/{name}', 'label': label}
            row.update({column: j / 100 + i / 1000 for j, column in enumerate(LM)})
            rows.append(row)
    # Deliberately reverse landmark columns to catch accidental lexicographic ordering.
    df = pd.DataFrame(rows)[['filepath', 'label', *reversed(LM)]]
    df.to_csv(folder / 'landmarks.csv', index=False)
    pd.DataFrame([{'filepath': '/kaggle/failed.png', 'label': 'train_val_one'}]).to_csv(
        folder / 'no_hand_log.csv', index=False)
    return df


def run_cli(data, output, seed=42, cwd=None):
    result = subprocess.run([sys.executable, str(SCRIPT), '--data-dir', str(data),
                             '--output-dir', str(output), '--seed', str(seed)],
                            cwd=cwd, text=True, capture_output=True)
    return result


def require_success(result):
    assert result.returncode == 0, result.stdout + result.stderr


def load_output(output):
    return (pd.read_csv(output / 'manifest.csv'),
            pd.read_csv(output / 'excluded_samples.csv'),
            json.loads((output / 'preparation_report.json').read_text()))


def test_shared_manifest_has_ordered_features_and_disjoint_stratified_splits(tmp_path):
    data = tmp_path / 'data'
    make_data(data)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*') if p.is_file()}
    out = tmp_path / 'out'
    require_success(run_cli(data, out, cwd=tmp_path))
    manifest, excluded, report = load_output(out)
    assert len(manifest) == 120
    assert excluded.empty
    assert manifest['split'].value_counts().to_dict() == {'train': 84, 'validation': 18, 'test': 18}
    assert manifest['sample_id'].is_unique and manifest['image_sha256'].is_unique
    assert manifest.groupby(['split', 'class_id']).size().to_dict() == {
        (split, c): count for split, count in [('train', 14), ('validation', 3), ('test', 3)] for c in range(6)}
    assert manifest.filter(regex=r'^lm_').columns.tolist() == LM
    row = manifest.loc[manifest.sample_id.eq('train_val_fist/original_000.png')].iloc[0]
    assert row.lm_2 == .02 and row.lm_10 == .10 and row.lm_62 == .62
    assert row.image_path == 'processed/train_val_fist/export_original_000.png'
    for class_id, gesture in enumerate(LABELS):
        assert manifest.loc[manifest.source_label.eq('train_val_' + gesture), 'class_id'].eq(class_id).all()
    assert report['counts']['selected_success_rows'] == 120
    assert report['counts']['selected_no_hand_rows'] == 1
    assert report['counts']['accepted_samples'] == 120
    assert set(report['input_sha256']) == {'landmarks.csv', 'no_hand_log.csv'}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())


def test_seed_and_input_order_reproducibility(tmp_path):
    data = tmp_path / 'data'
    df = make_data(data)
    out = tmp_path / 'out'
    require_success(run_cli(data, out))
    original = (out / 'manifest.csv').read_bytes()
    df.sample(frac=1, random_state=8).to_csv(data / 'landmarks.csv', index=False)
    require_success(run_cli(data, out, cwd=tmp_path))
    assert original == (out / 'manifest.csv').read_bytes()
    require_success(run_cli(data, out, seed=43))
    changed = pd.read_csv(out / 'manifest.csv')
    assert changed['split'].value_counts().to_dict() == {'train': 84, 'validation': 18, 'test': 18}
    assert original != (out / 'manifest.csv').read_bytes()
    assert changed['sample_id'].is_monotonic_increasing


def test_bad_landmarks_missing_ambiguous_and_corrupt_images_are_reported(tmp_path):
    data = tmp_path / 'data'
    df = make_data(data)
    folder = data / 'processed' / 'train_val_fist'
    df.loc[0, 'lm_9'] = np.inf
    df.loc[1, 'lm_10'] = np.nan
    df['lm_11'] = df['lm_11'].astype(object)
    df.loc[2, 'lm_11'] = 'not_numeric'
    (folder / 'export_original_003.png').unlink()
    (folder / 'export_original_004.png').write_bytes(b'not an image')
    (folder / 'other_original_005.png').write_bytes((folder / 'export_original_005.png').read_bytes())
    df.to_csv(data / 'landmarks.csv', index=False)
    out = tmp_path / 'out'
    require_success(run_cli(data, out))
    manifest, excluded, _ = load_output(out)
    assert len(manifest) == 114
    reasons = excluded.set_index('sample_id')['reason'].to_dict()
    assert reasons['train_val_fist/original_000.png'] == 'nonfinite_landmarks'
    assert reasons['train_val_fist/original_001.png'] == 'nonfinite_landmarks'
    assert reasons['train_val_fist/original_002.png'] == 'nonfinite_landmarks'
    assert reasons['train_val_fist/original_003.png'] == 'missing_image'
    assert reasons['train_val_fist/original_004.png'] == 'unreadable_image'
    assert reasons['train_val_fist/original_005.png'] == 'ambiguous_image'
    assert np.isfinite(manifest[LM].to_numpy()).all()


def test_duplicate_images_keep_smallest_id_and_cross_label_conflicts_are_all_excluded(tmp_path):
    data = tmp_path / 'data'
    make_data(data)
    fist = data / 'processed' / 'train_val_fist'
    one = data / 'processed' / 'train_val_one'
    (fist / 'export_original_001.png').write_bytes((fist / 'export_original_000.png').read_bytes())
    (one / 'export_original_000.png').write_bytes((fist / 'export_original_002.png').read_bytes())
    out = tmp_path / 'out'
    require_success(run_cli(data, out))
    manifest, excluded, report = load_output(out)
    reasons = excluded.set_index('sample_id')['reason'].to_dict()
    assert 'train_val_fist/original_000.png' in set(manifest.sample_id)
    assert reasons['train_val_fist/original_001.png'] == 'duplicate_image_same_class'
    assert reasons['train_val_fist/original_002.png'] == 'image_label_conflict'
    assert reasons['train_val_one/original_000.png'] == 'image_label_conflict'
    assert len(manifest) == 117
    assert manifest.image_sha256.is_unique
    assert report['duplicate_image_groups']['cross_label'] == 1
    assert report['duplicate_image_groups']['same_class'] == 1


@pytest.mark.parametrize('problem', ['schema', 'too_few'])
def test_invalid_input_leaves_existing_artifacts_untouched(tmp_path, problem):
    data = tmp_path / 'data'
    df = make_data(data, per_class=1 if problem == 'too_few' else 20)
    if problem == 'schema':
        df.drop(columns=['lm_62']).to_csv(data / 'landmarks.csv', index=False)
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'manifest.csv').write_text('previous valid artifact\n')
    result = run_cli(data, out)
    assert result.returncode != 0
    assert 'ERROR' in result.stderr
    assert (out / 'manifest.csv').read_text() == 'previous valid artifact\n'
    assert not (out / 'preparation_report.json').exists()


def test_nested_basename_suffixes_do_not_create_false_ambiguity(tmp_path):
    data = tmp_path / 'data'
    df = make_data(data)
    folder = data / 'processed' / 'train_val_fist'
    # original_000.png and 000.png are distinct original names.
    (folder / 'export_original_001.png').rename(folder / 'export_000.png')
    df.loc[1, 'filepath'] = '/kaggle/source/train_val_fist/000.png'
    df.to_csv(data / 'landmarks.csv', index=False)
    out = tmp_path / 'out'
    require_success(run_cli(data, out))
    manifest, excluded, _ = load_output(out)
    assert excluded.empty and len(manifest) == 120
    assert manifest.set_index('sample_id').loc['train_val_fist/000.png', 'image_path'] == 'processed/train_val_fist/export_000.png'


def test_duplicate_source_ids_still_contribute_to_cross_label_conflict_audit(tmp_path):
    data = tmp_path / 'data'
    df = make_data(data)
    fist = data / 'processed' / 'train_val_fist'
    one = data / 'processed' / 'train_val_one'
    (one / 'export_original_000.png').write_bytes((fist / 'export_original_000.png').read_bytes())
    pd.concat([df, df.iloc[[0]]], ignore_index=True).to_csv(data / 'landmarks.csv', index=False)
    out = tmp_path / 'out'
    require_success(run_cli(data, out))
    manifest, excluded, report = load_output(out)
    assert len(manifest) == 118
    assert len(excluded) == 3 and excluded['reason'].eq('image_label_conflict').all()
    assert 'train_val_one/original_000.png' not in set(manifest.sample_id)
    assert report['counts']['decode_attempts'] == 121
    assert report['duplicate_image_groups']['cross_label'] == 1
