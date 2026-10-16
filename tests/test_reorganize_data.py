"""Migration checks use real small datasets and never touch project data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
from test_prepare_data import make_data, run_cli, require_success, LM

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'reorganize_data.py'


def setup_source(tmp_path):
    data = tmp_path / 'data'
    make_data(data)
    require_success(run_cli(data, data / 'prepared'))
    (data / 'hand_landmarker.task').write_bytes(b'landmarker-model')
    extra = data / 'processed' / 'train_val_call'
    extra.mkdir()
    (extra / 'unused.png').write_bytes(b'extra-class-image')
    # This source row should be removed from the no-hand log.
    log = pd.read_csv(data / 'no_hand_log.csv')
    log = pd.concat([log, pd.DataFrame([{'filepath': '/kaggle/extra.png', 'label': 'train_val_call'}])])
    log.to_csv(data / 'no_hand_log.csv', index=False)
    report_path = data / 'prepared/preparation_report.json'
    report = json.loads(report_path.read_text())
    report['input_sha256']['no_hand_log.csv'] = hashlib.sha256((data / 'no_hand_log.csv').read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report))
    return data, tmp_path / 'assets' / 'mediapipe'


def migrate(data, assets, apply=False):
    args = [sys.executable, str(SCRIPT), '--data-dir', str(data), '--assets-dir', str(assets)]
    if apply:
        args += ['--apply']
    return subprocess.run(args, text=True, capture_output=True, cwd=data.parent)


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def test_dry_run_changes_nothing_and_apply_preserves_samples_and_model(tmp_path):
    data, assets = setup_source(tmp_path)
    old = pd.read_csv(data / 'prepared/manifest.csv', float_precision='round_trip')
    before = snapshot(data)
    require_success(migrate(data, assets))
    assert snapshot(data) == before and not assets.exists()
    require_success(migrate(data, assets, apply=True))
    new = pd.read_csv(data / 'metadata/manifest.csv', float_precision='round_trip')
    pd.testing.assert_frame_equal(old.drop(columns='image_path'), new.drop(columns='image_path'))
    assert len(new) == 120
    assert set(p.name for p in data.iterdir()) == {'images', 'metadata'}
    assert (assets / 'hand_landmarker.task').read_bytes() == b'landmarker-model'
    for row in new.itertuples():
        basename = row.sample_id.split('/', 1)[1]
        assert row.image_path == f'images/{row.split}/{row.gesture}/{basename}'
        assert hashlib.sha256((data / row.image_path).read_bytes()).hexdigest() == row.image_sha256
    log = pd.read_csv(data / 'metadata/no_hand_log.csv')
    assert len(log) == 1 and log.iloc[0].label == 'train_val_one'
    report = json.loads((data / 'metadata/preparation_report.json').read_text())
    assert report['migration']['source_full_data_removed'] is True
    assert report['migration']['removed_unused_images'] == 1
    assert report['migration']['model_sha256'] == hashlib.sha256(b'landmarker-model').hexdigest()
    before = snapshot(data)
    require_success(migrate(data, assets, apply=True))
    assert snapshot(data) == before


@pytest.mark.parametrize('problem', ['wrong_hash', 'missing_image', 'duplicate_target', 'existing_target'])
def test_invalid_source_or_existing_target_prevents_deletion(tmp_path, problem):
    data, assets = setup_source(tmp_path)
    manifest_path = data / 'prepared/manifest.csv'
    m = pd.read_csv(manifest_path, float_precision='round_trip')
    if problem == 'wrong_hash':
        (data / m.iloc[0].image_path).write_bytes(b'changed source')
    elif problem == 'missing_image':
        (data / m.iloc[0].image_path).unlink()
    elif problem == 'duplicate_target':
        m.loc[1, 'sample_id'] = m.loc[0, 'sample_id']
        m.to_csv(manifest_path, index=False, float_format='%.17g')
    else:
        (data / 'images').mkdir()
        (data / 'images/unrelated.txt').write_text('keep')
    before = snapshot(data)
    result = migrate(data, assets, apply=True)
    assert result.returncode != 0 and 'ERROR' in result.stderr
    assert snapshot(data) == before and (data / 'processed').exists()
    assert not assets.exists()


def test_canonical_prepare_keeps_split_and_rejects_seed_change(tmp_path):
    data, assets = setup_source(tmp_path)
    require_success(migrate(data, assets, apply=True))
    before = (data / 'metadata/manifest.csv').read_bytes()
    require_success(run_cli(data, data / 'metadata'))
    assert (data / 'metadata/manifest.csv').read_bytes() == before
    assert not (data / 'metadata/excluded_samples.csv').exists()
    result = run_cli(data, data / 'metadata', seed=43)
    assert result.returncode != 0 and 'ERROR' in result.stderr
    assert (data / 'metadata/manifest.csv').read_bytes() == before


def test_staging_hash_mismatch_cannot_delete_source(tmp_path, monkeypatch):
    data, assets = setup_source(tmp_path)
    # Exercise actual staging corruption rather than trusting source-only validation.
    sys.path.insert(0, str(SCRIPT.parent))
    import reorganize_data
    real_copy = reorganize_data.shutil.copy2

    def corrupt_copy(source, target, *args, **kwargs):
        result = real_copy(source, target, *args, **kwargs)
        if Path(target).suffix == '.png':
            Path(target).write_bytes(b'corrupt staging')
        return result

    before = snapshot(data)
    monkeypatch.setattr(reorganize_data.shutil, 'copy2', corrupt_copy)
    with pytest.raises(ValueError):
        reorganize_data.reorganize(data, assets, apply=True)
    assert snapshot(data) == before and not assets.exists()


def test_pending_cleanup_refuses_changed_legacy_sources(tmp_path):
    data, assets = setup_source(tmp_path)
    legacy = tmp_path / 'legacy'
    import shutil
    shutil.copytree(data, legacy)
    require_success(migrate(data, assets, apply=True))
    report_path = data / 'metadata/preparation_report.json'
    report = json.loads(report_path.read_text())
    report['migration']['source_full_data_removed'] = False
    report['migration']['status'] = 'published_cleanup_pending'
    report_path.write_text(json.dumps(report))
    shutil.copytree(legacy / 'processed', data / 'processed')
    for name in ['landmarks.csv', 'no_hand_log.csv', 'hand_landmarker.task']:
        shutil.copy2(legacy / name, data / name)
    shutil.copytree(legacy / 'prepared', data / 'prepared')
    m = pd.read_csv(data / 'prepared/manifest.csv')
    (data / m.iloc[0].image_path).write_bytes(b'changed after interrupted migration')
    before = snapshot(data)
    result = migrate(data, assets, apply=True)
    assert result.returncode != 0 and 'ERROR' in result.stderr
    assert snapshot(data) == before


def test_symlinked_prepared_directory_is_not_deleted(tmp_path):
    data, assets = setup_source(tmp_path)
    external = tmp_path / 'external'
    (data / 'prepared').rename(external)
    (data / 'prepared').symlink_to(external, target_is_directory=True)
    before = snapshot(external)
    result = migrate(data, assets, apply=True)
    assert result.returncode != 0 and 'ERROR' in result.stderr
    assert snapshot(external) == before
    assert (data / 'processed').exists()
