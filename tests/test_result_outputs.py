"""Contracts for measured results, reporting and lossless local migration."""
import importlib.util
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
HISTORY_COLUMNS = ['step', 'epoch', 'learning_rate', 'train_loss', 'validation_loss',
                   'train_accuracy', 'validation_accuracy', 'train_macro_f1',
                   'validation_macro_f1', 'smoke_test']


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def support():
    return module('kaggle_notebook_support')


def fixture_result(folder, name, data_root):
    mod = support()
    folder.mkdir(parents=True)
    image_paths = ['images/test/fist/a.png', 'images/test/one/b.png']
    for path in image_paths:
        image = data_root/path
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (16, 16), 'red').save(image)
    probabilities = np.array([[0., 1., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]])
    measured = mod.classification_metrics([0, 1], probabilities)
    config = {'model': name, 'run_name': 'cnn_tuned' if name == 'cnn' else name,
              'learning_rate': 3e-4 if name == 'cnn' else 1e-3,
              'scheduler': {'name': 'ReduceLROnPlateau'} if name == 'cnn' else None,
              'best_epoch': 1 if name != 'rf' else None}
    for filename in mod.MODEL_ARTIFACTS[name]:
        (folder/filename).write_bytes(b'opaque checkpoint/scaler')
    mod.save_json(folder/'model_config.json', config)
    mod.save_json(folder/'class_mapping.json', mod.CLASS_MAPPING)
    prediction = pd.DataFrame({'sample_id': ['a', 'b'], 'y_true': [0, 1],
                               'image_path': image_paths, 'smoke_test': True, 'y_pred': [1, 1]})
    for i in range(6):
        prediction[f'prob_{i}'] = probabilities[:, i]
    prediction.to_csv(folder/'test_predictions.csv', index=False)
    metrics = {**measured, 'model': name, 'config': config, 'validation_metrics': measured,
               'protocol': {'manifest_sha256': 'same-manifest', 'seed': 42, 'smoke_test': True,
                            'class_mapping': mod.CLASS_MAPPING, 'test_sample_ids': ['a', 'b'],
                            'sample_counts': {'train': 2, 'validation': 2, 'test': 2}},
               'environment': {'device': 'cpu', 'cpu': 'test CPU', 'gpu': None},
               'training_time_s': 1., 'fit_wall_time_s': 2.,
               'artifact_sha256': {f: mod.file_sha256(folder/f) for f in mod.MODEL_ARTIFACTS[name]},
               'test_predictions_sha256': mod.file_sha256(folder/'test_predictions.csv')}
    mod.save_json(folder/'metrics.json', metrics)
    history = {'train_accuracy': [.75], 'validation_accuracy': [.5],
               'train_macro_f1': [.7], 'validation_macro_f1': [measured['macro_f1']], 'smoke_test': [True]}
    if name != 'rf':
        history.update(epoch=[1], train_loss=[.8], validation_loss=[1.])
    if name == 'cnn':
        history['learning_rate'] = [3e-4]
    pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
    return metrics, prediction


def tree_hashes(root):
    mod = support()
    return {str(p.relative_to(root)): mod.file_sha256(p) for p in root.rglob('*') if p.is_file()}


def test_history_schema_preserves_measurements_and_marks_not_applicable():
    mod = support()
    raw = pd.DataFrame({'train_accuracy': [.8], 'validation_accuracy': [.7],
                        'train_macro_f1': [.75], 'validation_macro_f1': [.65]})
    rf = mod.standardize_history(raw, 'rf', {}, False)
    assert list(rf.columns) == HISTORY_COLUMNS
    assert rf.step.tolist() == [1]
    assert rf[['epoch', 'learning_rate', 'train_loss', 'validation_loss']].isna().all().all()
    assert rf.train_accuracy.tolist() == [.8]
    mlp = mod.standardize_history(raw.assign(epoch=1, train_loss=.4, validation_loss=.6),
                                  'mlp', {'learning_rate': 1e-3}, False)
    assert list(mlp.columns) == HISTORY_COLUMNS
    assert mlp.learning_rate.tolist() == [1e-3]
    assert mlp.validation_loss.tolist() == [.6]
    # A missing variable LR must remain unknown, never be backfilled from initial LR.
    cnn = mod.standardize_history(raw.assign(epoch=1), 'cnn',
                                  {'learning_rate': 3e-4, 'scheduler': {'name': 'ReduceLROnPlateau'}}, False)
    assert cnn.learning_rate.isna().all()


def test_model_outputs_share_tables_and_all_report_links_exist(tmp_path):
    mod = support()
    for name in ['rf', 'mlp', 'cnn']:
        folder = tmp_path/name
        metrics, predictions = fixture_result(folder, name, tmp_path/'data')
        before = {f: mod.file_sha256(folder/f) for f in [*mod.MODEL_ARTIFACTS[name], 'metrics.json', 'test_predictions.csv']}
        mod.show_errors(predictions, tmp_path/'data', folder/'misclassified_gallery.png')
        mod.write_model_outputs(folder, metrics, predictions)
        summary = pd.read_csv(folder/'summary.csv')
        assert summary.split.tolist() == ['validation', 'test']
        assert summary.accuracy.tolist() == [.5, .5]
        assert summary.sample_count.tolist() == [2, 2]
        per_class = pd.read_csv(folder/'per_class_metrics.csv')
        assert per_class.gesture.tolist() == mod.GESTURES * 2
        assert per_class.support.tolist() == [1, 1, 0, 0, 0, 0] * 2
        assert per_class.f1_score.tolist() == pytest.approx([0., 2/3, 0., 0., 0., 0.] * 2)
        report = (folder/'report.md').read_text()
        assert 'SMOKE TEST' in report
        if name == 'cnn':
            assert 'cnn_tuned' in report
        for target in re.findall(r'\]\(([^)]+)\)', report):
            assert (folder/target).is_file(), target
        assert before == {f: mod.file_sha256(folder/f) for f in before}
        assert Image.open(folder/'history.png').size == (2340, 520)
        assert Image.open(folder/'confusion_matrix.png').size == (1560, 650)


def test_local_normalization_archives_once_and_is_repeatable(tmp_path):
    normalize = module('normalize_results')
    root, data = tmp_path/'results', tmp_path/'data'
    for name in ['rf', 'mlp', 'cnn']:
        folder = root/name if name != 'cnn' else root/'cnn/hand_gesture/cnn_tuned'
        fixture_result(folder, name, data)
    (root/'cnn/report.md').write_text('original tuning report')
    (root/'cnn/run_status.json').write_text('{"status":"COMPLETE"}')
    (root/'cnn/session.log').write_text('original execution log')
    original_cnn = tree_hashes(root/'cnn')
    protected = {name: {f: support().file_sha256((root/name if name != 'cnn' else root/'cnn/hand_gesture/cnn_tuned')/f)
                        for f in [*support().MODEL_ARTIFACTS[name], 'metrics.json', 'test_predictions.csv']}
                 for name in ['rf', 'mlp', 'cnn']}
    normalize.normalize_results(root, data)
    archive = root/'archive/cnn_tuned_20261015'
    assert tree_hashes(archive) == original_cnn
    assert not (root/'cnn/hand_gesture').exists()
    for name, hashes in protected.items():
        assert hashes == {f: support().file_sha256(root/name/f) for f in hashes}
        assert list(pd.read_csv(root/name/'history.csv').columns) == HISTORY_COLUMNS
    comparison = pd.read_csv(root/'comparison/comparison.csv')
    assert comparison.model.tolist() == ['rf', 'mlp', 'cnn']
    assert comparison.accuracy.tolist() == [.5] * 3
    assert not any('latency' in c or 'fps' in c for c in comparison.columns)
    assert (root/'comparison/report.md').is_file()
    stable = tree_hashes(root)
    normalize.normalize_results(root, data)
    assert tree_hashes(root) == stable
    assert list((root/'archive').iterdir()) == [archive]
    assert support().find_result_dirs(search_root=root) == {n: root/n for n in ['rf', 'mlp', 'cnn']}


def test_normalization_rejects_bad_metrics_before_any_write(tmp_path):
    normalize = module('normalize_results')
    root, data = tmp_path/'results', tmp_path/'data'
    for name in ['rf', 'mlp', 'cnn']:
        folder = root/name if name != 'cnn' else root/'cnn/hand_gesture/cnn_tuned'
        fixture_result(folder, name, data)
    path = root/'mlp/metrics.json'
    metrics = json.loads(path.read_text())
    metrics['accuracy'] = .99
    path.write_text(json.dumps(metrics))
    before = tree_hashes(root)
    with pytest.raises(ValueError, match='accuracy'):
        normalize.normalize_results(root, data)
    assert tree_hashes(root) == before


def test_comparison_legacy_inputs_link_existing_artifacts_only(tmp_path):
    mod = support()
    for name in ['rf', 'mlp', 'cnn']:
        fixture_result(tmp_path/name, name, tmp_path/'data')
    results = mod.load_comparison({name: tmp_path/name for name in ['rf', 'mlp', 'cnn']})
    output = tmp_path/'comparison'
    mod.export_comparison(results, output)
    report = (output/'report.md').read_text()
    targets = re.findall(r'\]\(([^)]+)\)', report)
    for target in targets:
        assert (output/target).is_file(), target
    for name in ['rf', 'mlp', 'cnn']:
        assert f'../{name}/metrics.json' in targets


def test_cli_figure_export_never_opens_interactive_window(tmp_path,monkeypatch):
    mod=support()
    def forbidden(*args,**kwargs):
        raise AssertionError('CLI report generation must not call interactive show')
    monkeypatch.setattr(mod.plt,'show',forbidden)
    fig=mod.plt.figure()
    monkeypatch.setattr(mod.plt,'get_backend',lambda:'TkAgg')
    mod.finish_figure(fig,tmp_path/'plot.png')
    assert (tmp_path/'plot.png').is_file()
