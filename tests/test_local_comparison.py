"""Local comparison must consume saved outputs without dataset/model loading."""
import builtins
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from test_result_outputs import fixture_result, support, tree_hashes

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('working_dir', ['', 'notebooks/kaggle'])
def test_resolve_local_results_from_kernel_directory(tmp_path, working_dir):
    mod = support()
    local = tmp_path/'project'
    (local/'results').mkdir(parents=True)
    cwd = local/working_dir
    cwd.mkdir(parents=True, exist_ok=True)
    dirs, output_root, results_root = mod.resolve_comparison_paths(
        start=cwd, kaggle_input=tmp_path/'missing-kaggle-input')
    assert dirs == {name: local/'results'/name for name in mod.MODEL_ORDER}
    assert results_root == local/'results'
    assert output_root == local/'results'


def test_explicit_comparison_paths_take_precedence(tmp_path):
    mod = support()
    (tmp_path/'results').mkdir()
    custom = tmp_path/'custom-results'
    explicit = {name: tmp_path/'attached'/name for name in mod.MODEL_ORDER}
    dirs, output_root, _ = mod.resolve_comparison_paths(results_root=custom, start=tmp_path)
    assert dirs == {name: custom/name for name in mod.MODEL_ORDER}
    assert output_root == custom
    dirs, output_root, _ = mod.resolve_comparison_paths(
        result_dirs=explicit, results_root=custom, output_root=tmp_path/'output', start=tmp_path)
    assert dirs == explicit and output_root == tmp_path/'output'


def test_explicit_dirs_without_project_use_cwd_results(tmp_path):
    mod = support()
    explicit = {name: tmp_path/'attached'/name for name in mod.MODEL_ORDER}
    dirs, output_root, results_root = mod.resolve_comparison_paths(
        result_dirs=explicit, start=tmp_path, kaggle_input=tmp_path/'absent')
    assert dirs == explicit and output_root == tmp_path/'results'
    assert results_root is None


def test_manual_external_dirs_do_not_select_kaggle_output_merely_because_it_exists(tmp_path):
    mod = support()
    attached = {name: tmp_path/'external'/name for name in mod.MODEL_ORDER}
    kaggle_input = tmp_path/'kaggle/input'
    kaggle_input.mkdir(parents=True)
    dirs, output_root, _ = mod.resolve_comparison_paths(
        result_dirs=attached, start=tmp_path, kaggle_input=kaggle_input,
        kaggle_output=tmp_path/'kaggle/working/hand_gesture')
    assert dirs == attached
    assert output_root == tmp_path/'results'


def test_kaggle_fallback_excludes_archive(tmp_path):
    mod = support()
    source = tmp_path/'kaggle/input'
    for name in mod.MODEL_ORDER:
        fixture_result(source/'dataset'/name, name, tmp_path/'data')
    fixture_result(source/'dataset/archive/old_cnn', 'cnn', tmp_path/'data')
    dirs, output_root, local_root = mod.resolve_comparison_paths(
        start=tmp_path, kaggle_input=source, kaggle_output=tmp_path/'kaggle/working/hand_gesture')
    assert dirs == {name: source/'dataset'/name for name in mod.MODEL_ORDER}
    assert output_root == tmp_path/'kaggle/working/hand_gesture'
    assert local_root is None


def test_default_notebook_needs_no_dataset_torch_or_model_loader(tmp_path, monkeypatch):
    for name in ['rf', 'mlp', 'cnn']:
        fixture_result(tmp_path/'results'/name, name, tmp_path/'unused-images')
    before = {name: tree_hashes(tmp_path/'results'/name) for name in ['rf', 'mlp', 'cnn']}
    output = tmp_path/'results/comparison'
    output.mkdir()
    (output/'benchmark_protocol.json').write_text('{"stale":true}')
    nb = nbformat.read(ROOT/'notebooks/kaggle/04_compare_models.ipynb', as_version=4)
    namespace = {'__name__': '__comparison_test__'}
    original_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name.split('.')[0] == 'torch':
            raise AssertionError('Default comparison must not import Torch')
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('Default comparison must not load dataset/models')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, '__import__', no_torch)
    import joblib
    monkeypatch.setattr(joblib, 'load', forbidden)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            exec(compile(cell.source, 'comparison-cell', 'exec'), namespace)
            for name in ['load_dataset', 'find_data_root', 'load_saved_model', 'benchmark_models']:
                if name in namespace:namespace[name] = forbidden
    assert namespace['RUN_BENCHMARK'] is False
    assert not (output/'benchmark_protocol.json').exists()
    table = pd.read_csv(output/'comparison.csv')
    assert table.model.tolist() == ['rf', 'mlp', 'cnn']
    assert table.accuracy.tolist() == [.5] * 3
    assert not any('latency' in column for column in table.columns)
    assert 'Chưa chạy benchmark' in (output/'report.md').read_text()
    # The notebook has completed under the import guard; our support-based hash utility imports Torch.
    monkeypatch.setattr(builtins, '__import__', original_import)
    for name, hashes in before.items():
        assert tree_hashes(tmp_path/'results'/name) == hashes


def test_benchmark_missing_dataset_fails_without_writing_reports(tmp_path):
    mod = support()
    dirs = {name: tmp_path/'inputs'/name for name in mod.MODEL_ORDER}
    for name, folder in dirs.items():
        fixture_result(folder, name, tmp_path/'images')
    with pytest.raises(ValueError, match='DATA_ROOT'):
        mod.run_comparison(dirs, tmp_path/'output', run_benchmark=True, data_root=tmp_path/'missing')
    assert not (tmp_path/'output').exists()


def test_missing_local_model_does_not_select_archived_run(tmp_path):
    mod = support()
    for name in ['rf', 'mlp']:
        fixture_result(tmp_path/'results'/name, name, tmp_path/'images')
    fixture_result(tmp_path/'results/archive/old_cnn', 'cnn', tmp_path/'images')
    dirs, output_root, _ = mod.resolve_comparison_paths(start=tmp_path)
    assert dirs['cnn'] == tmp_path/'results/cnn'
    with pytest.raises(FileNotFoundError):
        mod.run_comparison(dirs, output_root/'comparison')
    assert not (output_root/'comparison').exists()
