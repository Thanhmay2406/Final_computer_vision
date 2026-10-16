"""Contracts for standalone Kaggle notebooks and their embedded support code."""
import importlib.util
import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / 'scripts/kaggle_notebook_support.py'


def support():
    assert SUPPORT.is_file(), 'Standalone notebook support has not been implemented.'
    spec = importlib.util.spec_from_file_location('kaggle_notebook_support', SUPPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebooks_are_valid_and_run_without_project_imports():
    for name in ['01_random_forest', '02_mlp', '03_cnn', '04_compare_models']:
        path = ROOT / 'notebooks/kaggle' / (name + '.ipynb')
        assert path.is_file(), f'Missing notebook: {name}'
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        code = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
        compile(code, name, 'exec')
        assert 'import kaggle_notebook_support' not in code
        assert all(not c.outputs for c in nb.cells if c.cell_type == 'code')


def test_smoke_sampling_keeps_original_splits_and_all_classes():
    mod = support()
    # Sampling contracts should run in a fresh clone without the optional dataset.
    full = pd.DataFrame([
        {'sample_id': f'{split}/{class_id}/{i:03d}', 'split': split, 'class_id': class_id}
        for split, size in [('train', 12), ('validation', 6), ('test', 6)]
        for class_id in range(6) for i in range(size)
    ])
    small = mod.smoke_subset(full, {'train': 8, 'validation': 4, 'test': 4}, seed=42)
    assert len(small) == 96
    assert small.groupby(['split', 'class_id']).size().to_dict() == {
        (split, c): n for split, n in [('train', 8), ('validation', 4), ('test', 4)] for c in range(6)}
    lookup = full.set_index('sample_id')
    assert small.set_index('sample_id')['split'].equals(lookup.loc[small.sample_id, 'split'])
    again = mod.smoke_subset(full.sample(frac=1, random_state=4), {'train': 8, 'validation': 4, 'test': 4}, 42)
    pd.testing.assert_frame_equal(small.reset_index(drop=True), again.reset_index(drop=True))


def test_scaler_only_fits_train_and_eval_images_are_deterministic():
    mod = support()
    train = np.array([[0.] * 63, [2.] * 63])
    validation = np.array([[500.] * 63])
    scaler, transformed = mod.fit_scaler(train)
    assert np.allclose(scaler.mean_, 1.)
    assert np.allclose(transformed, [[-1.] * 63, [1.] * 63])
    assert np.allclose(scaler.transform(validation), 499.)
    image = Image.fromarray(np.arange(16*32*3, dtype=np.uint8).reshape(16,32,3))
    a = mod.image_tensor(image, training=False, image_size=128)
    b = mod.image_tensor(image, training=False, image_size=128)
    assert a.shape == (3,128,128) and a.dtype == torch.float32
    assert torch.equal(a,b) and float(a.min()) >= 0 and float(a.max()) <= 1
    mod.seed_everything(42)
    augmented = mod.image_tensor(image, training=True, image_size=128)
    assert not torch.equal(a, augmented)
    assert np.allclose(scaler.mean_, 1.)


def test_device_selection_covers_cpu_and_available_gpu():
    mod = support()
    assert str(mod.select_device('auto', cuda_available=False)) == 'cpu'
    assert str(mod.select_device('auto', cuda_available=True)) == 'cuda'
    assert str(mod.select_device('cpu', cuda_available=True)) == 'cpu'
    with pytest.raises(ValueError):
        mod.select_device('cuda', cuda_available=False)


def test_checkpoint_round_trip_and_validation_selection(tmp_path):
    mod = support()
    model = mod.GestureMLP()
    model.eval()
    x = torch.randn(3,63)
    expected = model(x).detach()
    torch.save(model.state_dict(), tmp_path/'model.pt')
    restored = mod.GestureMLP()
    restored.load_state_dict(torch.load(tmp_path/'model.pt', weights_only=True, map_location='cpu'))
    restored.eval()
    assert torch.equal(restored(x).detach(), expected)
    assert mod.checkpoint_is_better(.8, .4, .7, .2)
    assert mod.checkpoint_is_better(.8, .3, .8, .4)
    assert not mod.checkpoint_is_better(.7, .1, .8, .4)


@pytest.mark.parametrize('tuned', [False, True])
def test_fit_neural_logs_used_lr_and_restores_best_checkpoint(tmp_path, monkeypatch, tuned):
    mod = support()
    model = torch.nn.Linear(1, 6)
    train_loader, validation_loader = object(), object()
    # Script validation plateaus and a loss tie-break so this tests orchestration,
    # not whether a tiny synthetic dataset happens to converge.
    validation_f1 = [.7, .8, .8, .8, .8, .8, .8]
    validation_loss = [.9, .8, .7, .9, .9, .9, .9]
    used_lr, used_weight_decay, saved_states = [], [], []

    def controlled_pass(model, loader, device, optimizer=None):
        if optimizer is not None:
            assert loader is train_loader
            used_lr.append(optimizer.param_groups[0]['lr'])
            used_weight_decay.append(optimizer.param_groups[0]['weight_decay'])
            with torch.no_grad():
                model.weight.add_(1.)
            saved_states.append({k: v.clone() for k, v in model.state_dict().items()})
            return 1., {'accuracy': .6, 'macro_f1': .6}, None
        assert loader is validation_loader
        index = len(used_lr) - 1
        return validation_loss[index], {'accuracy': .8, 'macro_f1': validation_f1[index]}, None

    monkeypatch.setattr(mod, 'neural_pass', controlled_pass)
    kwargs = {}
    if tuned:
        kwargs = {'optimizer_config': {'lr': 3e-4, 'weight_decay': 1e-4},
                  'scheduler_config': {'mode': 'max', 'factor': .5, 'patience': 3,
                                       'threshold': 1e-4, 'threshold_mode': 'abs', 'min_lr': 1e-6}}
    restored, history, _, _, best_epoch = mod.fit_neural(
        model, train_loader, validation_loader, torch.device('cpu'),
        epochs=7, patience=4, output_dir=tmp_path, **kwargs)
    expected_lr = [3e-4] * 6 + [1.5e-4] if tuned else [1e-3] * 7
    assert used_lr == pytest.approx(expected_lr)
    assert used_weight_decay == pytest.approx([1e-4] * 7)
    assert history.learning_rate.tolist() == pytest.approx(expected_lr)
    assert pd.read_csv(tmp_path/'history.csv').learning_rate.tolist() == pytest.approx(expected_lr)
    assert best_epoch == 3 and len(history) == 7
    assert not restored.training
    checkpoint = torch.load(tmp_path/'model.pt', weights_only=True, map_location='cpu')
    for key, value in saved_states[2].items():
        assert torch.equal(restored.state_dict()[key], value)
        assert torch.equal(checkpoint[key], value)


def test_cnn_tuned_notebook_config_and_mlp_defaults():
    configs = {}
    for name in ['02_mlp', '03_cnn']:
        nb = nbformat.read(ROOT/'notebooks/kaggle'/f'{name}.ipynb', as_version=4)
        config_cell = next(c for c in nb.cells if c.cell_type == 'code')
        namespace = {}
        exec(config_cell.source, namespace)
        configs[name] = namespace
    cnn, mlp = configs['03_cnn'], configs['02_mlp']
    assert cnn['MAX_EPOCHS'] == 50 and cnn['PATIENCE'] == 12
    assert cnn['RUN_NAME'] == 'cnn_tuned' and cnn['MODEL_NAME'] == 'cnn'
    assert cnn['OPTIMIZER_CONFIG'] == {'lr': 3e-4, 'weight_decay': 1e-4}
    assert cnn['SCHEDULER_CONFIG'] == {'mode': 'max', 'factor': .5, 'patience': 3,
                                      'threshold': 1e-4, 'threshold_mode': 'abs', 'min_lr': 1e-6}
    assert mlp['MAX_EPOCHS'] == 60 and mlp['PATIENCE'] == 8
    assert mlp['OPTIMIZER_CONFIG'] == {'lr': 1e-3, 'weight_decay': 1e-4}
    assert mlp['SCHEDULER_CONFIG'] is None


def comparison_fixture(folder, model, ids=('a','b')):
    folder.mkdir(parents=True)
    mapping = [{'class_id':i,'gesture':name} for i,name in enumerate(['fist','one','peace','three','four','palm'])]
    metric = {'model':model,'protocol':{'manifest_sha256':'abc','class_mapping':mapping,'smoke_test':True,
                                      'test_sample_ids':list(ids)}}
    import hashlib
    names = ['model_config.json','class_mapping.json','model.joblib' if model=='rf' else 'model.pt']
    if model=='mlp':names.append('scaler.joblib')
    for name in names:(folder/name).write_text('fixture artifact')
    metric['artifact_sha256'] = {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}
    (folder/'metrics.json').write_text(json.dumps(metric))
    frame = pd.DataFrame({'sample_id':list(ids),'y_true':[0,1],'y_pred':[0,1]})
    for i in range(6): frame[f'prob_{i}'] = [float(i==0),float(i==1)]
    frame.to_csv(folder/'test_predictions.csv',index=False)
    metric['test_predictions_sha256']=hashlib.sha256((folder/'test_predictions.csv').read_bytes()).hexdigest()
    (folder/'metrics.json').write_text(json.dumps(metric))


def test_comparison_rejects_missing_artifacts_and_different_protocols(tmp_path):
    mod = support()
    dirs = {model:tmp_path/model for model in ['rf','mlp','cnn']}
    with pytest.raises((ValueError,FileNotFoundError)):
        mod.load_comparison(dirs)
    for model,folder in dirs.items():comparison_fixture(folder,model)
    results = mod.load_comparison(dirs)
    assert set(results)=={'rf','mlp','cnn'}
    p=dirs['cnn']/'metrics.json'
    m=json.loads(p.read_text());m['protocol']['manifest_sha256']='different';p.write_text(json.dumps(m))
    with pytest.raises(ValueError):mod.load_comparison(dirs)
    m['protocol']['manifest_sha256']='abc';p.write_text(json.dumps(m))
    p=dirs['cnn']/'test_predictions.csv';frame=pd.read_csv(p);frame.loc[0,'y_true']=2;frame.to_csv(p,index=False)
    with pytest.raises(ValueError):mod.load_comparison(dirs)


def test_image_integrity_rejects_missing_and_replaced_content(tmp_path):
    from types import SimpleNamespace
    mod = support()
    path = tmp_path/'images/test/fist/id.jpg'
    path.parent.mkdir(parents=True)
    Image.new('RGB',(16,16),'red').save(path)
    row = SimpleNamespace(sample_id='train_val_fist/id.jpg', split='test', gesture='fist',
                          image_path='images/test/fist/id.jpg', image_sha256=mod.file_sha256(path))
    mod.validate_image(tmp_path,row)
    path.write_bytes(b'corrupt replacement')
    with pytest.raises(ValueError,match='SHA-256'):mod.validate_image(tmp_path,row)
    path.unlink()
    with pytest.raises(FileNotFoundError):mod.validate_image(tmp_path,row)


def test_comparison_rejects_mixed_checkpoints(tmp_path):
    mod = support()
    dirs = {model:tmp_path/model for model in ['rf','mlp','cnn']}
    for model,folder in dirs.items():comparison_fixture(folder,model)
    (dirs['cnn']/'model.pt').write_bytes(b'another compatible checkpoint')
    with pytest.raises(ValueError,match='artifact'):mod.load_comparison(dirs)


def test_comparison_rejects_modified_prediction_file(tmp_path):
    mod = support()
    dirs = {model:tmp_path/model for model in ['rf','mlp','cnn']}
    for model,folder in dirs.items():comparison_fixture(folder,model)
    path=dirs['rf']/'test_predictions.csv'
    prediction=pd.read_csv(path)
    prediction.loc[0,'prob_0']=.75
    prediction.loc[0,'prob_1']=.25
    prediction.to_csv(path,index=False)
    with pytest.raises(ValueError,match='prediction'):mod.load_comparison(dirs)
