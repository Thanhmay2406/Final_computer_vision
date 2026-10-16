"""Manual Kaggle workflow exposes each expensive stage as a separate native cell."""
from pathlib import Path
import sys
import ast
import nbformat
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/kaggle_v2/07_interactive_workflow.ipynb'


def notebook():
    assert NOTEBOOK.exists(), 'A single manual workflow notebook must be distributed'
    return nbformat.read(NOTEBOOK,4)


def test_manual_workflow_has_independent_model_cells_and_no_background_runner():
    n=notebook();nbformat.validate(n)
    cells={c.metadata.get('step'):c for c in n.cells if c.cell_type=='code' and c.metadata.get('step')}
    assert {'configuration','dependencies','environment','inputs','preprocessing','rf','mlp',
            'gap','spatial','mobilenet_v3_small','validation','test_evaluation','export'}<=set(cells)
    for c in cells.values():
        tree=ast.parse(c.source)
        assert not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)
                       and x.func.attr in {'Popen','check_call','run_logged'} for x in ast.walk(tree))
        assert not c.outputs and c.execution_count is None
    config=cells['configuration'].source
    assert 'SMOKE_TEST = True' in config and 'EVALUATE_TEST = False' in config
    assert 'NUM_WORKERS = 0' in config
    for name in ['rf','mlp','gap','spatial','mobilenet_v3_small']:
        assert 'train_run' in cells[name].source and 'finalize(' not in cells[name].source
    assert 'finalize(' in cells['test_evaluation'].source


def test_manual_test_cell_does_not_evaluate_without_explicit_flag(capsys,tmp_path):
    cell=next(c for c in notebook().cells if c.metadata.get('step')=='test_evaluation')
    def unexpected(*args,**kwargs):pytest.fail('Test must not be evaluated before manual confirmation')
    scope={'EVALUATE_TEST':False,'finalize':unexpected}
    exec(compile(cell.source,'manual-test-cell','exec'),scope)
    assert 'EVALUATE_TEST' in capsys.readouterr().out


def test_manual_notebook_embeds_exact_shared_sources():
    n=notebook()
    for module,path in __import__('scripts.build_v2_notebooks',fromlist=['SOURCES']).SOURCES:
        c=next(c for c in n.cells if c.metadata.get('source_module')==module)
        # The AST literal is the source actually executed, not an approximate copy.
        tree=ast.parse(c.source)
        source=next(x.value.value for x in tree.body if isinstance(x,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='_source' for t in x.targets))
        assert source==(ROOT/path).read_text()


def test_configuration_change_cannot_finalize_stale_smoke_outputs(tmp_path):
    n=notebook()
    helpers=next(c for c in n.cells if c.metadata.get('step')=='display_helpers')
    manifest=tmp_path/'smoke/data_v2/metadata/manifest.csv';manifest.parent.mkdir(parents=True);manifest.write_text('sample_id\nfixture\n')
    from handgesture.preprocessing import sha256
    scope={'Path':Path,'OUTPUT_ROOT':tmp_path/'smoke','SMOKE_TEST':True,'SEED':42,
           'NUM_WORKERS':0,'DEVICE':'cpu','DATA_ROOT':None,'DETECTOR_PATH':None,'WEIGHTS_PATH':None,
           'sha256':sha256,'data':manifest.parent.parent}
    exec(helpers.source,scope)
    assert 'workflow_configuration' in scope and 'require_current_run' in scope
    scope['RUN_CONTEXT']=scope['workflow_configuration']()
    scope['RUN_DATA_MANIFEST']=sha256(manifest)
    scope['require_current_run']()
    scope['EVALUATE_TEST']=True
    scope['require_current_run']()  # Only the explicit test gate may be toggled.
    scope['SMOKE_TEST']=False;scope['OUTPUT_ROOT']=tmp_path/'full'
    def unexpected(*args,**kwargs):pytest.fail('Stale smoke data must never be finalized as full')
    scope['finalize']=unexpected
    cell=next(c for c in n.cells if c.metadata.get('step')=='test_evaluation')
    with pytest.raises(ValueError,match='cấu hình'):
        exec(cell.source,scope)


def test_input_discovery_ignores_image_tree(tmp_path):
    from scripts.build_v2_notebooks import RESOLVE
    source=tmp_path/'dataset';(source/'metadata').mkdir(parents=True)
    (source/'metadata/preparation_report.json').write_text('{}')
    decoy=source/'images/deep/metadata';decoy.mkdir(parents=True)
    (decoy/'preparation_report.json').write_text('{}')
    assets=tmp_path/'assets';assets.mkdir();(assets/'hand_landmarker.task').write_text('real detector')
    (decoy/'hand_landmarker.task').write_text('not an asset')
    scope={};exec(RESOLVE,scope);scope['SEARCH_ROOT']=tmp_path
    assert scope['find_dataset'](False)==source
    assert scope['find_asset']('hand_landmarker.task')==assets/'hand_landmarker.task'
