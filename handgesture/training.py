"""Train on train/validation only. Test evaluation is a separate finalization step."""
from dataclasses import asdict
from pathlib import Path
import copy
import json
import os
import random
import shutil
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from handgesture.data import load_dataset, write_json, MAPPING, GESTURES, LM_COLUMNS
from handgesture.models import build_model, image_tensor, GesturePredictor
from handgesture.preprocessing import sha256
from handgesture.runtime import progress, validate_workers, require_cuda
from scripts import kaggle_notebook_support as reporting


class ROIImages(Dataset):
    def __init__(self,frame,root,architecture,training=False):
        self.frame=frame.reset_index(drop=True);self.root=Path(root)
        self.architecture=architecture;self.training=training
    def __len__(self): return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        with Image.open(self.root/row.image_path) as image:
            x=image_tensor(image,self.architecture,self.training)
        return x,int(row.class_id)


def environment(device):
    value=reporting.environment_info(device)
    import importlib.metadata
    for name in ['torch','torchvision','mediapipe']:
        try:value['versions'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    value['gpu']=torch.cuda.get_device_name() if str(device).startswith('cuda') else None
    return value


def prediction_frame(frame,probabilities):
    p=np.asarray(probabilities)
    if p.shape!=(len(frame),6) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Invalid probability matrix')
    result=frame[['sample_id','class_id','image_path']].rename(columns={'class_id':'y_true'}).copy()
    result['y_pred']=p.argmax(1)
    for i in range(6):result[f'prob_{i}']=p[:,i]
    return result


def pass_epoch(model,loader,device,optimizer=None,freeze_backbone=False,progress_info=None):
    started=time.monotonic()
    model.train(optimizer is not None)
    if freeze_backbone: model.features.eval()
    losses=0.;prob=[];truth=[]
    with torch.set_grad_enabled(optimizer is not None):
        for batch_index,(x,y) in enumerate(loader,1):
            x,y=x.to(device),y.to(device)
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            logits=model(x);loss=nn.functional.cross_entropy(logits,y)
            if optimizer is not None:loss.backward();optimizer.step()
            losses+=float(loss.detach().cpu())*len(y)
            prob.append(logits.detach().softmax(1).cpu().numpy());truth.extend(y.cpu().tolist())
            if progress_info and (batch_index==1 or batch_index%25==0 or batch_index==len(loader)):
                progress(**progress_info, completed=batch_index, total=len(loader),
                         elapsed_seconds=time.monotonic()-started)
    values=np.concatenate(prob)
    return losses/len(truth),reporting.classification_metrics(truth,values),values


def make_loader(dataset,batch,shuffle,seed,workers):
    validate_workers(workers)
    kwargs={'multiprocessing_context':'fork'} if workers else {}
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=workers,
                      generator=torch.Generator().manual_seed(seed),
                      worker_init_fn=reporting.seed_worker,**kwargs)


def train_run(data_root,output_dir,kind,architecture=None,weights_path=None,
              smoke=False,device='auto',workers=0,seed=42,max_epochs=None):
    """No test image, feature, label or probability enters training/selection."""
    validate_workers(workers)
    if kind!='rf' and os.environ.get('HG_REQUIRE_CUDA')=='1':
        if device not in {'auto','cuda'}:raise RuntimeError('CUDA device required on Kaggle')
        require_cuda()
    if kind not in {'rf','mlp','cnn'}: raise ValueError('Unknown model')
    architecture=architecture or ('gap' if kind=='cnn' else kind)
    if kind!='cnn' and architecture!=kind:raise ValueError('Architecture does not match model kind')
    if kind=='cnn' and architecture not in {'gap','spatial','mobilenet_v3_small'}:
        raise ValueError('Unknown CNN architecture')
    progress(stage='training_setup',model=architecture,status='running',completed=0,total=None)
    frame,preparation=load_dataset(data_root)
    smoke=bool(smoke or preparation.get('smoke_test',False))
    splits={name:frame.loc[frame.split.eq(name)].copy() for name in ['train','validation']}
    if smoke:
        splits={name:rows.groupby('class_id',group_keys=False).head(8 if name=='train' else 4)
                for name,rows in splits.items()}
    output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f'Refusing to overwrite run: {output_dir}')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device)
    if kind=='rf':device=torch.device('cpu')
    mobile=architecture=='mobilenet_v3_small'
    official_weights='047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f'
    if mobile and weights_path is not None and sha256(weights_path)!=official_weights and not smoke:
        raise ValueError('Pretrained weights hash is not official IMAGENET1K_V1')
    # Build pretrained model before creating any output, and never silently fall back.
    model=None if kind=='rf' else build_model(architecture,initialize_pretrained=mobile,weights_path=weights_path)
    output_dir.mkdir(parents=True,exist_ok=True)
    fingerprint=preparation['preprocessing_fingerprint']
    protocol={'manifest_sha256':sha256(Path(data_root)/'metadata/manifest.csv'),
              'preprocessing_fingerprint':fingerprint,'class_mapping':MAPPING,'seed':seed,
              'smoke_test':smoke,'sample_counts':{**frame.split.value_counts().to_dict(),
                      **{name:len(rows) for name,rows in splits.items()}},
              'validation_sample_ids':splits['validation'].sample_id.tolist(),
              'limits':preparation['limits']}
    config={'model':kind,'architecture':architecture,'class_mapping':MAPPING,'features':LM_COLUMNS,
            'seed':seed,'smoke_test':smoke,'run_name':output_dir.name,
            'preprocessing':preparation['preprocessing'],'preprocessing_fingerprint':fingerprint,
            'image_size':224 if mobile else 128,'dropout':.3 if architecture=='spatial' else .2,
            'optimizer':'AdamW','weight_decay':1e-4,
            'train_augmentation':{'rotation_degrees':10,'brightness':[.85,1.15],
                                  'contrast':[.85,1.15],'flip':False},
            'selection':'validation macro-F1 descending, validation loss ascending'}
    if kind!='cnn':
        config.update(image_size=None,train_augmentation=None)
    if kind=='rf':
        config.update(dropout=None,optimizer=None,weight_decay=None)
    if mobile:
        config['pretrained']={'name':'IMAGENET1K_V1' if sha256(weights_path)==official_weights else 'TEST_FIXTURE_NOT_PRETRAINED',
              'sha256':sha256(weights_path),
              'url':'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
              'mean':[.485,.456,.406],'std':[.229,.224,.225]}
    history=[];start=time.perf_counter();training_time=0.;best_loss=None
    if kind=='rf':
        options={'n_estimators':10 if smoke else 300,'max_depth':20,'min_samples_leaf':2,
                 'max_features':'sqrt','n_jobs':1 if smoke else -1,'random_state':seed}
        config.update(options)
        model=RandomForestClassifier(**options)
        progress(stage='training',model='rf',status='running',completed=0,total=1)
        t=time.perf_counter();model.fit(splits['train'][LM_COLUMNS].to_numpy(),splits['train'].class_id)
        training_time=time.perf_counter()-t
        train_metrics=reporting.classification_metrics(splits['train'].class_id,model.predict_proba(splits['train'][LM_COLUMNS].to_numpy()))
        probabilities=model.predict_proba(splits['validation'][LM_COLUMNS].to_numpy())
        validation=reporting.classification_metrics(splits['validation'].class_id,probabilities)
        history=[{'step':1,'train_accuracy':train_metrics['accuracy'],'train_macro_f1':train_metrics['macro_f1'],
                  'validation_accuracy':validation['accuracy'],'validation_macro_f1':validation['macro_f1']}]
        joblib.dump(model,output_dir/'model.joblib')
    else:
        batch=256 if kind=='mlp' else 64
        if kind=='mlp':
            scaler=StandardScaler().fit(splits['train'][LM_COLUMNS].to_numpy())
            joblib.dump(scaler,output_dir/'scaler.joblib')
            datasets={name:TensorDataset(torch.from_numpy(scaler.transform(rows[LM_COLUMNS].to_numpy()).astype(np.float32)),
                          torch.tensor(rows.class_id.to_numpy(),dtype=torch.long)) for name,rows in splits.items()}
        else:
            datasets={name:ROIImages(rows,data_root,architecture,name=='train') for name,rows in splits.items()}
        loaders={name:make_loader(ds,batch,name=='train',seed,workers) for name,ds in datasets.items()}
        model=model.to(device)
        epochs=2 if smoke else (max_epochs or (60 if kind=='mlp' or not mobile else 40))
        patience=8 if kind=='mlp' or mobile else 12
        warmup=1 if smoke and mobile else 5 if mobile else 0
        lr=1e-3 if kind=='mlp' or mobile else 3e-4
        if mobile:
            for p in model.features.parameters():p.requires_grad_(False)
            optimizer=torch.optim.AdamW(model.classifier.parameters(),lr=1e-3,weight_decay=1e-4)
        else:optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        scheduler=None if kind=='mlp' or mobile else torch.optim.lr_scheduler.ReduceLROnPlateau(
              optimizer,mode='max',factor=.5,patience=3,threshold=1e-4,threshold_mode='abs',min_lr=1e-6)
        config.update(batch_size=batch,max_epochs=epochs,patience=patience,learning_rate=lr,
                      scheduler=None if scheduler is None else {'name':'ReduceLROnPlateau','mode':'max','factor':.5,
                           'patience':3,'threshold':1e-4,'threshold_mode':'abs','min_lr':1e-6},
                      transfer_schedule={'frozen_epochs':warmup,'backbone_lr':1e-5,'classifier_lr':1e-4} if mobile else None)
        best=(-float('inf'),float('inf'));best_state=None;bad=0
        for epoch in range(1,epochs+1):
            epoch_started=time.monotonic()
            info={'model':architecture,'epoch':epoch,'max_epochs':epochs,'status':'running'}
            if mobile and epoch==warmup+1:
                for p in model.features.parameters():p.requires_grad_(True)
                optimizer=torch.optim.AdamW([{'params':model.features.parameters(),'lr':1e-5},
                         {'params':model.classifier.parameters(),'lr':1e-4}],weight_decay=1e-4)
                bad=0
            rates=[group['lr'] for group in optimizer.param_groups]
            t=time.perf_counter()
            train_loss,train_metrics,_=pass_epoch(model,loaders['train'],device,optimizer,mobile and epoch<=warmup,
                progress_info={**info,'stage':'train_batches'})
            training_time+=time.perf_counter()-t
            loss,val,_=pass_epoch(model,loaders['validation'],device,
                progress_info={**info,'stage':'validation_batches'})
            history.append({'step':epoch,'epoch':epoch,'learning_rate':rates[-1],
                 'backbone_learning_rate':0 if mobile and epoch<=warmup else rates[0],
                 'phase':'frozen' if mobile and epoch<=warmup else 'finetune' if mobile else 'train',
                 'train_loss':train_loss,'validation_loss':loss,'train_accuracy':train_metrics['accuracy'],
                 'validation_accuracy':val['accuracy'],'train_macro_f1':train_metrics['macro_f1'],
                 'validation_macro_f1':val['macro_f1'],'smoke_test':smoke})
            progress(stage='epoch',model=architecture,epoch=epoch,max_epochs=epochs,
                     validation_macro_f1=val['macro_f1'],validation_loss=loss,
                     elapsed_seconds=time.monotonic()-epoch_started)
            print(f'{output_dir.name} epoch {epoch}/{epochs}: val F1={val["macro_f1"]:.5f}, loss={loss:.5f}',flush=True)
            if val['macro_f1']>best[0] or (val['macro_f1']==best[0] and loss<best[1]):
                best=(val['macro_f1'],loss);best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                config['best_epoch']=epoch;bad=0
            else:bad+=1
            if scheduler:scheduler.step(val['macro_f1'])
            if bad>=patience and (not mobile or epoch>warmup):break
        model.load_state_dict(best_state);model.eval()
        torch.save(best_state,output_dir/'model.pt')
        best_loss,validation,probabilities=pass_epoch(model,loaders['validation'],device)
    wall=time.perf_counter()-start
    write_json(output_dir/'model_config.json',config);write_json(output_dir/'class_mapping.json',MAPPING)
    hist=pd.DataFrame(history);hist['smoke_test']=smoke
    for col in reporting.HISTORY_COLUMNS:
        if col not in hist:hist[col]=np.nan
    hist.to_csv(output_dir/'history.csv',index=False)
    prediction_frame(splits['validation'],probabilities).to_csv(output_dir/'validation_predictions.csv',index=False)
    reporting.show_history(hist,output_dir/'history.png',smoke,kind)
    artifacts=reporting.MODEL_ARTIFACTS[kind]
    record={'model':kind,'config':config,'protocol':protocol,'validation_metrics':validation,
            'validation_loss':best_loss,'training_time_s':training_time,'fit_wall_time_s':wall,
            'environment':environment(device),'artifact_sha256':{f:sha256(output_dir/f) for f in artifacts},
            'validation_predictions_sha256':sha256(output_dir/'validation_predictions.csv'),
            'history_sha256':sha256(output_dir/'history.csv')}
    write_json(output_dir/'run.json',record)
    return record


def read_run(directory):
    directory=Path(directory)
    record=json.loads((directory/'run.json').read_text())
    for f,digest in {**record['artifact_sha256'],'validation_predictions.csv':record['validation_predictions_sha256'],
                     'history.csv':record['history_sha256']}.items():
        if sha256(directory/f)!=digest:raise ValueError(f'Run artifact hash mismatch: {f}')
    if json.loads((directory/'model_config.json').read_text())!=record['config']:
        raise ValueError('Run config mismatch')
    p=pd.read_csv(directory/'validation_predictions.csv')
    validate_predictions(p)
    if set(p.sample_id)!=set(record['protocol']['validation_sample_ids']):raise ValueError('Validation IDs mismatch')
    measured=reporting.classification_metrics(p.y_true,p[[f'prob_{i}' for i in range(6)]].to_numpy())
    for key in reporting.METRIC_COLUMNS:
        if not np.isclose(measured[key],record['validation_metrics'][key],rtol=1e-9,atol=1e-12):
            raise ValueError('Validation metric mismatch')
    return {**record,'directory':str(directory)}


def validate_predictions(predictions):
    if not predictions.sample_id.is_unique or not set(predictions.y_true).issubset(range(6)):
        raise ValueError('Invalid prediction labels/IDs')
    p=predictions[[f'prob_{i}' for i in range(6)]].to_numpy()
    if (not np.isfinite(p).all() or (p<0).any() or (p>1).any()
        or not np.allclose(p.sum(1),1,atol=1e-5) or not np.array_equal(p.argmax(1),predictions.y_pred)):
        raise ValueError('Invalid probabilities/predictions')


def protocol_identity(record):
    p=record['protocol']
    return tuple(json.dumps(p.get(k),sort_keys=True) for k in ['manifest_sha256','preprocessing_fingerprint',
                                    'smoke_test','seed','class_mapping','sample_counts','validation_sample_ids'])


def choose_cnn(records):
    if not records or any(r['model']!='cnn' for r in records):raise ValueError('Need CNN candidates')
    if len({protocol_identity(r) for r in records})!=1:raise ValueError('Mixed candidate protocol')
    return sorted(records,key=lambda r:(-r['validation_metrics']['macro_f1'],
                       float('inf') if r['validation_loss'] is None else r['validation_loss'],r['directory']))[0]


def require_cnn_candidates(records):
    if (len(records)!=3 or len({str(Path(r['directory']).resolve()) for r in records})!=3
            or any(r['model']!='cnn' for r in records)
            or {r['config']['architecture'] for r in records}!={'gap','spatial','mobilenet_v3_small'}):
        raise ValueError('Need exactly three distinct CNN candidates: gap, spatial, mobilenet_v3_small')
    mobile=next(r for r in records if r['config']['architecture']=='mobilenet_v3_small')
    if not mobile['protocol']['smoke_test']:
        weights=mobile['config'].get('pretrained',{})
        if weights.get('name')!='IMAGENET1K_V1' or weights.get('sha256')!='047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f':
            raise ValueError('MobileNet candidate lacks official pretrained provenance')


def predict_dataset(predictor,frame,data_root):
    if predictor.kind=='rf':return predictor.model.predict_proba(frame[LM_COLUMNS].to_numpy())
    if predictor.kind=='mlp':
        values=predictor.scaler.transform(frame[LM_COLUMNS].to_numpy()).astype(np.float32)
        dataset=TensorDataset(torch.from_numpy(values),torch.tensor(frame.class_id.to_numpy()))
    else:dataset=ROIImages(frame,data_root,predictor.architecture)
    return pass_epoch(predictor.model,DataLoader(dataset,batch_size=64),'cpu')[2]


def finalize(data_root,rf_dir,mlp_dir,cnn_dirs,output_root):
    records=[read_run(rf_dir),read_run(mlp_dir)]+[read_run(p) for p in cnn_dirs]
    require_cnn_candidates(records[2:])
    if len({protocol_identity(r) for r in records})!=1:raise ValueError('Mixed experiment protocol')
    if [r['model'] for r in records[:2]]!=['rf','mlp']:raise ValueError('Wrong RF/MLP runs')
    frame,preparation=load_dataset(data_root)
    reference=records[0]['protocol']
    if reference['manifest_sha256']!=sha256(Path(data_root)/'metadata/manifest.csv'):
        raise ValueError('Dataset manifest differs from training')
    if reference['preprocessing_fingerprint']!=preparation['preprocessing_fingerprint']:
        raise ValueError('Dataset preprocessing differs from training')
    # Validate validation IDs/truth against original dataset before selection.
    validation=frame.loc[frame.split.eq('validation')].set_index('sample_id')
    for r in records:
        p=pd.read_csv(Path(r['directory'])/'validation_predictions.csv').set_index('sample_id')
        if (not set(p.index).issubset(validation.index)
                or (not reference['smoke_test'] and set(p.index)!=set(validation.index))
                or not np.array_equal(p.y_true,validation.loc[p.index].class_id)):
            raise ValueError('Validation truth differs from dataset')
    chosen=choose_cnn(records[2:]);selected=[*records[:2],chosen]
    output_root=Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):raise ValueError('Final output exists; use a new experiment directory')
    output_root.mkdir(parents=True,exist_ok=True)
    write_json(output_root/'selection.json',{'criterion':'validation macro-F1, then loss; test not used',
           'cnn_candidates':[{'directory':r['directory'],'validation_macro_f1':r['validation_metrics']['macro_f1'],
                              'validation_loss':r['validation_loss'],'checkpoint_sha256':r['artifact_sha256']['model.pt']} for r in records[2:]],
           'selected_cnn':chosen['directory'],'protocol':reference,'smoke_test':reference['smoke_test']})
    test=frame.loc[frame.split.eq('test')].copy()
    if reference['smoke_test']:test=test.groupby('class_id',group_keys=False).head(4)
    protocol={**reference,'test_sample_ids':test.sample_id.tolist(),
              'sample_counts':{**reference['sample_counts'],'test':len(test)}}
    for r in selected:
        kind=r['model'];folder=output_root/kind;shutil.copytree(r['directory'],folder)
        predictor=GesturePredictor(folder)
        probabilities=predict_dataset(predictor,test,data_root)
        original_history=(folder/'history.csv').read_bytes()
        metrics,predictions=reporting.export_evaluation(folder,kind,test,probabilities,protocol,r['environment'],
                  r['training_time_s'],r['fit_wall_time_s'],r['validation_metrics'],r['config'])
        # Legacy plot/table generation standardizes the CSV. Preserve all measured
        # v2 columns (transfer phase and backbone LR) and the original run hash.
        (folder/'history.csv').write_bytes(original_history)
        metrics['validation_predictions_sha256']=sha256(folder/'validation_predictions.csv')
        metrics['preprocessing_fingerprint']=reference['preprocessing_fingerprint']
        write_json(folder/'metrics.json',metrics)
        reporting.show_errors(predictions,data_root,folder/'misclassified_gallery.png',smoke=reference['smoke_test'])
        # Three most common directed confusions, up to four examples per pair.
        errors=predictions.loc[predictions.y_true.ne(predictions.y_pred)]
        top=errors.groupby(['y_true','y_pred']).size().sort_values(ascending=False).head(3)
        for index,(pair,_) in enumerate(top.items(),1):
            examples=errors.loc[errors.y_true.eq(pair[0]) & errors.y_pred.eq(pair[1])].head(4)
            reporting.show_errors(examples,data_root,folder/f'confusion_pair_{index}.png',limit=4,smoke=reference['smoke_test'])
        reporting.write_model_report(folder,metrics,predictions)
        pairs=sorted(folder.glob('confusion_pair_*.png'))
        if pairs:
            with (folder/'report.md').open('a') as report:
                report.write('\n## Các cặp nhầm lẫn nhiều nhất\n\n')
                for path in pairs:report.write(f'[{path.stem}]({path.name})\n\n')
    results=verify_outputs(output_root,data_root)
    reporting.export_comparison(results,output_root/'comparison')
    return results


def verify_outputs(root,data_root=None,verify_derived=True):
    root=Path(root)
    results=reporting.load_comparison({n:root/n for n in ['rf','mlp','cnn']})
    dataset=None
    if data_root is not None:
        dataset,preparation=load_dataset(data_root)
        if results['rf']['metrics']['protocol']['manifest_sha256']!=sha256(Path(data_root)/'metadata/manifest.csv'):
            raise ValueError('Dataset manifest mismatch')
        if results['rf']['metrics']['protocol']['preprocessing_fingerprint']!=preparation['preprocessing_fingerprint']:
            raise ValueError('Dataset preprocessing fingerprint mismatch')
    for name,result in results.items():
        metrics=result['metrics'];folder=result['directory'];p=result['predictions']
        run=read_run(folder)
        if run['config']!=metrics['config']:raise ValueError('Final config differs from selected run')
        if (metrics['config']['preprocessing_fingerprint']!=metrics['protocol']['preprocessing_fingerprint']
                or metrics['protocol']['preprocessing_fingerprint']!=results['rf']['metrics']['protocol']['preprocessing_fingerprint']):
            raise ValueError('Mixed preprocessing fingerprint')
        reporting.validate_result_metrics(metrics,p)
        if verify_derived:
            expected_summary,expected_classes=reporting.evaluation_tables(metrics)
            for filename,expected in [('summary.csv',expected_summary),('per_class_metrics.csv',expected_classes)]:
                try:
                    pd.testing.assert_frame_equal(pd.read_csv(folder/filename),expected,check_dtype=False,
                                                  rtol=1e-9,atol=1e-12)
                except AssertionError as exc:raise ValueError(f'Derived output mismatch: {filename}') from exc
            cm=reporting.confusion_matrix(p.y_true,p.y_pred,labels=range(6))
            for filename,expected in [('confusion_matrix.csv',cm),
                                     ('confusion_matrix_normalized.csv',cm/np.maximum(cm.sum(1,keepdims=True),1))]:
                if not np.allclose(pd.read_csv(folder/filename,index_col=0).to_numpy(),expected,rtol=1e-9,atol=1e-12):
                    raise ValueError(f'Derived output mismatch: {filename}')
            for filename in ['report.md','history.png','confusion_matrix.png']:
                if not (folder/filename).is_file():raise ValueError(f'Missing output: {filename}')
        if sha256(folder/'validation_predictions.csv')!=metrics['validation_predictions_sha256']:
            raise ValueError('Validation prediction hash mismatch')
        val=pd.read_csv(folder/'validation_predictions.csv');validate_predictions(val)
        measured=reporting.classification_metrics(val.y_true,val[[f'prob_{i}' for i in range(6)]].to_numpy())
        for k in reporting.METRIC_COLUMNS:
            if not np.isclose(metrics['validation_metrics'][k],measured[k],rtol=1e-9,atol=1e-12):
                raise ValueError('Validation metric mismatch')
        if dataset is not None:
            for split,pred in [('test',p),('validation',val)]:
                truth=dataset.loc[dataset.split.eq(split)].set_index('sample_id')
                if (not set(pred.sample_id).issubset(truth.index)
                    or (not metrics['protocol']['smoke_test'] and set(pred.sample_id)!=set(truth.index))
                    or not np.array_equal(pred.y_true,truth.loc[pred.sample_id].class_id)):
                    raise ValueError('Prediction truth/IDs differ from dataset')
    return results


def regenerate_outputs(root,data_root):
    """Repair derived reports after validating all measured artifacts, without fitting."""
    results=verify_outputs(root,data_root,verify_derived=False)
    for result in results.values():
        folder=result['directory'];metrics=result['metrics'];predictions=result['predictions']
        history=(folder/'history.csv').read_bytes()
        try:
            reporting.show_errors(predictions,data_root,folder/'misclassified_gallery.png',
                                  smoke=metrics['protocol']['smoke_test'])
            reporting.write_model_outputs(folder,metrics,predictions)
            pairs=sorted(folder.glob('confusion_pair_*.png'))
            if pairs:
                with (folder/'report.md').open('a') as report:
                    report.write('\n## Các cặp nhầm lẫn nhiều nhất\n\n')
                    for path in pairs:report.write(f'[{path.stem}]({path.name})\n\n')
        finally:
            (folder/'history.csv').write_bytes(history)
    reporting.export_comparison(results,Path(root)/'comparison')
    return verify_outputs(root,data_root)
