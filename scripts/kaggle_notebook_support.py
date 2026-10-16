"""Source embedded into standalone notebooks by build_kaggle_notebooks.py.
No notebook imports this file at runtime.
"""
# %% common
from pathlib import Path
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib
from IPython import get_ipython
if get_ipython() is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from IPython.display import display, Markdown

GESTURES = ['fist', 'one', 'peace', 'three', 'four', 'palm']
CLASS_MAPPING = [{'class_id':i, 'gesture':name} for i,name in enumerate(GESTURES)]
LM_COLUMNS = [f'lm_{i}' for i in range(63)]
MODEL_ARTIFACTS = {'rf':['model.joblib','model_config.json','class_mapping.json'],
    'mlp':['model.pt','scaler.joblib','model_config.json','class_mapping.json'],
    'cnn':['model.pt','model_config.json','class_mapping.json']}
EXPECTED_SPLITS = {'train':27080, 'validation':5803, 'test':5804}
MODEL_ORDER = ['rf','mlp','cnn']
METRIC_COLUMNS = ['accuracy','macro_precision','macro_recall','macro_f1']
HISTORY_COLUMNS = ['step','epoch','learning_rate','train_loss','validation_loss',
                   'train_accuracy','validation_accuracy','train_macro_f1','validation_macro_f1','smoke_test']
PLOT_STYLE = {'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
              'axes.prop_cycle':plt.cycler(color=['#1675cb','#ed8733','#37a36b'])}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def find_data_root(explicit=None, search_root='/kaggle/input'):
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if not (root/'metadata/manifest.csv').is_file() or not (root/'images').is_dir():
            raise ValueError(f'DATA_ROOT không chứa metadata/manifest.csv và images/: {root}')
        return root
    candidates = sorted({p.parent.parent for p in Path(search_root).rglob('manifest.csv')
                         if p.parent.name == 'metadata' and (p.parent.parent/'images').is_dir()})
    if len(candidates) != 1:
        raise ValueError(f'Tìm thấy {len(candidates)} bộ dữ liệu. Đặt DATA_ROOT thủ công. Candidates: {candidates}')
    return candidates[0]


def validate_image(root, row):
    relative = Path(row.image_path)
    pieces = row.sample_id.split('/', 1)
    if len(pieces) != 2:
        raise ValueError(f'ID mẫu sai: {row.sample_id}')
    expected = f'images/{row.split}/{row.gesture}/{pieces[1]}'
    if relative.is_absolute() or '..' in relative.parts or row.image_path != expected:
        raise ValueError(f'Đường dẫn ảnh sai: {row.image_path}')
    path = Path(root)/relative
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != row.image_sha256:
        raise ValueError(f'SHA-256 ảnh không khớp manifest: {row.image_path}')


def load_dataset(root):
    root = Path(root)
    report = json.loads((root/'metadata/preparation_report.json').read_text())
    manifest_path = root/'metadata/manifest.csv'
    digest = file_sha256(manifest_path)
    if digest != report['artifact_sha256']['manifest.csv']:
        raise ValueError('Hash manifest không khớp preparation_report.json.')
    frame = pd.read_csv(manifest_path, float_precision='round_trip')
    required = ['sample_id','source_label','class_id','gesture','image_path','image_sha256','split',*LM_COLUMNS]
    if set(required)-set(frame.columns) or {col for col in frame.columns if col.startswith('lm_')} != set(LM_COLUMNS):
        raise ValueError('Manifest thiếu cột bắt buộc.')
    if frame[required].isna().any().any() or not np.isfinite(frame[LM_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError('Manifest có dữ liệu thiếu/không hữu hạn.')
    if not frame.sample_id.is_unique or not frame.image_sha256.is_unique or not frame.image_path.is_unique:
        raise ValueError('ID/hash/path ảnh trùng, có thể rò rỉ giữa các split.')
    if (set(frame['split']) != set(EXPECTED_SPLITS)
            or frame['split'].value_counts().to_dict() != report['split_counts']):
        raise ValueError('Split không khớp preparation_report.json.')
    for mapping in CLASS_MAPPING:
        class_id, gesture = mapping['class_id'], mapping['gesture']
        rows = frame.loc[frame.class_id.eq(class_id)]
        if rows.empty or not rows.gesture.eq(gesture).all() or not rows.source_label.eq('train_val_'+gesture).all():
            raise ValueError('Mapping lớp trong manifest không khớp 6 lớp đã chốt.')
        source = report['class_mapping'].get('train_val_'+gesture, {})
        if source.get('class_id') != class_id or source.get('gesture') != gesture:
            raise ValueError('Mapping trong báo cáo không khớp.')
    if set(frame.class_id) != set(range(6)):
        raise ValueError('Manifest có class ID ngoài 0..5.')
    for split in EXPECTED_SPLITS:
        if set(frame.loc[frame.split.eq(split),'class_id']) != set(range(6)):
            raise ValueError(f'Split {split} thiếu lớp.')
    for row in frame[['sample_id','split','gesture','image_path','image_sha256']].itertuples(index=False):
        validate_image(root, row)
    return frame.sort_values('sample_id').reset_index(drop=True), report


def smoke_subset(frame, per_class, seed=42):
    parts=[]
    for split in ['train','validation','test']:
        for class_id in range(6):
            rows=frame.loc[frame.split.eq(split)&frame.class_id.eq(class_id)].sort_values('sample_id')
            if len(rows)<per_class[split]:
                raise ValueError('Không đủ mẫu smoke test theo lớp/split.')
            parts.append(rows.sample(per_class[split], random_state=seed+class_id))
    return pd.concat(parts).sort_values('sample_id').reset_index(drop=True)


def fit_scaler(train_features):
    scaler=StandardScaler()
    transformed=scaler.fit_transform(train_features).astype(np.float32)
    return scaler, transformed


def classification_metrics(y_true, probabilities):
    predictions=np.asarray(probabilities).argmax(axis=1)
    precision,recall,f1,_=precision_recall_fscore_support(
        y_true,predictions,labels=range(6),average='macro',zero_division=0)
    return {'accuracy':float(accuracy_score(y_true,predictions)), 'macro_precision':float(precision),
            'macro_recall':float(recall),'macro_f1':float(f1),
            'classification_report':classification_report(y_true,predictions,labels=range(6),
                target_names=GESTURES,output_dict=True,zero_division=0)}


def save_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def environment_info(device='cpu'):
    cpu_name = platform.processor()
    cpu_file = Path('/proc/cpuinfo')
    if cpu_file.is_file():
        cpu_name = next((line.split(':',1)[1].strip() for line in cpu_file.read_text().splitlines() if line.startswith('model name')), 'unknown')
    info={'python':platform.python_version(),'platform':platform.platform(),'cpu':cpu_name,
          'cpu_count':os.cpu_count(),'device':str(device),
          'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','Pillow','joblib','matplotlib']}}
    return info


def protocol_info(frame, source_manifest, smoke, seed):
    return {'manifest_sha256':source_manifest,'class_mapping':CLASS_MAPPING,
            'smoke_test':bool(smoke),'seed':seed,
            'sample_counts':frame['split'].value_counts().to_dict(),
            'test_sample_ids':frame.loc[frame.split.eq('test'),'sample_id'].tolist(),
            'limits':['random split does not separate subjects','crop/landmark provenance unverified',
                      'paired classifier evaluation, not webcam end-to-end']}


def standardize_history(history, model_name, config, smoke=False):
    """Retain observed values; leave unavailable/non-applicable measurements blank."""
    history=history.copy()
    if 'step' not in history:history['step']=np.arange(1,len(history)+1)
    if model_name=='rf':
        for column in ['epoch','learning_rate','train_loss','validation_loss']:history[column]=np.nan
    elif 'learning_rate' not in history and config.get('scheduler') is None:
        history['learning_rate']=config.get('learning_rate',np.nan)
    if 'smoke_test' not in history:history['smoke_test']=bool(smoke)
    return history.reindex(columns=HISTORY_COLUMNS)


def evaluation_tables(metrics):
    summary=[];per_class=[]
    for split,values in [('validation',metrics['validation_metrics']),('test',metrics)]:
        summary.append({'model':metrics['model'],'split':split,
                        'sample_count':metrics['protocol']['sample_counts'][split],
                        **{key:values[key] for key in METRIC_COLUMNS}})
        for class_id,gesture in enumerate(GESTURES):
            row=values['classification_report'][gesture]
            per_class.append({'model':metrics['model'],'split':split,'class_id':class_id,'gesture':gesture,
                              'precision':row['precision'],'recall':row['recall'],
                              'f1_score':row['f1-score'],'support':int(row['support'])})
    return pd.DataFrame(summary),pd.DataFrame(per_class)


def validate_result_metrics(metrics,predictions):
    """Check stored test measurements against predictions without replacing them."""
    measured=classification_metrics(predictions.y_true,predictions[[f'prob_{i}' for i in range(6)]].to_numpy())
    for key in METRIC_COLUMNS:
        if not np.isclose(metrics[key],measured[key],rtol=1e-9,atol=1e-12):
            raise ValueError(f"{metrics['model']}: {key} khác CSV dự đoán.")
    for gesture in GESTURES:
        for key in ['precision','recall','f1-score','support']:
            if not np.isclose(metrics['classification_report'][gesture][key],
                              measured['classification_report'][gesture][key],rtol=1e-9,atol=1e-12):
                raise ValueError(f"{metrics['model']}: {gesture}/{key} khác CSV dự đoán.")
    for split,values in [('validation',metrics['validation_metrics']),('test',metrics)]:
        support=sum(values['classification_report'][g]['support'] for g in GESTURES)
        if support!=metrics['protocol']['sample_counts'][split]:
            raise ValueError(f"{metrics['model']}: support {split} không khớp protocol.")


def finish_figure(fig,path=None):
    # Fixed canvas: no tight bounding-box cropping, so matching chart types have identical sizes.
    if path:fig.savefig(path,dpi=130)
    if get_ipython() is not None and plt.get_backend().lower()!='agg':plt.show()
    plt.close(fig)


def show_history(history,path,smoke=False,model_name=None):
    with plt.rc_context(PLOT_STYLE):
        fig,axes=plt.subplots(1,4,figsize=(18,4),layout='constrained')
        if smoke:fig.suptitle('SMOKE TEST — chỉ kiểm tra pipeline')
        if model_name=='rf':
            for index,label in [(0,'Loss'),(3,'Learning rate')]:
                axes[index].axis('off');axes[index].text(.5,.5,label+'\nKhông áp dụng cho RF',ha='center',va='center')
            for ax,metric,label in zip(axes[1:3],['accuracy','macro_f1'],['Accuracy','Macro-F1']):
                ax.bar(['train','validation'],[history[f'{s}_{metric}'].iloc[0] for s in ['train','validation']],
                       color=['#1675cb','#ed8733'])
                ax.set(title=label,ylim=(0,1),xlabel='Một lượt fit')
        else:
            for ax,metric,label in zip(axes,['loss','accuracy','macro_f1'],['Loss','Accuracy','Macro-F1']):
                for split in ['train','validation']:
                    ax.plot(history.epoch,history[f'{split}_{metric}'],label=split)
                ax.set(title=label,xlabel='Epoch');ax.legend()
                if metric!='loss':ax.set_ylim(0,1)
            if 'learning_rate' in history and history.learning_rate.notna().any():
                axes[3].plot(history.epoch,history.learning_rate)
                axes[3].set(title='Learning rate',xlabel='Epoch',yscale='log')
            else:
                axes[3].axis('off');axes[3].text(.5,.5,'Learning rate chưa được ghi',ha='center',va='center')
        finish_figure(fig,path)


def markdown_table(frame):
    """Small dependency-free Markdown tables, with blanks rendered as N/A."""
    def cell(value):
        if pd.isna(value):return 'N/A'
        if isinstance(value,(float,np.floating)):return f'{value:.6f}'
        return str(value).replace('|','\\|').replace('\n',' ')
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines.extend('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)


def top_confusions(predictions,model_name,limit=3):
    cm=confusion_matrix(predictions.y_true,predictions.y_pred,labels=range(6))
    errors=sorted([(int(cm[i,j]),i,j) for i in range(6) for j in range(6) if i!=j and cm[i,j]>0],
                  key=lambda x:(-x[0],x[1],x[2]))[:limit]
    return [{'model':model_name,'true':GESTURES[i],'predicted':GESTURES[j],'count':count} for count,i,j in errors]


def write_model_report(output_dir,metrics,predictions):
    output_dir=Path(output_dir);name=metrics['model'];config=metrics['config'];env=metrics['environment']
    summary,_=evaluation_tables(metrics)
    confusions=pd.DataFrame(top_confusions(predictions,name),columns=['model','true','predicted','count'])
    lines=[f'# Kết quả {name.upper()}','']
    if metrics['protocol']['smoke_test']:lines+=['**SMOKE TEST — chỉ kiểm tra pipeline, không dùng để chọn model.**','']
    lines+=['## Cấu hình','',f"Phiên: `{config.get('run_name',name)}`; seed: {metrics['protocol']['seed']}.",
            f"Kiến trúc: {config.get('architecture','Random Forest')}. ",
            f"Preprocessing: {config.get('preprocessing','Không có thông tin')}."]
    if name=='cnn' and config.get('run_name')=='cnn_tuned':lines+=['CNN trong báo cáo là phiên tinh chỉnh `cnn_tuned`.']
    if name=='rf':
        lines += [f"Số cây: {config.get('n_estimators','N/A')}; max depth: {config.get('max_depth','N/A')}. "]
    else:
        lines += [f"Optimizer: {config.get('optimizer','N/A')}; learning rate ban đầu: {config.get('learning_rate','N/A')}; "
                  f"weight decay: {config.get('weight_decay','N/A')}; batch size: {config.get('batch_size','N/A')}.",
                  f"Epoch tối đa: {config.get('max_epochs','N/A')}; patience: {config.get('patience','N/A')}; "
                  f"checkpoint được chọn tại epoch: {config.get('best_epoch','N/A')}.",
                  'Scheduler: `'+json.dumps(config.get('scheduler'),ensure_ascii=False)+'`.']
    lines+=['','[Cấu hình đầy đủ](model_config.json)','', '## Kết quả','',markdown_table(summary),'',
            '[Metric từng lớp](per_class_metrics.csv) · [Metric gốc](metrics.json) · [Dự đoán test](test_predictions.csv)',
            '', '## Phân tích nhầm lẫn','',markdown_table(confusions) if len(confusions) else 'Không có dự đoán sai.',
            '', '## Thời gian và thiết bị','',
            f"Training: {metrics['training_time_s']:.3f} s; fit wall time: {metrics['fit_wall_time_s']:.3f} s.",
            f"Device: {env.get('device')}; CPU: {env.get('cpu')}; GPU: {env.get('gpu') or 'N/A'}.",
            'RF đo estimator.fit; mạng đo train batches gồm loader, không gồm validation. '
            'Fit wall time của mạng gồm validation và chọn checkpoint. '
            'Không so sánh tốc độ train công bằng khi phần cứng khác nhau.',
            '', '## Biểu đồ','']
    for filename,label in [('history.png','Lịch sử huấn luyện'),('confusion_matrix.png','Confusion matrix'),
                           ('misclassified_gallery.png','Gallery dự đoán sai')]:
        if (output_dir/filename).is_file():lines.append(f'[{label}]({filename})')
    lines+=['','[Dữ liệu lịch sử](history.csv)','',
            'RF chỉ có một lượt fit; epoch/loss/learning rate không áp dụng. '
            'Train metrics của MLP/CNN được đo trong batch có dropout/augmentation; validation/test ở eval. '
            'Ô trống là số đo không áp dụng hoặc chưa có, không phải 0. '
            'Test được đánh giá sau khi model/checkpoint cố định. '
            'Split chưa tách người chụp; kết quả không đại diện FPS hoặc webcam end-to-end.']
    (output_dir/'report.md').write_text('\n'.join(lines)+'\n')


def write_model_outputs(output_dir,metrics,predictions):
    """Derive comparable tables and plots; never rewrite measured metrics or model artifacts."""
    output_dir=Path(output_dir)
    validate_result_metrics(metrics,predictions)
    summary,per_class=evaluation_tables(metrics)
    history=standardize_history(pd.read_csv(output_dir/'history.csv',float_precision='round_trip'),
                                metrics['model'],metrics['config'],metrics['protocol']['smoke_test'])
    history.to_csv(output_dir/'history.csv',index=False)
    summary.to_csv(output_dir/'summary.csv',index=False)
    per_class.to_csv(output_dir/'per_class_metrics.csv',index=False)
    cm=confusion_matrix(predictions.y_true,predictions.y_pred,labels=range(6))
    pd.DataFrame(cm,index=GESTURES,columns=GESTURES).to_csv(output_dir/'confusion_matrix.csv')
    pd.DataFrame(cm/np.maximum(cm.sum(axis=1,keepdims=True),1),index=GESTURES,columns=GESTURES).to_csv(output_dir/'confusion_matrix_normalized.csv')
    show_confusion(cm,output_dir/'confusion_matrix.png',smoke=metrics['protocol']['smoke_test'])
    show_history(history,output_dir/'history.png',smoke=metrics['protocol']['smoke_test'],model_name=metrics['model'])
    write_model_report(output_dir,metrics,predictions)


def export_evaluation(output_dir, model_name, frame, probabilities, protocol, env,
                      training_time, fit_wall_time, validation_metrics, config):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    metrics=classification_metrics(frame.class_id.to_numpy(),probabilities)
    artifact_names=MODEL_ARTIFACTS[model_name]
    metrics.update({'model':model_name,'protocol':protocol,'environment':env,
                    'training_time_s':float(training_time),'fit_wall_time_s':float(fit_wall_time),
                    'training_time_scope':'RF estimator.fit; neural training batches including loader, excluding validation',
                    'validation_metrics':validation_metrics,'config':config,
                    'artifact_sha256':{name:file_sha256(output_dir/name) for name in artifact_names},
                    'artifact_bytes':sum((output_dir/name).stat().st_size for name in artifact_names if (output_dir/name).is_file()),
                    'target_accuracy':.92,'target_met':metrics['accuracy']>=.92})
    result=frame[['sample_id','class_id','image_path']].rename(columns={'class_id':'y_true'}).copy()
    result['smoke_test']=protocol['smoke_test']
    result['y_pred']=np.asarray(probabilities).argmax(axis=1)
    for class_id in range(6):result[f'prob_{class_id}']=probabilities[:,class_id]
    result.to_csv(output_dir/'test_predictions.csv',index=False)
    metrics['test_predictions_sha256']=file_sha256(output_dir/'test_predictions.csv')
    save_json(output_dir/'metrics.json',metrics)
    write_model_outputs(output_dir,metrics,result)
    summary,per_class=evaluation_tables(metrics)
    display(summary.round(4))
    display(per_class.loc[per_class.split.eq('test')].round(4))
    display(Markdown(f"**{'SMOKE TEST — chưa phải huấn luyện đầy đủ. ' if protocol['smoke_test'] else ''}"
                     f"Test Accuracy: {metrics['accuracy']:.2%}; macro-F1: {metrics['macro_f1']:.4f}. "
                     f"{'Đạt' if metrics['target_met'] else 'Chưa đạt'} mục tiêu 92%.**"))
    return metrics,result


def show_confusion(cm, path=None, smoke=False):
    with plt.rc_context(PLOT_STYLE):
        return _show_confusion(cm,path,smoke)


def _show_confusion(cm, path=None, smoke=False):
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    if smoke:fig.suptitle('SMOKE TEST — chỉ kiểm tra pipeline')
    normalized=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
    for ax,values,title in zip(axes,[cm,normalized],['Số lượng','Tỷ lệ theo nhãn thật']):
        vmax=1 if values is normalized else max(int(cm.sum(axis=1).max()),1)
        im=ax.imshow(values,cmap='Blues',vmin=0,vmax=vmax)
        ax.set_xticks(range(6),GESTURES,rotation=35,ha='right');ax.set_yticks(range(6),GESTURES)
        ax.set(title=title,xlabel='Dự đoán',ylabel='Nhãn thật')
        for i in range(6):
            for j in range(6):
                label=f'{values[i,j]:.2f}' if values is normalized else str(int(values[i,j]))
                ax.text(j,i,label,ha='center',va='center',color='white' if values[i,j]>vmax/2 else 'black')
        fig.colorbar(im,ax=ax,shrink=.8)
    finish_figure(fig,path)


def show_errors(predictions,data_root,path,limit=12,smoke=False):
    with plt.rc_context(PLOT_STYLE):
        return _show_errors(predictions,data_root,path,limit,smoke)


def _show_errors(predictions,data_root,path,limit=12,smoke=False):
    errors=(predictions.loc[predictions.y_true.ne(predictions.y_pred)].sort_values('sample_id')
            .groupby('y_true',group_keys=False).head(2).head(limit))
    if errors.empty:
        Path(path).unlink(missing_ok=True)
        print('Không có dự đoán sai trong tập được đánh giá.');return
    rows=max(3,int(np.ceil(limit/4)))
    fig,axes=plt.subplots(rows,4,figsize=(12,3*rows),squeeze=False,layout='constrained')
    if smoke:fig.suptitle('SMOKE TEST — chỉ kiểm tra pipeline')
    for ax in axes.flat:ax.axis('off')
    for ax,row in zip(axes.flat,errors.itertuples()):
        with Image.open(Path(data_root)/row.image_path) as image:
            ax.imshow(ImageOps.exif_transpose(image).convert('RGB'))
        ax.set_title(f'Thật: {GESTURES[row.y_true]}\nDự đoán: {GESTURES[row.y_pred]}',fontsize=10)
    finish_figure(fig,path)

# %% torch
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def select_device(requested='auto',cuda_available=None):
    available=torch.cuda.is_available() if cuda_available is None else cuda_available
    if requested=='auto':return torch.device('cuda' if available else 'cpu')
    if requested=='cuda' and not available:raise ValueError('Đã yêu cầu CUDA nhưng GPU không khả dụng.')
    if requested not in {'cpu','cuda'}:raise ValueError('DEVICE phải là auto/cpu/cuda.')
    return torch.device(requested)


class GestureMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(63,128),nn.ReLU(),nn.Dropout(.2),
                                  nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,6))
    def forward(self,x):return self.network(x)


class GestureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[];channels=3
        for output in [32,64,128]:
            layers.extend([nn.Conv2d(channels,output,3,padding=1),nn.BatchNorm2d(output),nn.ReLU(),nn.MaxPool2d(2)])
            channels=output
        self.features=nn.Sequential(*layers)
        self.classifier=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.2),nn.Linear(128,6))
    def forward(self,x):return self.classifier(self.features(x))


def image_tensor(image,training=False,image_size=128):
    image=ImageOps.exif_transpose(image).convert('RGB').resize((image_size,image_size),Image.Resampling.BILINEAR)
    if training:
        image=image.rotate(random.uniform(-10,10),resample=Image.Resampling.BILINEAR)
        image=ImageEnhance.Brightness(image).enhance(random.uniform(.85,1.15))
        image=ImageEnhance.Contrast(image).enhance(random.uniform(.85,1.15))
    array=np.array(image,dtype=np.float32,copy=True)/255.
    return torch.from_numpy(array.transpose(2,0,1).copy())


class GestureImages(Dataset):
    def __init__(self,frame,root,training=False,image_size=128):
        self.frame=frame.reset_index(drop=True);self.root=Path(root)
        self.training=training;self.image_size=image_size
    def __len__(self):return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        with Image.open(self.root/row.image_path) as image:
            tensor=image_tensor(image,self.training,self.image_size)
        return tensor,int(row.class_id)


def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed)


def make_loader(dataset,batch_size,training,seed,workers=0):
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(dataset,batch_size=batch_size,shuffle=training,num_workers=workers,
                      worker_init_fn=seed_worker,generator=generator,drop_last=False)


def checkpoint_is_better(f1,loss,best_f1,best_loss):
    return f1>best_f1 or (f1==best_f1 and loss<best_loss)


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def neural_pass(model,loader,device,optimizer=None):
    training=optimizer is not None;model.train(training)
    total_loss=0.;probabilities=[];labels=[]
    with torch.set_grad_enabled(training):
        for features,target in loader:
            features=features.to(device);target=target.to(device)
            if training:optimizer.zero_grad(set_to_none=True)
            logits=model(features);loss=nn.functional.cross_entropy(logits,target)
            if training:loss.backward();optimizer.step()
            total_loss+=float(loss.detach().cpu())*len(target)
            probabilities.append(logits.detach().softmax(dim=1).cpu().numpy());labels.append(target.cpu().numpy())
    values=np.concatenate(probabilities);truth=np.concatenate(labels)
    metrics=classification_metrics(truth,values)
    return total_loss/len(truth),metrics,values


def fit_neural(model,train_loader,validation_loader,device,epochs,patience,output_dir,
               optimizer_config=None,scheduler_config=None):
    model=model.to(device)
    optimizer_options={'lr':1e-3,'weight_decay':1e-4}
    if optimizer_config is not None:optimizer_options.update(optimizer_config)
    optimizer=torch.optim.AdamW(model.parameters(),**optimizer_options)
    scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,**scheduler_config)
               if scheduler_config is not None else None)
    history=[];best_f1=-1.;best_loss=float('inf');best_state=None;bad_epochs=0;best_epoch=None
    training_time=0.;synchronize(device);wall_start=time.perf_counter()
    for epoch in range(1,epochs+1):
        learning_rate=optimizer.param_groups[0]['lr']
        synchronize(device);start=time.perf_counter()
        train_loss,train_metrics,_=neural_pass(model,train_loader,device,optimizer)
        synchronize(device);training_time+=time.perf_counter()-start
        validation_loss,validation_metrics,_=neural_pass(model,validation_loader,device)
        f1=validation_metrics['macro_f1']
        history.append({'epoch':epoch,'learning_rate':learning_rate,'train_loss':train_loss,'validation_loss':validation_loss,
                        'train_accuracy':train_metrics['accuracy'],'validation_accuracy':validation_metrics['accuracy'],
                        'train_macro_f1':train_metrics['macro_f1'],'validation_macro_f1':f1})
        print(f'Epoch {epoch:02d}: lr={learning_rate:.2e}, train loss={train_loss:.4f}, validation loss={validation_loss:.4f}, macro-F1={f1:.4f}')
        if checkpoint_is_better(f1,validation_loss,best_f1,best_loss):
            best_f1,best_loss,best_epoch=f1,validation_loss,epoch
            best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            bad_epochs=0
        else:bad_epochs+=1
        # Step after validation: the resulting LR is used in the next epoch.
        if scheduler is not None:scheduler.step(f1)
        if bad_epochs>=patience:
            print('Early stopping.');break
    synchronize(device);wall_time=time.perf_counter()-wall_start
    model.load_state_dict(best_state);model.eval()
    torch.save(best_state,Path(output_dir)/'model.pt')
    pd.DataFrame(history).to_csv(Path(output_dir)/'history.csv',index=False)
    return model,pd.DataFrame(history),training_time,wall_time,best_epoch


# %% comparison

def resolve_comparison_paths(results_root=None,result_dirs=None,output_root=None,start=None,
                             kaggle_input='/kaggle/input',kaggle_output='/kaggle/working/hand_gesture'):
    """Resolve canonical local artifacts first, then attached Kaggle outputs."""
    start=Path(start or Path.cwd()).expanduser().resolve()
    local_root=next((p/'results' for p in [start,*start.parents] if (p/'results').is_dir()),None)
    if results_root is not None:local_root=Path(results_root).expanduser().resolve()
    kaggle_input=Path(kaggle_input).expanduser().resolve()
    if result_dirs is not None:
        dirs={name:Path(folder).expanduser().resolve() for name,folder in result_dirs.items()}
    elif local_root is not None:
        dirs={name:local_root/name for name in MODEL_ORDER}
    elif kaggle_input.is_dir():
        dirs=find_result_dirs(search_root=kaggle_input)
    else:
        raise ValueError('Không tìm thấy results/ local hoặc input Kaggle. Đặt RESULTS_ROOT hoặc RESULT_DIRS.')
    if output_root is not None:
        output=Path(output_root).expanduser().resolve()
    elif dirs and all(folder.is_relative_to(kaggle_input) for folder in dirs.values()):
        output=Path(kaggle_output).expanduser().resolve()
    else:
        output=local_root if local_root is not None else start/'results'
    return dirs,output,local_root


def find_comparison_data_root(explicit=None,start=None,search_root='/kaggle/input'):
    """Called only when benchmarking: dataset is optional for saved-metric comparison."""
    if explicit is not None:return find_data_root(explicit)
    start=Path(start or Path.cwd()).expanduser().resolve()
    for parent in [start,*start.parents]:
        candidate=parent/'data'
        if (candidate/'metadata/manifest.csv').is_file() and (candidate/'images').is_dir():return candidate
    try:
        return find_data_root(search_root=search_root)
    except ValueError as exc:
        raise ValueError('RUN_BENCHMARK=True cần dataset gốc. Đặt DATA_ROOT tới thư mục chứa metadata/ và images/.') from exc


def run_comparison(result_dirs,output_dir,run_benchmark=False,data_root=None,seed=42,
                   warmup=20,sample_count=200,data_start=None):
    """Validate saved predictions and optionally benchmark against the original dataset."""
    output_dir=Path(output_dir).expanduser().resolve()
    print('Input artifacts:')
    for name,folder in result_dirs.items():print(f'  {name}: {Path(folder).resolve()}')
    print('Output:',output_dir)
    print('RUN_BENCHMARK:',bool(run_benchmark))
    if output_dir in {Path(p).resolve() for p in result_dirs.values()}:
        raise ValueError('Thư mục comparison phải khác thư mục artifact của model.')
    results=load_comparison(result_dirs)
    reference=results['rf']['metrics']['protocol']
    if reference.get('seed')!=seed:
        raise ValueError(f'Seed/protocol không khớp SEED={seed}.')
    for result in results.values():validate_result_metrics(result['metrics'],result['predictions'])
    latency=None;benchmark_protocol=None
    if run_benchmark:
        from threadpoolctl import threadpool_limits
        root=find_comparison_data_root(data_root,start=data_start)
        print('Benchmark dataset:',root)
        full_frame,_=load_dataset(root)
        if reference['manifest_sha256']!=file_sha256(root/'metadata/manifest.csv'):
            raise ValueError('Output không thuộc manifest đang dùng cho benchmark.')
        test_frame=full_frame.loc[full_frame.split.eq('test')].set_index('sample_id')
        test_ids=reference['test_sample_ids']
        if not set(test_ids).issubset(test_frame.index) or (not reference['smoke_test'] and set(test_ids)!=set(test_frame.index)):
            raise ValueError('Output thiếu ID test hoặc có ID không thuộc test.')
        test_frame=test_frame.loc[test_ids].reset_index()
        for result in results.values():
            truth=result['predictions'].set_index('sample_id').y_true
            if not np.array_equal(truth.loc[test_frame.sample_id].to_numpy(),test_frame.class_id.to_numpy()):
                raise ValueError('Nhãn thật output khác manifest.')
        with threadpool_limits(limits=1):
            latency=benchmark_models(results,test_frame,root,seed,warmup,sample_count)
        benchmark_protocol={'environment':environment_info('cpu'),'torch_version':torch.__version__,
                            'protocol':reference,'batch_size':1,'threads':1,'warmup':warmup,
                            'requested_samples':sample_count,
                            'test_ids':test_frame.sort_values('sample_id').sample(min(sample_count,len(test_frame)),random_state=seed).sample_id.tolist(),
                            'scope':'classifier only; MLP includes scaler; CNN precomputed image tensor; no IO/crop/MediaPipe'}
    table=export_comparison(results,output_dir,latency=latency)
    if benchmark_protocol is not None:save_json(output_dir/'benchmark_protocol.json',benchmark_protocol)
    else:(output_dir/'benchmark_protocol.json').unlink(missing_ok=True)
    return table


def load_comparison(result_dirs):
    if set(result_dirs)!={'rf','mlp','cnn'}:raise ValueError('Cần đủ rf, mlp, cnn.')
    results={};reference=None
    for name,folder in result_dirs.items():
        folder=Path(folder)
        metrics=json.loads((folder/'metrics.json').read_text())
        hashes=metrics.get('artifact_sha256', {})
        for artifact in MODEL_ARTIFACTS[name]:
            if artifact not in hashes or not (folder/artifact).is_file() or file_sha256(folder/artifact)!=hashes[artifact]:
                raise ValueError(f'Hash artifact không khớp/thiếu: {name}/{artifact}')
        if file_sha256(folder/'test_predictions.csv')!=metrics.get('test_predictions_sha256'):
            raise ValueError(f'Hash prediction CSV không khớp: {name}')
        prediction=pd.read_csv(folder/'test_predictions.csv')
        required=['sample_id','y_true','y_pred',*[f'prob_{i}' for i in range(6)]]
        if set(required)-set(prediction.columns) or prediction[required].isna().any().any():
            raise ValueError(f'Predictions thiếu/sai: {name}')
        if metrics['model']!=name or not prediction.sample_id.is_unique:
            raise ValueError('Model/ID không nhất quán.')
        probability=prediction[[f'prob_{i}' for i in range(6)]].to_numpy()
        if (not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any()
                or not np.allclose(probability.sum(axis=1),1,atol=1e-5)
                or not np.array_equal(probability.argmax(axis=1),prediction.y_pred.to_numpy())):
            raise ValueError('Xác suất/dự đoán không hợp lệ.')
        protocol=metrics['protocol']
        if protocol['class_mapping'] != CLASS_MAPPING or not set(prediction.y_true).issubset(range(6)):
            raise ValueError('Mapping hoặc nhãn thật không hợp lệ.')
        if len(protocol['test_sample_ids']) != len(set(protocol['test_sample_ids'])):
            raise ValueError('Protocol có ID test trùng.')
        if set(prediction.sample_id)!=set(protocol['test_sample_ids']):raise ValueError('ID test không khớp protocol.')
        identity=(protocol['manifest_sha256'],protocol['class_mapping'],protocol['smoke_test'],protocol.get('seed'),
                  protocol.get('sample_counts'),
                  prediction.sort_values('sample_id')[['sample_id','y_true']].to_dict('records'))
        if reference is None:reference=identity
        elif identity!=reference:raise ValueError('Khác mapping, manifest, chế độ smoke hoặc ID/nhãn test; không so sánh.')
        results[name]={'directory':folder,'metrics':metrics,'predictions':prediction}
    return results


def find_result_dirs(explicit=None,search_root='/kaggle/input'):
    if explicit is not None:return {key:Path(value) for key,value in explicit.items()}
    found={name:[] for name in ['rf','mlp','cnn']}
    for path in Path(search_root).rglob('metrics.json'):
        if 'archive' in path.relative_to(search_root).parts:continue
        try:
            obj=json.loads(path.read_text());name=obj.get('model')
            if name in found and (path.parent/'test_predictions.csv').is_file():found[name].append(path.parent)
        except (OSError,ValueError):continue
    if any(len(paths)!=1 for paths in found.values()):
        raise ValueError(f'Cần một output mỗi model; đặt RESULT_DIRS thủ công: {found}')
    return {name:paths[0] for name,paths in found.items()}


def export_comparison(results,output_dir,latency=None):
    """Summarize verified measurements; latency is included only when actually benchmarked."""
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    rows=[];classes=[];confusions=[]
    for name in MODEL_ORDER:
        result=results[name];metrics=result['metrics'];predictions=result['predictions']
        validate_result_metrics(metrics,predictions)
        _,per_class=evaluation_tables(metrics);classes.append(per_class)
        env=metrics['environment'];folder=result['directory']
        rows.append({'model':name,'run_name':metrics['config'].get('run_name',name),
                     'smoke_test':metrics['protocol']['smoke_test'],
                     **{key:metrics[key] for key in METRIC_COLUMNS},
                     **{f'validation_{key}':metrics['validation_metrics'][key] for key in METRIC_COLUMNS},
                     'test_samples':metrics['protocol']['sample_counts']['test'],
                     'training_time_s':metrics['training_time_s'],'fit_wall_time_s':metrics['fit_wall_time_s'],
                     'training_device':env.get('device'),'training_cpu':env.get('cpu'),'training_gpu':env.get('gpu'),
                     'artifact_bytes':sum((folder/f).stat().st_size for f in MODEL_ARTIFACTS[name])})
        confusions.extend(top_confusions(predictions,name))
    table=pd.DataFrame(rows);per_class=pd.concat(classes,ignore_index=True)
    if latency is not None:table=table.merge(latency,on='model',validate='one_to_one')
    table.to_csv(output_dir/'comparison.csv',index=False)
    per_class.to_csv(output_dir/'per_class_metrics.csv',index=False)
    confusion_table=pd.DataFrame(confusions,columns=['model','true','predicted','count'])
    confusion_table.to_csv(output_dir/'top_confusions.csv',index=False)
    smoke=bool(table.smoke_test.iloc[0])
    with plt.rc_context(PLOT_STYLE):
        fig,axes=plt.subplots(1,3,figsize=(18,4),layout='constrained')
        if smoke:fig.suptitle('SMOKE TEST — chỉ kiểm tra pipeline')
        x=np.arange(len(MODEL_ORDER))
        for ax,metric,label in zip(axes[:2],['accuracy','macro_f1'],['Accuracy','Macro-F1']):
            for offset,split in [(-.18,'validation'),(.18,'test')]:
                values=table[f'validation_{metric}'] if split=='validation' else table[metric]
                bars=ax.bar(x+offset,values,width=.36,label=split,
                            color='#ed8733' if split=='validation' else '#37a36b')
                ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=2)
            ax.set(xticks=x,xticklabels=[n.upper() for n in MODEL_ORDER],ylim=(0,1.1),title=label);ax.legend()
        for index,name in enumerate(MODEL_ORDER):
            values=per_class.loc[per_class.model.eq(name)&per_class.split.eq('test'),'f1_score']
            axes[2].plot(range(6),values,marker='o',label=name.upper())
        axes[2].set(xticks=range(6),xticklabels=GESTURES,ylim=(0,1.05),title='Test F1 theo lớp');axes[2].legend()
        finish_figure(fig,output_dir/'comparison.png')
        fig,axes=plt.subplots(1,3,figsize=(18,5),layout='constrained')
        if smoke:fig.suptitle('SMOKE TEST — chỉ kiểm tra pipeline')
        for ax,name in zip(axes,MODEL_ORDER):
            p=results[name]['predictions'];cm=confusion_matrix(p.y_true,p.y_pred,labels=range(6))
            normalized=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
            im=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
            ax.set(title=name.upper(),xticks=range(6),xticklabels=GESTURES,
                   yticks=range(6),yticklabels=GESTURES,xlabel='Dự đoán',ylabel='Nhãn thật')
            for i in range(6):
                for j in range(6):
                    ax.text(j,i,f'{normalized[i,j]:.2f}',ha='center',va='center',
                            color='white' if normalized[i,j]>.5 else 'black',fontsize=9)
        fig.colorbar(im,ax=list(axes),shrink=.8)
        finish_figure(fig,output_dir/'confusion_matrices.png')
    report=['# So sánh RF, MLP và CNN','']
    if smoke:report+=['**SMOKE TEST — không dùng để xếp hạng hoặc chọn model.**','']
    report+=['Cùng manifest, seed, mapping và tập test. CNN hiện tại là phiên `'+str(table.loc[table.model.eq('cnn'),'run_name'].iloc[0])+'`.',
             '', '## Chất lượng','',markdown_table(table[['model','validation_accuracy','validation_macro_f1',
                'accuracy','macro_f1','test_samples']]),'',
             '[Bảng tổng hợp đầy đủ](comparison.csv) · [Metric từng lớp](per_class_metrics.csv)',
             '', '## Thời gian và thiết bị','',markdown_table(table[['model','training_time_s','fit_wall_time_s',
                'training_device','training_cpu','training_gpu']]),'',
             'Thời gian train mô tả từng phiên; phần cứng khác nhau nên không dùng làm kết luận tốc độ công bằng. '
             'RF/MLP dùng landmark, CNN dùng ảnh. Không tính detector, crop hoặc pipeline webcam.',
             '', '## Phân tích lỗi','',markdown_table(confusion_table),'',
             '[Biểu đồ chất lượng](comparison.png) · [Confusion matrix cùng thang màu](confusion_matrices.png)',
             '', '## Báo cáo từng model','']
    for name in MODEL_ORDER:
        folder=results[name]['directory']
        has_report=(folder/'report.md').is_file()
        target=os.path.relpath(folder/('report.md' if has_report else 'metrics.json'),output_dir)
        label=name.upper() if has_report else name.upper()+' — metric gốc (output cũ chưa có report.md)'
        report.append(f'[{label}]({target})')
    if latency is not None:
        report+=['','## Benchmark CPU','',markdown_table(table[['model','cpu_latency_median_ms','cpu_latency_p95_ms']]),
                 '', '[Protocol benchmark](benchmark_protocol.json)',
                 'Đây là classifier latency đã đo, không phải FPS toàn hệ thống.']
    else:report+=['','Chưa chạy benchmark latency trong báo cáo này; không suy ra latency hoặc FPS.']
    (output_dir/'report.md').write_text('\n'.join(report)+'\n')
    return table


def load_saved_model(name,folder):
    folder=Path(folder);config=json.loads((folder/'model_config.json').read_text())
    if config.get('model')!=name or config.get('class_mapping')!=CLASS_MAPPING:
        raise ValueError('Config model/mapping không khớp baseline.')
    if name in {'rf','mlp'} and config.get('features')!=LM_COLUMNS:
        raise ValueError('Thứ tự features không khớp baseline.')
    if name=='mlp' and (config.get('architecture')!=[63,128,64,6] or config.get('activation')!='ReLU' or config.get('dropout')!=.2
            or config.get('preprocessing')!='StandardScaler fit chỉ train; scaler.joblib'):
        raise ValueError('Kiến trúc/preprocessing MLP không khớp baseline.')
    if name=='cnn' and (config.get('architecture')!='3 x Conv3x3(padding1)-BN-ReLU-MaxPool2, channels32/64/128; GAP-Dropout-Linear6'
            or config.get('dropout')!=.2 or config.get('image_size')!=128
            or config.get('preprocessing')!='EXIF orientation, RGB, bilinear resize128x128, pixels/255'):
        raise ValueError('Kiến trúc/preprocessing CNN không khớp baseline.')
    if name=='rf':
        model=joblib.load(folder/'model.joblib')
        for parameter in ['n_estimators','max_depth','min_samples_leaf','max_features']:
            if model.get_params()[parameter]!=config.get(parameter):raise ValueError('RF params khác config.')
        model.set_params(n_jobs=1)
        return model,None,config
    model=GestureMLP() if name=='mlp' else GestureCNN()
    model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True));model.eval()
    scaler=joblib.load(folder/'scaler.joblib') if name=='mlp' else None
    return model,scaler,config


def benchmark_models(results,test_frame,data_root,seed=42,warmup=20,sample_count=200):
    torch.set_num_threads(1)
    ordered=test_frame.sort_values('sample_id')
    benchmark=ordered.sample(min(sample_count,len(ordered)),random_state=seed).reset_index(drop=True)
    features=benchmark[LM_COLUMNS].to_numpy(dtype=np.float64)
    tensors=[]
    for path in benchmark.image_path:
        with Image.open(Path(data_root)/path) as image:tensors.append(image_tensor(image,False,128).unsqueeze(0))
    rows=[]
    for name,result in results.items():
        model,scaler,config=load_saved_model(name,result['directory'])
        if name=='cnn' and config.get('image_size')!=128:raise ValueError('Benchmark CNN yêu cầu preprocessing128 như baseline.')
        def predict(index):
            if name=='rf':return model.predict_proba(features[index:index+1])
            with torch.inference_mode():
                if name=='mlp':
                    values=scaler.transform(features[index:index+1]).astype(np.float32)
                    return model(torch.from_numpy(values)).softmax(dim=1)
                return model(tensors[index]).softmax(dim=1)
        for i in range(warmup):predict(i%len(benchmark))
        timings=[]
        for i in range(len(benchmark)):
            start=time.perf_counter_ns();predict(i);timings.append((time.perf_counter_ns()-start)/1e6)
        rows.append({'model':name,'cpu_latency_median_ms':float(np.median(timings)),
                     'cpu_latency_p95_ms':float(np.percentile(timings,95)),'benchmark_samples':len(benchmark),
                     'batch_size':1,'threads':1,'warmup':warmup})
    return pd.DataFrame(rows)
