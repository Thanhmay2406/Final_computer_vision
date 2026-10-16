"""Measured CPU classifier/video benchmark, separate from training-device times."""
from pathlib import Path
import json
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits
import torch

from handgesture.data import load_dataset,LM_COLUMNS,write_json
from handgesture.preprocessing import HandPreprocessor,PreprocessingConfig,sha256
from handgesture.models import GesturePredictor
from handgesture.training import verify_outputs,environment
from handgesture.live import run_stream
from scripts import kaggle_notebook_support as reporting


def select_deployment(rows,validation_scores,target_fps=15,min_hand_fraction=.8):
    candidates=[r for r in rows if r['complete'] and r['hand_frames']>0
                and r['hand_frames']/max(r.get('measured_frames',r['hand_frames']),1)>=min_hand_fraction
                and r['fps']>=target_fps and r.get('hand_positive_fps',r['fps'])>=target_fps]
    candidates.sort(key=lambda r:(-validation_scores[r['model']],r['median_ms'],r['model']))
    return {'model':candidates[0]['model'] if candidates else None,'target_fps':target_fps,'min_hand_fraction':min_hand_fraction,
            'target_met':bool(candidates),'criterion':'completed end-to-end FPS >= target with detected hands; highest validation macro-F1, then latency'}


def validate_deployment(deployment,folder):
    folder=Path(folder)
    config=json.loads((folder/'model_config.json').read_text())
    if (not deployment.get('target_met') or deployment.get('model')!=config['model']
            or deployment.get('preprocessing_fingerprint')!=config['preprocessing_fingerprint']
            or not deployment.get('artifact_sha256')):
        raise ValueError('Deployment benchmark does not match checkpoint')
    for name,digest in deployment['artifact_sha256'].items():
        if sha256(folder/name)!=digest:raise ValueError('Checkpoint differs from benchmark artifacts')


def benchmark_classifiers(results_root,data_root,output_dir,warmup=20,sample_count=200):
    results=verify_outputs(results_root,data_root)
    frame,report=load_dataset(data_root)
    ids=results['rf']['metrics']['protocol']['test_sample_ids']
    test=frame.set_index('sample_id').loc[ids].reset_index().sort_values('sample_id')
    test=test.sample(min(sample_count,len(test)),random_state=42)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    rows=[]
    with threadpool_limits(limits=1):
        for name in ['rf','mlp','cnn']:
            predictor=GesturePredictor(Path(results_root)/name)
            inputs=[]
            for row in test.itertuples(index=False):
                features=np.array([getattr(row,col) for col in LM_COLUMNS])
                if name=='cnn':
                    with Image.open(Path(data_root)/row.image_path) as im:roi=np.array(im.convert('RGB'))
                else:roi=None
                inputs.append(predictor.prepare_input(features,roi))
            for i in range(warmup):predictor.predict_input(inputs[i%len(inputs)])
            timings=[]
            for value in inputs:
                start=time.perf_counter_ns();predictor.predict_input(value)
                timings.append((time.perf_counter_ns()-start)/1e6)
            rows.append({'model':name,'cpu_latency_median_ms':float(np.median(timings)),
                   'cpu_latency_p95_ms':float(np.percentile(timings,95)),
                   'benchmark_samples':len(inputs),'warmup':warmup,'threads':1,'batch_size':1})
    table=pd.DataFrame(rows);table.to_csv(output_dir/'classifier_latency.csv',index=False)
    protocol={'environment':environment('cpu'),'scope':'classifier only; MLP includes scaler; CNN tensor preparation excluded',
              'warmup':warmup,'sample_ids':test.sample_id.tolist(),'threads':1,'batch_size':1,
              'manifest_sha256':sha256(Path(data_root)/'metadata/manifest.csv'),
              'preprocessing_fingerprint':report['preprocessing_fingerprint'],
              'smoke_test':results['rf']['metrics']['protocol']['smoke_test']}
    write_json(output_dir/'classifier_protocol.json',protocol)
    comparison=output_dir/'comparison';reporting.export_comparison(results,comparison,table)
    write_json(comparison/'benchmark_protocol.json',protocol)
    return table


def benchmark_video(results_root,video,detector_path,output_dir,display=False,warmup=30,frames=300):
    results=verify_outputs(results_root)
    if not Path(video).is_file():raise ValueError('Benchmark requires the same existing video for all models')
    output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):raise ValueError('Benchmark output exists; choose a new directory')
    torch.set_num_threads(1);rows=[]
    with threadpool_limits(limits=1):
        for name in ['rf','mlp','cnn']:
            config=PreprocessingConfig(**results[name]['metrics']['config']['preprocessing'])
            predictor=GesturePredictor(Path(results_root)/name,config)
            processor=HandPreprocessor(config,detector_path,'VIDEO')
            capture=cv2.VideoCapture(str(video))
            if not capture.isOpened():
                capture.release();processor.close();raise ValueError('Cannot open benchmark video')
            row=run_stream(capture,processor,predictor,output_dir/name,warmup,frames,display)
            rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(output_dir/'end_to_end.csv',index=False)
    smoke=any(r['metrics']['protocol']['smoke_test'] for r in results.values())
    deployment=select_deployment(rows,{n:r['metrics']['validation_metrics']['macro_f1'] for n,r in results.items()})
    if smoke:
        deployment.update(model=None,target_met=False,reason='Smoke output cannot select a deployment model')
    if deployment['model'] is not None:
        record=results[deployment['model']]['metrics']
        deployment.update(artifact_sha256=record['artifact_sha256'],
                          preprocessing_fingerprint=record['protocol']['preprocessing_fingerprint'])
    write_json(output_dir/'deployment.json',deployment)
    protocol={'environment':environment('cpu'),'video_sha256':sha256(video),'warmup':warmup,
              'requested_frames':frames,'display':display,'smoke_test':smoke,
              'preprocessing_fingerprint':config.fingerprint,
              'models':{n:r['metrics']['artifact_sha256'] for n,r in results.items()},
              'scope':'same decoded video; not a live camera accuracy measurement'}
    write_json(output_dir/'video_protocol.json',protocol)
    report=['# End-to-end CPU benchmark','',reporting.markdown_table(table[['model','fps','median_ms','p95_ms','measured_frames','hand_frames','complete']]),
            '', 'Target FPS: 15. Selected model: '+str(deployment['model']),
            '', 'No model met the selection criteria.' if deployment['model'] is None else 'Selection uses validation quality among eligible models.',
            '', 'Video benchmark and live webcam measurements are distinct. No webcam accuracy is inferred.']
    (output_dir/'report.md').write_text('\n'.join(report)+'\n')
    return table
