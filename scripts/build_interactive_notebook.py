#!/usr/bin/env python3
"""Generate a single native-kernel, manually controlled Kaggle experiment."""
from pathlib import Path
import sys
import nbformat as nbf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path(__file__).resolve().parents[1]


def build_interactive():
    from scripts.build_v2_notebooks import SOURCES, RESOLVE, bootstrap, write
    cells=[]
    def markdown(text):cells.append(nbf.v4.new_markdown_cell(text))
    def code(step,source):
        cell=nbf.v4.new_code_cell(source);cell.metadata['step']=step;cells.append(cell)
    markdown('''# Thực nghiệm cử chỉ tay — chạy từng cell trên Kaggle

**Bắt đầu:** Import notebook này vào Kaggle, bật GPU, gắn hai Input `thanhmay2406/hand-gesture` và `thanhmay2406/hand-gesture-v2-runtime-assets-20261015`. Chạy từ trên xuống bằng **Shift+Enter**. Nhìn dấu `[*]`, log và thời gian dưới từng cell; dùng nút Stop/Interrupt của notebook để dừng.

Mặc định là **SMOKE**, tối đa 8 ảnh/lớp/split và 2 epoch/model mạng; RF dùng 10 cây. Đây chỉ là kiểm tra kỹ thuật. Muốn full: đổi `SMOKE_TEST=False` và chọn `RUN_ID` mới. Không sửa split hay preprocessing khi so sánh model. Test chỉ chạy khi bật cờ riêng sau khi xem validation.

Code huấn luyện chạy ngay trong kernel notebook. Không có script nền, watcher hoặc tiến trình con điều khiển pipeline. Python 3.11/3.12 là điều kiện của môi trường MediaPipe0.10.21 đã chọn; cell môi trường sẽ báo rõ nếu image hiện tại không phù hợp. Không tự đổi Python: nếu image không tương thích, dùng hai cell setup/chuyển kernel có cờ xác nhận bên dưới. Không dùng detector khác.''')
    code('configuration','''from pathlib import Path
import os, sys, json, time, importlib.metadata

SMOKE_TEST = True
RUN_ID = 'manual_001'  # Đổi khi thử cấu hình khác; không ghi đè checkpoint.
EVALUATE_TEST = False  # Chỉ bật sau khi kiểm tra cell validation.
INSTALL_DEPENDENCIES = False  # Bật một lần nếu cell dependency báo thiếu/sai phiên bản.
REQUIRE_CUDA = True
PREPARE_COMPATIBLE_KERNEL = False  # Bật nếu image Kaggle đang dùng Python 3.13+.
SWITCH_COMPATIBLE_KERNEL = False   # Bật sau khi setup báo NATIVE_READY.
NATIVE_RUNTIME = Path('/tmp/handgesture-native-python311')
NUM_WORKERS = 0
SEED = 42
DATA_ROOT = None      # Tự tìm manifest từ Kaggle Input; điền rõ nếu có nhiều dataset.
DETECTOR_PATH = None
WEIGHTS_PATH = None
OUTPUT_ROOT = Path('/kaggle/working') / ('handgesture_' + ('smoke_' if SMOKE_TEST else 'full_') + RUN_ID)
DEVICE = 'cuda' if REQUIRE_CUDA else 'cpu'

print('Python:',sys.version)
print('Chế độ:', 'SMOKE — không dùng để kết luận chất lượng' if SMOKE_TEST else 'FULL')
print('Output:',OUTPUT_ROOT)
print('Test được bật:',EVALUATE_TEST)
if NUM_WORKERS != 0:raise ValueError('Notebook này yêu cầu NUM_WORKERS=0')
if not RUN_ID or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in RUN_ID):
    raise ValueError('RUN_ID chỉ gồm chữ, số, dấu _ hoặc -')
''')
    markdown('''## 1. Dependency — kiểm tra trước khi import thư viện

Cell này giữ cặp Torch/Torchvision CUDA có sẵn của Kaggle. Nếu cần cài gói, bật `INSTALL_DEPENDENCIES=True` ở trên và chạy cell này. Sau khi cài: đổi lại `False`, **Restart Session/kernel**, rồi chạy từ đầu. Cài dependency là thao tác duy nhất dùng `%pip`; toàn bộ bước dữ liệu/huấn luyện chạy trực tiếp.

Nếu Python ngoài 3.11/3.12, dùng hai cell setup/chuyển kernel bên dưới hoặc chọn image/kernel phù hợp nếu Kaggle cung cấp. Không chạy tiếp bằng môi trường khác vì fingerprint/checkpoint phải khớp detector đã thống nhất.''')
    requirements=[line.strip() for line in (ROOT/'requirements/kaggle.txt').read_text().splitlines()
                  if line.strip() and not line.startswith('#')]
    markdown('''### Image Kaggle dùng Python 3.13+: setup kernel tương thích (tùy chọn)

Nếu cell cấu hình hiển thị Python3.13+, bật `PREPARE_COMPATIBLE_KERNEL=True` rồi chạy cell setup dưới. Nó chỉ cài Python3.11, dependency và CUDA Torch vào `/tmp`; log cài đặt hiện ngay dưới cell. Khi có `NATIVE_READY`, bật `SWITCH_COMPATIBLE_KERNEL=True` và chạy cell chuyển kernel riêng. Sau khi kết nối lại, chạy lại từ cell cấu hình; kernel mới là Python3.11 nên các bước setup tự bỏ qua. Các cell train vẫn chạy trực tiếp trong kernel này.

Không chọn Restart Session thông thường ngay sau chuyển kernel: Kaggle có thể tạo lại Python3.13 từ image gốc. Khi session mới/máy mới, thực hiện lại hai bước setup/chuyển nếu cần. Cờ setup và chuyển mặc định tắt; không tự chạy khi import notebook.''')
    code('kernel_setup',(ROOT/'handgesture/kernel_bootstrap.py').read_text()+f'''
NATIVE_REQUIREMENTS = {requirements!r}
if sys.version_info[:2] in {{(3,11),(3,12)}}:
    print('Kernel đã tương thích:',sys.version)
elif PREPARE_COMPATIBLE_KERNEL:
    prepare_native_kernel(NATIVE_RUNTIME,NATIVE_REQUIREMENTS,get_ipython())
else:
    print('Python chưa tương thích. Bật PREPARE_COMPATIBLE_KERNEL=True trong cell cấu hình để chuẩn bị môi trường; chưa chạy training.')
''')
    code('kernel_switch','''if sys.version_info[:2] in {(3,11),(3,12)}:
    print('Đang dùng kernel tương thích:',sys.version)
elif SWITCH_COMPATIBLE_KERNEL:
    switch_native_kernel(NATIVE_RUNTIME)
else:
    print('Chỉ chuyển sau khi setup báo NATIVE_READY; bật SWITCH_COMPATIBLE_KERNEL=True trong cell cấu hình.')
''')
    code('dependencies',f'''PINNED_REQUIREMENTS = {requirements!r}
if sys.version_info[:2] not in {{(3,11),(3,12)}}:
    raise RuntimeError(f'Kernel đang dùng Python {{sys.version_info.major}}.{{sys.version_info.minor}}; cần Python 3.11/3.12 cho môi trường MediaPipe0.10.21. Chạy cell setup/chuyển kernel tương thích ở trên; không khởi chạy pipeline nền thay thế.')
if INSTALL_DEPENDENCIES:
    get_ipython().run_line_magic('pip', 'install ' + ' '.join(PINNED_REQUIREMENTS))
    raise RuntimeError('Đã chạy lệnh cài. Đặt INSTALL_DEPENDENCIES=False, restart kernel rồi chạy lại từ đầu để kiểm tra phiên bản thực tế.')
package_rows=[]
for requirement in PINNED_REQUIREMENTS:
    name,expected=requirement.split('==')
    try:actual=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:actual='missing'
    package_rows.append({{'package':name,'expected':expected,'actual':actual,'ok':actual==expected}})
for row in package_rows:print(row)
if not all(row['ok'] for row in package_rows):
    raise RuntimeError('Dependency chưa khớp. Bật INSTALL_DEPENDENCIES=True, cài rồi restart kernel; không tiếp tục huấn luyện trong kernel cũ.')
print('Dependency đã khớp; có thể tiếp tục.')
''')
    markdown('''## 2. Nạp source dùng chung

Các cell thư viện bên dưới được sinh trực tiếp từ source repo, mặc định thu gọn để dễ xem các bước thực nghiệm. Có thể mở rộng để kiểm tra. Chúng chỉ định nghĩa module trong kernel; không chạy huấn luyện hoặc tạo tiến trình nền.''')
    shared=bootstrap()
    for index,cell in enumerate(shared):
        cell.metadata.update(jupyter={'source_hidden':True},collapsed=True,tags=['shared-source'])
        if index:cell.metadata['source_module']=SOURCES[index-1][0]
        cells.append(cell)
    code('environment','''import numpy as np
import pandas as pd
import torch, torchvision, mediapipe
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Image as DisplayImage, FileLink
from handgesture.runtime import stage, require_cuda
from handgesture.data import prepare_dataset, load_dataset, GESTURES
from handgesture.training import train_run, read_run, finalize, verify_outputs, environment
from handgesture.preprocessing import sha256

torch.set_num_threads(2)
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
os.environ['HG_PROGRESS_PATH']=str(OUTPUT_ROOT/'progress.json')
os.environ['HG_EVENT_LOG_PATH']=str(OUTPUT_ROOT/'execution.log')
os.environ['HG_EVENT_LOG_OWNER_PID']=str(os.getpid())
if Path('/kaggle').exists():
    os.environ['HG_KAGGLE']='1'
    if not REQUIRE_CUDA:raise ValueError('Bật REQUIRE_CUDA=True cho thực nghiệm Kaggle')
if REQUIRE_CUDA:
    os.environ['HG_REQUIRE_CUDA']='1'
    with stage('gpu_check'):require_cuda()
else:
    os.environ.pop('HG_REQUIRE_CUDA',None)
with stage('environment'):
    env=environment(DEVICE)
    (OUTPUT_ROOT/'environment.json').write_text(json.dumps(env,indent=2)+'\\n')
    display(pd.DataFrame([{'package':k,'version':v} for k,v in env['versions'].items()]))
print('GPU:',env.get('gpu'))
print('Kernel PID:',os.getpid(),'— mọi bước train chạy trong kernel này')
''')
    markdown('''## 3. Input và ảnh gốc

Kiểm tra đúng dataset/model asset. Bảng dùng số mẫu thật trong manifest; ảnh xem trước chỉ lấy từ train. Nếu không tìm được duy nhất một Input, đặt đường dẫn rõ tại cell cấu hình.''')
    code('inputs',RESOLVE+'''
print('Đang tìm metadata/model Input, bỏ qua cây ảnh...',flush=True)
source=Path(DATA_ROOT) if DATA_ROOT else find_dataset(False)
detector=Path(DETECTOR_PATH) if DETECTOR_PATH else find_asset('hand_landmarker.task')
weights=Path(WEIGHTS_PATH) if WEIGHTS_PATH else find_asset('mobilenet_v3_small-047dcff4.pth')
if sha256(weights)!='047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f':
    if not SMOKE_TEST:raise ValueError('Weights MobileNet không đúng IMAGENET1K_V1')
    print('SMOKE: weights fixture không dùng để đánh giá chất lượng')
source_frame=pd.read_csv(source/'metadata/manifest.csv')
print('Dataset:',source,'\\nDetector:',detector,'\\nWeights:',weights)
display(source_frame.groupby(['split','gesture']).size().unstack(fill_value=0))
def show_train_images(frame,root,title):
    rows=frame.loc[frame.split.eq('train')].groupby('class_id',sort=True).head(1)
    fig,axes=plt.subplots(1,6,figsize=(15,3))
    for axis in axes:axis.axis('off')
    for axis,row in zip(axes,rows.itertuples()):
        with Image.open(Path(root)/row.image_path) as image:axis.imshow(image.convert('RGB'))
        axis.set_title(row.gesture)
    fig.suptitle(title);plt.tight_layout();plt.show();plt.close(fig)
show_train_images(source_frame,source,'Ảnh nguồn — train')
''')
    markdown('''## 4. Tiền xử lý

Theo dõi số mẫu ngay dưới cell này. Sau khi xong, kiểm tra số mẫu trước/sau, lý do loại và ROI. Split/ID được giữ nguyên. Mẫu không phát hiện tay bị loại khỏi cả ba nhánh. Nếu một split thiếu lớp, pipeline dừng; không tự chia lại tập hoặc tự tăng số mẫu smoke.''')
    code('preprocessing','''data=OUTPUT_ROOT/'data_v2'
with stage('preprocessing'):
    frame,preparation=prepare_dataset(source,data,detector,
        limit=8 if SMOKE_TEST else None,workers=NUM_WORKERS)
display(pd.DataFrame([preparation['counts']]))
display(frame.groupby(['split','gesture']).size().unstack(fill_value=0))
display(pd.DataFrame(list(preparation['rejection_reasons'].items()),columns=['reason','count']))
RUN_CONTEXT=workflow_configuration()
RUN_DATA_MANIFEST=sha256(data/'metadata/manifest.csv')
print('Fingerprint:',preparation['preprocessing_fingerprint'])
show_train_images(frame,data,'ROI sau preprocessing — train')
''')
    markdown('''## 5. Huấn luyện từng model — chỉ train/validation

Chạy từng cell, kiểm tra metric và history trước khi chuyển cell kế tiếp. Full giữ protocol trong source; smoke dùng 10 cây RF và 2 epoch/model mạng. Cấu hình thực tế nằm trong `model_config.json` và được hiển thị dưới cell. Cell đã hoàn tất không được ghi đè: muốn train lại, chọn `RUN_ID` mới. Khi lỗi/Interrupt giữa chừng, output dở được giữ để chẩn đoán.''')
    code('display_helpers','''def workflow_configuration():
    return {'output_root':str(OUTPUT_ROOT.resolve()),'smoke_test':bool(SMOKE_TEST),
            'seed':SEED,'workers':NUM_WORKERS,'device':DEVICE,
            'source':str(DATA_ROOT),'detector':str(DETECTOR_PATH),'weights':str(WEIGHTS_PATH)}

def require_current_run():
    if globals().get('RUN_CONTEXT')!=workflow_configuration():
        raise ValueError('Đã đổi cấu hình phiên. Chạy lại từ bước Input và preprocessing với RUN_ID mới; chỉ EVALUATE_TEST được đổi sau validation.')
    if data.resolve()!=(OUTPUT_ROOT/'data_v2').resolve():
        raise ValueError('Đường dẫn dữ liệu không thuộc cấu hình hiện tại')
    if sha256(data/'metadata/manifest.csv')!=RUN_DATA_MANIFEST:
        raise ValueError('Manifest đã thay đổi sau preprocessing')

def show_run(directory):
    record=read_run(directory)
    display(pd.DataFrame([{'model':record['config']['architecture'],
        'validation_accuracy':record['validation_metrics']['accuracy'],
        'validation_macro_f1':record['validation_metrics']['macro_f1'],
        'validation_loss':record['validation_loss'],'seconds':record['fit_wall_time_s'],
        'smoke_test':record['protocol']['smoke_test']}]))
    display(pd.json_normalize(record['config']))
    image=Path(directory)/'history.png'
    if image.exists():display(DisplayImage(filename=str(image)))
    return record
''')
    for kind,architecture,title in [('rf','rf','Random Forest'),('mlp','mlp','MLP'),
                                   ('cnn','gap','CNN GAP'),('cnn','spatial','CNN spatial'),
                                   ('cnn','mobilenet_v3_small','MobileNetV3-Small pretrained')]:
        markdown('### '+title)
        directory=f"OUTPUT_ROOT/'runs'/{kind!r}" if kind!='cnn' else f"OUTPUT_ROOT/'candidates'/{architecture!r}"
        code(architecture,f'''require_current_run()
out={directory}
with stage('training',model={architecture!r}):
    train_run(data,out,{kind!r},architecture={architecture!r},weights_path=weights if {kind!r}=='cnn' else None,
              smoke=SMOKE_TEST,device=DEVICE,workers=NUM_WORKERS,seed=SEED)
show_run(out)
''')
    markdown('''## 6. So sánh validation và chốt CNN

Chưa đánh giá test. Chọn CNN theo validation macro-F1, phá hòa bằng validation loss. Kết quả smoke chỉ giúp kiểm tra code, không dùng công bố model cuối cùng.''')
    code('validation','''require_current_run()
rf_dir=OUTPUT_ROOT/'runs/rf'
mlp_dir=OUTPUT_ROOT/'runs/mlp'
cnn_dirs=[OUTPUT_ROOT/'candidates'/a for a in ['gap','spatial','mobilenet_v3_small']]
records=[read_run(p) for p in [rf_dir,mlp_dir,*cnn_dirs]]
validation_table=pd.DataFrame([{'model':r['config']['architecture'],
    'macro_f1':r['validation_metrics']['macro_f1'],'accuracy':r['validation_metrics']['accuracy'],
    'loss':r['validation_loss'],'seconds':r['fit_wall_time_s'],'smoke_test':r['protocol']['smoke_test']} for r in records])
display(validation_table.sort_values('macro_f1',ascending=False))
best_cnn=min(records[2:],key=lambda r:(-r['validation_metrics']['macro_f1'],r['validation_loss']))
print('CNN theo validation:',best_cnn['config']['architecture'])
print('CHỈ SMOKE' if SMOKE_TEST else 'FULL — cần thêm benchmark CPU/video trước khi chọn model webcam')
''')
    markdown('''## 7. Đánh giá test — bước xác nhận riêng

Dừng tại đây để xem validation. Khi đã chốt protocol và đủ năm lượt train, đổi `EVALUATE_TEST=True` trong cell cấu hình (giữ cùng `RUN_ID`), chạy lại cell cấu hình rồi chạy cell này. Không cần chạy lại các cell train. Không dùng test để điều chỉnh ứng viên. Cell này không ghi đè final output đã có.''')
    code('test_evaluation','''if not EVALUATE_TEST:
    print('Chưa chạy test. Xem validation rồi đặt EVALUATE_TEST=True ở cell cấu hình nếu muốn đánh giá.')
else:
    require_current_run()
    rf_dir=OUTPUT_ROOT/'runs/rf'
    mlp_dir=OUTPUT_ROOT/'runs/mlp'
    cnn_dirs=[OUTPUT_ROOT/'candidates'/a for a in ['gap','spatial','mobilenet_v3_small']]
    final_dir=OUTPUT_ROOT/'final'
    with stage('test_evaluation'):
        results=finalize(data,rf_dir,mlp_dir,cnn_dirs,final_dir)
        verify_outputs(final_dir,data)
    display(pd.read_csv(final_dir/'comparison/comparison.csv'))
    for name in ['rf','mlp','cnn']:
        print(name.upper())
        for filename in ['confusion_matrix.png','per_class_metrics.png','misclassified_gallery.png']:
            image=final_dir/name/filename
            if image.exists():display(DisplayImage(filename=str(image)))
''')
    markdown('''## 8. Kiểm chứng và tải kết quả

Chỉ xuất sau khi final đã hoàn tất và kiểm chứng thành công. Hai ZIP tách dữ liệu dẫn xuất và kết quả, kèm SHA-256. Tải trong mục Output của Kaggle. Smoke được đánh dấu riêng trong artifact; không trộn với baseline hoặc full.''')
    code('export','''if not EVALUATE_TEST:
    print('Chưa xuất final: EVALUATE_TEST=False. Checkpoint train/validation vẫn ở',OUTPUT_ROOT)
else:
    require_current_run()
    import zipfile, hashlib
    with stage('verify_export'):
        verified=verify_outputs(OUTPUT_ROOT/'final',data)
        verified_smoke=verified['cnn']['metrics']['protocol']['smoke_test']
        export_dir=OUTPUT_ROOT/'exports';export_dir.mkdir(exist_ok=True)
        archives={}
        for name,folder in [('data_v2.zip',data),('results_v2.zip',OUTPUT_ROOT)]:
            target=export_dir/name
            with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
                for path in sorted(folder.rglob('*')):
                    relative=path.relative_to(folder)
                    if path.is_file() and not (name=='results_v2.zip' and relative.parts[0] in {'exports','data_v2'}):
                        archive.write(path,str(relative))
            with target.open('rb') as stream:archives[name]=hashlib.file_digest(stream,'sha256').hexdigest()
            print(name,target.stat().st_size,'bytes',archives[name])
        (export_dir/'export_manifest.json').write_text(json.dumps({'archives':archives,'smoke_test':verified_smoke},indent=2)+'\\n')
    print('Tải hai ZIP và export_manifest.json từ Output:',export_dir)
    for path in sorted(export_dir.iterdir()):display(FileLink(str(path)))
''')
    markdown('''## Theo dõi hoặc xử lý lỗi

- Cell có `[*]` đang chạy; log batch/epoch nằm ngay dưới cell model. Cell lỗi giữ traceback tại đúng bước.
- Có thể chạy `print((OUTPUT_ROOT/'progress.json').read_text())` trong cell mới để xem milestone cuối. Đây là tiến độ nội bộ, không phải chỉ trạng thái RUNNING của Kaggle.
- Sau Interrupt, không tự tiếp tục bằng checkpoint dở. Chọn `RUN_ID` mới để train lại; preprocessing có journal riêng hỗ trợ tiếp tục cùng fingerprint.
- Notebook này không đặt watchdog nền/deadline của wrapper API. Quản lý thời lượng qua session Kaggle và Stop/Interrupt.
- Sau khi tải artifact: kiểm chứng lại trên local, benchmark classifier CPU và video/webcam theo protocol trước khi công bố model triển khai.
''')
    # Helpers must be defined before preprocessing binds the immutable run context.
    helper=next(c for c in cells if c.metadata.get('step')=='display_helpers')
    cells.remove(helper)
    index=next(i for i,c in enumerate(cells) if c.metadata.get('step')=='preprocessing')
    cells.insert(index,helper)
    write('07_interactive_workflow.ipynb',cells)


if __name__=='__main__':
    build_interactive();print('Built native interactive Kaggle notebook')
