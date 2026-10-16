"""Generate standalone Kaggle notebooks; runtime does not need project source."""
from pathlib import Path
import nbformat as nbf
import textwrap

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT/'scripts/kaggle_notebook_support.py').read_text()
common, rest = source.split('# %% common\n',1)[1].split('# %% torch\n',1)
torch_code, comparison = rest.split('# %% comparison\n',1)


def markdown(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text.strip())
def write(name,cells):
    notebook=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.11'}})
    nbf.validate(notebook)
    target=ROOT/'notebooks/kaggle'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,target)

INTRO='''Bộ 6 lớp: `fist/one/peace/three/four/palm → 0/1/2/3/4/5`.
Gắn dataset **thanhmay2406/hand-gesture** bằng Add Input. Notebook tự tìm dữ liệu;
đặt `DATA_ROOT` nếu có nhiều bộ phù hợp. Output nằm trong `/kaggle/working/hand_gesture/`.

Giữ split đã cố định **27.080 / 5.803 / 5.804**, seed 42. Không chia lại, không cân bằng lại lớp.
Đánh giá các mẫu có cả ảnh và landmark hợp lệ; split chưa tách người chụp, công thức
chuẩn hóa landmark/crop chưa được xác minh. Accuracy này không đại diện cho webcam end-to-end.
Không cam kết baseline đạt 92%. `SMOKE_TEST=True` chỉ kiểm tra code với tập nhỏ, không dùng làm kết quả thực nghiệm.'''
config='''from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
DATA_ROOT = None  # ví dụ: '/kaggle/input/hand-gesture'
OUTPUT_ROOT = Path('/kaggle/working/hand_gesture')
SEED = 42
SMOKE_TEST = False
SMOKE_PER_CLASS = {'train': 8, 'validation': 4, 'test': 4}
DEVICE = 'auto'  # auto/cpu/cuda; RF luôn chạy CPU
NUM_WORKERS = 2
BATCH_SIZE = {batch}
MAX_EPOCHS = {epochs}
PATIENCE = 8
RUN_NAME = None  # Tên phiên trong metadata; thư mục output luôn theo MODEL_NAME
'''
setup='''root = find_data_root(DATA_ROOT)
full_frame, preparation_report = load_dataset(root)
if SEED != 42 or SEED != preparation_report['config']['seed']:
    raise ValueError('Baseline dùng seed 42, phải khớp báo cáo dữ liệu.')
frame = smoke_subset(full_frame, SMOKE_PER_CLASS, SEED) if SMOKE_TEST else full_frame
splits = {name: frame.loc[frame.split.eq(name)].reset_index(drop=True) for name in EXPECTED_SPLITS}
protocol = protocol_info(frame, file_sha256(root/'metadata/manifest.csv'), SMOKE_TEST, SEED)
output_dir = OUTPUT_ROOT / MODEL_NAME
output_dir.mkdir(parents=True, exist_ok=True)
save_json(output_dir/'class_mapping.json', CLASS_MAPPING)
display(frame.groupby(['split','gesture']).size().unstack().reindex(columns=GESTURES))
print('Manifest SHA-256:', protocol['manifest_sha256'])
print('SMOKE TEST' if SMOKE_TEST else 'HUẤN LUYỆN ĐẦY ĐỦ', '— output:', output_dir)
'''
end='''metrics, predictions = export_evaluation(
    output_dir, MODEL_NAME, splits['test'], test_probabilities, protocol, env,
    training_time, fit_wall_time, validation_metrics, model_config)
show_errors(predictions, root, output_dir/'misclassified_gallery.png',smoke=SMOKE_TEST)
write_model_report(output_dir, metrics, predictions)
'''
ending=markdown('''### Cách đọc kết quả và sử dụng model
Confusion matrix hàng là nhãn thật, cột là nhãn dự đoán; tỷ lệ chuẩn hóa theo hàng cho thấy
lớp nào bị nhầm nhiều. Quan sát thêm cặp `four`–`palm`, macro-F1 và gallery lỗi.
Test chỉ được dùng sau khi mô hình/checkpoint đã cố định. Không thay cấu hình dựa trên test.

Lưu **Save Version → Save & Run All** để giữ artifact rồi gắn output vào notebook 04.
`model_config.json` ghi kiến trúc và preprocessing; `class_mapping.json` ghi nhãn.
RF dùng `joblib.load`; MLP/CNN khởi tạo đúng lớp trong notebook rồi tải `state_dict`
bằng `torch.load(..., map_location='cpu', weights_only=True)`, đặt `eval()`.
MLP phải tải `scaler.joblib`; CNN suy luận bằng `image_tensor(image, training=False)`.
Thời gian huấn luyện có kèm thiết bị; chưa đo FPS hay MediaPipe/crop/đọc ảnh end-to-end.''')

rf_train='''random.seed(SEED); np.random.seed(SEED)
env = environment_info('cpu')
display(env)
model_config = {'model':'rf','seed':SEED,'smoke_test':SMOKE_TEST,
    'n_estimators':10 if SMOKE_TEST else 300, 'max_depth':20,
    'min_samples_leaf':2, 'max_features':'sqrt','n_jobs':-1,
    'preprocessing':'63 landmark hiện có, không chuẩn hóa lại','features':LM_COLUMNS,
    'class_mapping':CLASS_MAPPING}
model = RandomForestClassifier(**{key:model_config[key] for key in
    ['n_estimators','max_depth','min_samples_leaf','max_features','n_jobs']}, random_state=SEED)
start = time.perf_counter()
model.fit(splits['train'][LM_COLUMNS].to_numpy(), splits['train'].class_id.to_numpy())
training_time = fit_wall_time = time.perf_counter()-start
train_metrics = classification_metrics(splits['train'].class_id, model.predict_proba(splits['train'][LM_COLUMNS].to_numpy()))
validation_metrics = classification_metrics(splits['validation'].class_id, model.predict_proba(splits['validation'][LM_COLUMNS].to_numpy()))
display(pd.DataFrame([train_metrics,validation_metrics],index=['train','validation'])[['accuracy','macro_f1']])
pd.DataFrame([{'smoke_test':SMOKE_TEST,'train_accuracy':train_metrics['accuracy'],'validation_accuracy':validation_metrics['accuracy'],
    'train_macro_f1':train_metrics['macro_f1'],'validation_macro_f1':validation_metrics['macro_f1']}]).to_csv(output_dir/'history.csv',index=False)
joblib.dump(model, output_dir/'model.joblib')
save_json(output_dir/'model_config.json', model_config)
# Cấu hình cố định; không refit train+validation.
test_probabilities = model.predict_proba(splits['test'][LM_COLUMNS].to_numpy())
'''
write('01_random_forest.ipynb',[
    markdown('# Random Forest — baseline trên 63 landmark\n\n'+INTRO),
    code(config.replace('{batch}','256').replace('{epochs}','1')+"\nMODEL_NAME = 'rf'"),
    markdown('### Kiểm tra dữ liệu\nHash manifest/ảnh, schema, mapping và split phải hợp lệ; lỗi sẽ dừng notebook.'),code(common),code(setup),
    markdown('### Huấn luyện RF\n300 cây, độ sâu tối đa 20, lá tối thiểu 2 mẫu. So sánh Accuracy train/validation để quan sát overfitting. Cấu hình này được chọn trước khi xem test.'),
    code(rf_train),markdown('### Đánh giá test đã cố định\nAccuracy, macro-Precision/Recall/F1 và báo cáo từng lớp. Gallery chỉ phục vụ phân tích lỗi.'),code(end),ending])

neural_setup='''seed_everything(SEED)
device = select_device(DEVICE)
env = environment_info(device)
env['versions']['torch'] = torch.__version__
env['gpu'] = torch.cuda.get_device_name(device) if device.type=='cuda' else None
display(env)
'''
mlp_data='''scaler, train_features = fit_scaler(splits['train'][LM_COLUMNS].to_numpy())
joblib.dump(scaler, output_dir/'scaler.joblib')
features = {'train':train_features,
    'validation':scaler.transform(splits['validation'][LM_COLUMNS].to_numpy()).astype(np.float32),
    'test':scaler.transform(splits['test'][LM_COLUMNS].to_numpy()).astype(np.float32)}
loaders = {name:make_loader(TensorDataset(torch.from_numpy(features[name]),
    torch.tensor(rows.class_id.to_numpy(),dtype=torch.long)), BATCH_SIZE,
    name=='train',SEED,NUM_WORKERS) for name,rows in splits.items()}
model = GestureMLP()
model_config = {'model':'mlp','architecture':[63,128,64,6],'activation':'ReLU','dropout':.2,
    'preprocessing':'StandardScaler fit chỉ train; scaler.joblib','features':LM_COLUMNS}
'''
cnn_data='''loaders = {name:make_loader(GestureImages(rows,root,training=name=='train',image_size=128),
    BATCH_SIZE,name=='train',SEED,NUM_WORKERS) for name,rows in splits.items()}
model = GestureCNN()
model_config = {'model':'cnn','architecture':'3 x Conv3x3(padding1)-BN-ReLU-MaxPool2, channels32/64/128; GAP-Dropout-Linear6',
    'dropout':.2,'image_size':128,'preprocessing':'EXIF orientation, RGB, bilinear resize128x128, pixels/255',
    'train_augmentation':{'rotation_degrees':10,'brightness':[.85,1.15],'contrast':[.85,1.15],'flip':False}}
'''
neural_train='''model_config.update({'seed':SEED,'smoke_test':SMOKE_TEST,'batch_size':BATCH_SIZE,
    'max_epochs':2 if SMOKE_TEST else MAX_EPOCHS,'patience':PATIENCE,
    'run_name':RUN_NAME or MODEL_NAME,
    'optimizer':'AdamW','learning_rate':OPTIMIZER_CONFIG['lr'],'weight_decay':OPTIMIZER_CONFIG['weight_decay'],
    'optimizer_config':dict(OPTIMIZER_CONFIG),
    'scheduler':None if SCHEDULER_CONFIG is None else {'name':'ReduceLROnPlateau',
        'monitor':'validation_macro_f1',**SCHEDULER_CONFIG},
    'loss':'CrossEntropyLoss','class_mapping':CLASS_MAPPING,
    'selection':'highest validation macro-F1, then lowest validation loss'})
model, history, training_time, fit_wall_time, best_epoch = fit_neural(
    model,loaders['train'],loaders['validation'],device,model_config['max_epochs'],PATIENCE,output_dir,
    optimizer_config=OPTIMIZER_CONFIG,scheduler_config=SCHEDULER_CONFIG)
model_config['best_epoch'] = best_epoch
save_json(output_dir/'model_config.json',model_config)
history['smoke_test'] = SMOKE_TEST
history.to_csv(output_dir/'history.csv',index=False)
print('Checkpoint đã cố định tại epoch:',best_epoch)
_, validation_metrics, _ = neural_pass(model,loaders['validation'],device)
# Test được đánh giá sau khi đã tải checkpoint tốt nhất theo validation.
_, _, test_probabilities = neural_pass(model,loaders['test'],device)
'''
for name,title,batch,epochs,data,explanation in [
    ('mlp','MLP — baseline trên landmark',256,60,mlp_data,
     'StandardScaler fit **chỉ train**; validation/test chỉ transform. Mạng 63 → 128 → 64 → 6, ReLU và dropout 0,2.'),
    ('cnn','CNN nhẹ — tinh chỉnh trên ảnh RGB',64,50,cnn_data,
     'Ảnh EXIF → RGB → resize 128×128 → pixel [0,1]. Chỉ train xoay ±10° và thay sáng/tương phản ±15%; không flip. Ba block Conv–BN–ReLU–MaxPool, 32/64/128 kênh, GAP và dropout 0,2; không pretrained.')]:
    neural_config=config.replace('{batch}',str(batch)).replace('{epochs}',str(epochs))
    neural_config+=f"\nMODEL_NAME = '{name}'\n"
    if name=='cnn':
        neural_config=neural_config.replace('PATIENCE = 8','PATIENCE = 12')
        neural_config+="""RUN_NAME = 'cnn_tuned'
OPTIMIZER_CONFIG = {'lr':3e-4,'weight_decay':1e-4}
SCHEDULER_CONFIG = {'mode':'max','factor':.5,'patience':3,
    'threshold':1e-4,'threshold_mode':'abs','min_lr':1e-6}
"""
        training_explanation='AdamW lr=3e-4, weight decay=1e-4, CrossEntropyLoss. Tối đa 50 epoch, early stopping patience 12. ReduceLROnPlateau theo validation macro-F1 (max), factor 0,5, patience 3, threshold tuyệt đối 1e-4, min lr 1e-6. Learning rate trong history là giá trị dùng để train epoch đó; scheduler cập nhật cho epoch kế tiếp. Output: hand_gesture/cnn/; tên phiên cnn_tuned được giữ trong metadata.'
    else:
        neural_config+="OPTIMIZER_CONFIG = {'lr':1e-3,'weight_decay':1e-4}\nSCHEDULER_CONFIG = None\n"
        training_explanation='AdamW lr=1e-3, weight decay=1e-4, CrossEntropyLoss. Patience 8; không dùng scheduler.'
    training_explanation+=' Chọn macro-F1 validation cao nhất và validation loss nhỏ hơn khi hòa. Train metrics đo trong các batch có dropout/augmentation; validation đo ở chế độ eval.'
    write('02_mlp.ipynb' if name=='mlp' else '03_cnn.ipynb',[
        markdown('# '+title+'\n\n'+INTRO),code(neural_config),
        markdown('### Kiểm tra dữ liệu và thiết bị\nChọn GPU nếu có; CPU vẫn chạy được. Không tải thêm thư viện hoặc weights.'),
        code(common),code(torch_code),code(setup),code(neural_setup),
        markdown('### Preprocessing và kiến trúc\n'+explanation),code(data),
        markdown('### Huấn luyện và chọn checkpoint\n'+training_explanation),
        code(neural_train),markdown('### Đánh giá test\nKhông refit và không dùng test để chọn checkpoint.'),code(end),ending])

compare_config='''from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
RESULTS_ROOT = None  # None: tự tìm results/ từ thư mục kernel lên các thư mục cha
RESULT_DIRS = None  # Ghi đè từng model nếu cần: {'rf':'.../rf','mlp':'.../mlp','cnn':'.../cnn'}
OUTPUT_ROOT = None  # None: results/ local hoặc /kaggle/working/hand_gesture
RUN_BENCHMARK = False  # True: cần dataset gốc, Torch và checkpoint tin cậy để đo CPU
DATA_ROOT = None  # Chỉ dùng khi RUN_BENCHMARK=True; tự tìm data/ local hoặc dataset Kaggle
SEED = 42
DEVICE = 'cpu'
WARMUP = 20
BENCHMARK_SAMPLES = 200
'''
compare_run='''result_dirs, output_root, local_results_root = resolve_comparison_paths(
    results_root=RESULTS_ROOT,result_dirs=RESULT_DIRS,output_root=OUTPUT_ROOT)
output_dir = output_root/'comparison'
table = run_comparison(result_dirs,output_dir,run_benchmark=RUN_BENCHMARK,
    data_root=DATA_ROOT,seed=SEED,warmup=WARMUP,sample_count=BENCHMARK_SAMPLES,
    data_start=local_results_root.parent if local_results_root is not None else None)
display(table)
best = table.loc[table.macro_f1.idxmax()]
message = f"Trong lần đo này, **{best.model}** có macro-F1 cao nhất ({best.macro_f1:.4f}). "
if RUN_BENCHMARK:
    fastest = table.loc[table.cpu_latency_median_ms.idxmin()]
    message += f"**{fastest.model}** có median classifier latency thấp nhất ({fastest.cpu_latency_median_ms:.3f} ms). "
else:
    message += 'Benchmark đang tắt: không đọc dataset, không tải model và không công bố latency. '
if bool(table.smoke_test.iloc[0]):
    message += 'Đây là smoke test, không dùng các thứ hạng này để chọn mô hình. '
message += 'Thời gian train chỉ mô tả từng phiên; không kết luận công bằng khi phần cứng khác nhau. '
display(Markdown(message))
print('Báo cáo:',output_dir/'report.md')
'''
write('04_compare_models.ipynb',[
    markdown('# So sánh RF, MLP và CNN từ kết quả đã lưu\n\n'
        'Local: mở notebook và chọn **Run All**. Notebook tự tìm `results/rf`, `results/mlp`, `results/cnn` '
        'từ thư mục kernel hoặc thư mục cha; chạy được từ gốc dự án và `notebooks/kaggle/`. '
        'Mặc định không cần dataset, Torch hoặc tải checkpoint. Output nằm tại `results/comparison/`.\n\n'
        'Đặt `RESULTS_ROOT` nếu results ở nơi khác; `RESULT_DIRS` ưu tiên cao hơn để chọn từng model. '
        '`OUTPUT_ROOT` ghi đè root output (notebook tạo thư mục `comparison/` bên trong). '
        'Không tìm trong `archive/` khi dùng cấu trúc local chuẩn.\n\n'
        'Kaggle: gắn output của ba notebook bằng Add Input → Notebook Output; '
        'khi không có results local, notebook tìm trong `/kaggle/input` và xuất vào `/kaggle/working/hand_gesture/comparison/`.\n\n'
        'Muốn đo CPU: đặt `RUN_BENCHMARK=True`, chọn kernel có Torch và gắn dataset gốc hoặc đặt `DATA_ROOT`. '
        'Chỉ bật với checkpoint/joblib tin cậy. Không huấn luyện lại model.'),
    code(compare_config),
    markdown('### Kiểm tra kết quả đã lưu\n'
        'Kiểm tra hash artifact/dự đoán, mapping, seed, manifest hash, ID/nhãn test và metric khớp CSV. '
        'Chế độ mặc định chỉ xác minh tính nhất quán giữa các bộ kết quả; đối chiếu dataset gốc khi bật benchmark.'),
    code(common),
    markdown('### Phần phụ thuộc benchmark (mặc định bỏ qua)\nChỉ import Torch và định nghĩa model khi `RUN_BENCHMARK=True`. Khi đổi cờ, chạy lại toàn bộ notebook.'),
    code('if RUN_BENCHMARK:\n'+textwrap.indent(torch_code.strip(),'    ')),
    code(comparison),
    markdown('### Tổng hợp kết quả và benchmark tùy chọn\n'
        'Luôn xuất CSV, biểu đồ và báo cáo Markdown theo cùng mẫu. '
        'Benchmark khi bật: cùng CPU, một thread, batch 1, warm-up 20 lượt và 200 mẫu test chọn cố định. '
        'MLP gồm scaler; CNN dùng tensor đã chuẩn bị; không tính đọc ảnh, crop hoặc MediaPipe. '
        'Thiếu dữ liệu hoặc sai protocol sẽ dừng, không tự tắt benchmark.'),
    code(compare_run),
    markdown('### Giới hạn diễn giải\n'
        'RF/MLP dùng landmark, CNN dùng ảnh. Split chưa tách người chụp. '
        'Classifier latency không phải FPS toàn hệ thống; không suy ra tốc độ train công bằng từ phần cứng khác nhau. '
        'Model/checkpoint đã được cố định trước test; notebook này không fit hoặc tinh chỉnh mô hình.')])
print('Generated four standalone notebooks.')
