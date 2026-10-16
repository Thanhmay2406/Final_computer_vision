"""Normalize saved RF/MLP/CNN outputs and regenerate reports without fitting models."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import json

import matplotlib
matplotlib.use('Agg')
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('result_support', ROOT/'scripts/kaggle_notebook_support.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def tree_hashes(folder):
    return {p.relative_to(folder): support.file_sha256(p) for p in folder.rglob('*') if p.is_file()}


def normalize_results(results_root, data_root):
    results_root, data_root = Path(results_root).resolve(), Path(data_root).resolve()
    sources = {}
    for name in support.MODEL_ORDER:
        candidates = list((results_root/name).rglob('metrics.json'))
        if len(candidates) != 1:
            raise ValueError(f'{name}: cần đúng một metrics.json, tìm thấy {candidates}')
        sources[name] = candidates[0].parent
    # Validate all sources before backing up, moving or overwriting anything.
    results = support.load_comparison(sources)
    for name, result in results.items():
        folder = result['directory'];metrics = result['metrics'];predictions = result['predictions']
        support.validate_result_metrics(metrics, predictions)
        config = json.loads((folder/'model_config.json').read_text())
        if config != metrics['config']:
            raise ValueError(f'{name}: config trong metrics khác model_config.json')
        if json.loads((folder/'class_mapping.json').read_text()) != support.CLASS_MAPPING:
            raise ValueError(f'{name}: mapping không khớp')
        history = pd.read_csv(folder/'history.csv', float_precision='round_trip')
        if history.empty:
            raise ValueError(f'{name}: history trống')
        for column in ['train_accuracy','validation_accuracy','train_macro_f1','validation_macro_f1']:
            if column not in history or history[column].isna().any():
                raise ValueError(f'{name}: history thiếu {column}')
        errors = (predictions.loc[predictions.y_true.ne(predictions.y_pred)].sort_values('sample_id')
                  .groupby('y_true',group_keys=False).head(2).head(12))
        for relative in errors.image_path:
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Đường dẫn ảnh không hợp lệ: {path}')
            with Image.open(data_root/path) as image:
                image.verify()
    cnn_root = results_root/'cnn'
    cnn_source = sources['cnn']
    if cnn_source != cnn_root:
        for path in cnn_source.iterdir():
            target = cnn_root/path.name
            if target.exists():
                raise ValueError(f'Không ghi đè artifact khi chuyển CNN: {target}')
        parent = cnn_source
        while parent.parent != cnn_root:
            if list(parent.parent.iterdir()) != [parent]:
                raise ValueError(f'Thư mục CNN lồng chứa dữ liệu ngoài phiên: {parent.parent}')
            parent = parent.parent
        archive = results_root/'archive/cnn_tuned_20261015'
        original = tree_hashes(cnn_root)
        if archive.exists():
            if tree_hashes(archive) != original:
                raise ValueError(f'Archive đã có nội dung khác: {archive}')
        else:
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(cnn_root, archive)
        if tree_hashes(archive) != original:
            raise ValueError('Bản lưu CNN không khớp nguyên trạng.')
        # The complete backup has been verified before any source is moved.
        for path in list(cnn_source.iterdir()):
            shutil.move(str(path), cnn_root/path.name)
        parent = cnn_source
        while parent != cnn_root:
            parent.rmdir()
            parent = parent.parent
        # Retain execution provenance in the verified archive, outside the canonical artifacts.
        for path in list(cnn_root.iterdir()):
            if path.name == 'run_status.json' or path.suffix == '.log':
                path.unlink()
        sources['cnn'] = cnn_root
        results['cnn']['directory'] = cnn_root
    for name in support.MODEL_ORDER:
        result = results[name];folder = result['directory']
        support.show_errors(result['predictions'], data_root, folder/'misclassified_gallery.png',
                            smoke=result['metrics']['protocol']['smoke_test'])
        support.write_model_outputs(folder, result['metrics'], result['predictions'])
    support.export_comparison(results, results_root/'comparison')
    (results_root/'README.md').write_text(
        '# Kết quả nhận diện cử chỉ tay\n\n'
        'Bắt đầu tại [báo cáo so sánh](comparison/report.md).\n\n'
        '- [Random Forest](rf/report.md)\n- [MLP](mlp/report.md)\n- [CNN tinh chỉnh](cnn/report.md)\n\n'
        'Mỗi model có cùng summary.csv, per_class_metrics.csv, history.csv và report.md. '
        'Metric, checkpoint, scaler và dự đoán gốc được giữ nguyên. '
        'Ô trống trong history là không áp dụng hoặc chưa được ghi; RF có một lượt fit.\n\n'
        'Nếu có bản lưu, xem thư mục archive/. Bản lưu bị loại khỏi việc tự tìm output so sánh.\n\n'
        'Có thể mở notebooks/kaggle/04_compare_models.ipynb và chọn Run All từ gốc dự án '
        'hoặc notebooks/kaggle/. Notebook tự đọc results/ và ghi comparison/, không cần dataset '
        'hoặc nạp model. Muốn đo CPU, bật RUN_BENCHMARK=True và cung cấp data/ hoặc DATA_ROOT.\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=ROOT/'results')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data')
    args = parser.parse_args()
    normalize_results(args.results_root, args.data_root)
    print('Normalized results:', args.results_root.resolve())
    print('Comparison:', (args.results_root/'comparison/report.md').resolve())


if __name__ == '__main__':
    main()
