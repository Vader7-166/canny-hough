"""Generate Hough notebook; no second copy of the algorithm or metrics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.strip().splitlines(True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.strip().splitlines(True), execution_count=None, outputs=[]))


md('''
# Phần 4 · Hough và hậu xử lý road boundary

Đầu vào là ảnh cạnh uint8 0/255 đã được module Canny bàn giao tại `outputs/canny/edges`.
Giữ nguyên ba module trước, gồm nhánh tiền xử lý HSV trắng/vàng hiện có.
Mục tiêu đánh giá vẫn là mép vật lý lòng đường, không phải vạch chia làn.

ROI hình thang **cố định theo tỷ lệ**, P-Hough → lọc slope/vị trí → chọn cụm đồng thuận
→ WLS x(y) → cắt biên theo khung ảnh → vùng chung dưới giao điểm hai biên.
Đây là mô hình đường thẳng, không phải thuật toán giải quyết đầy đủ đường cong.

Tải cả dự án lên Drive, chỉnh ROOT; cần có đầu ra module 2 và 3 trước khi chạy.
Notebook chỉ đọc đầu vào và ghi kết quả Hough, không chạy lại hoặc thay đổi Canny.
Mã nguồn Hough dùng chung: `hough.py`, `evaluation.py`, `scripts/generate_hough_assets.py`.
''')
code('''
import sys
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from google.colab import drive
# drive.mount('/content/drive')
candidates = [Path.cwd(), Path.cwd().parent, Path('/content/drive/MyDrive/canny-hough-imageprocessing'), Path('/content/drive/MyDrive/Bai_Tap_Lon')]
ROOT = next((p for p in candidates if (p / 'hough.py').is_file() and (p / 'dataset').is_dir()), None)
if ROOT is None:
    raise FileNotFoundError('Chỉnh ROOT đến thư mục chứa hough.py và dataset')
sys.path.insert(0, str(ROOT))
from hough import HoughConfig, detect_road_boundaries, draw_hough_visualization, get_trapezoid_vertices, apply_roi
from evaluation import evaluate_road_result
from scripts.generate_hough_assets import load_samples
samples = load_samples(ROOT)
config = HoughConfig()
print('Số ảnh:', len(samples))
print(config)
''')
md('''
## 1. Quan sát các bước trên normal_05
Các đoạn cùng dấu slope không nhất thiết nằm trên cùng một biên. Mỗi đoạn đề xuất
một mô hình; hai đầu mút của inlier phải nằm trong dung sai vuông góc 0,02W.
Chọn cụm có tổng chiều dài lớn nhất rồi WLS, kiểm tra lại hướng.

Hai biên được cắt theo khung ảnh. Đa giác chỉ dùng miền y chung, dưới giao điểm;
không còn miền hợp lệ thì trả None. Không dùng ground truth trong quá trình chọn biên.
''')
code('''
sample = next(s for s in samples if s.name == 'normal_05')
result = detect_road_boundaries(sample.edges, config)
h, w = sample.edges.shape
roi, _ = apply_roi(sample.edges, get_trapezoid_vertices(h, w, config))
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
panels = [(sample.edges, 'Canny'), (roi, 'Sau ROI cố định'),
          (draw_hough_visualization(sample.rgb, result, True), 'Đoạn ứng viên và biên đã fit'),
          (draw_hough_visualization(sample.rgb, result, False, False), 'Vùng đường hợp lệ')]
for ax, (image, title) in zip(axes.flat, panels):
    ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
    ax.set_title(title); ax.axis('off')
plt.tight_layout(); plt.show(); plt.close(fig)
print(evaluate_road_result(result, sample.ground_truth))
''')
md('''
## 2. Chạy thí nghiệm tái lập trên toàn bộ dữ liệu
Lệnh sau chỉ đọc ảnh Canny đã bàn giao, tạo CSV, 50 ảnh overlay, bốn hình báo cáo và lưới 27 tổ hợp.
Lưới chứa mặc định (25,25,25); chỉ là khảo sát trên chính tập dữ liệu, không tự chọn
tham số hay công bố cấu hình tối ưu trên test.

- IoU chính: toàn ảnh, toàn bộ ảnh, thất bại dự đoán mask rỗng.
- Tỷ lệ có đầu ra không phải độ chính xác.
- MAE: cần cả hai biên tại đủ 60%, 75%, 90% H; không ngoại suy ngoài miền đã xuất.
- SR15: MAE < 15px, mẫu số là toàn bộ ảnh; ca thiếu biên không thành công.
- Runtime Hough không gồm tiền xử lý/Canny/I/O. Benchmark hai biến đổi có warmup,
  lấy median 5 lượt trên cùng ROI, không khẳng định P-Hough luôn nhanh hơn.
''')
code('''
completed = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_hough_assets.py')], cwd=ROOT, check=True, capture_output=True, text=True)
print(completed.stdout)
OUT = ROOT / 'outputs/hough'
summary = pd.read_csv(OUT / 'hough_group_statistics.csv')
display(summary.round(4))
''')
md('## 3. So sánh hai loại Hough và độ nhạy tham số')
code('''
benchmark = pd.read_csv(OUT / 'hough_method_benchmark.csv')
display(benchmark.groupby('group')[['standard_median_ms', 'probabilistic_median_ms']].mean().round(3))
grid = pd.read_csv(OUT / 'hough_parameters_grid.csv')
display(grid[['threshold', 'min_line_length', 'max_line_gap', 'polygon_available_pct', 'mean_iou_all', 'mae_evaluable_count', 'mean_mae_evaluable', 'success_mae15_pct']].round(4))
for name in ['figure_2_standard_vs_probabilistic.png', 'figure_3_parameter_grid_sensitivity.png']:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(plt.imread(OUT / 'figures' / name)); ax.axis('off')
    plt.tight_layout(); plt.show(); plt.close(fig)
''')
md('''
## 4. Các ca lỗi và giới hạn đầu vào
Tiền xử lý hiện có chỉ giữ trắng/vàng nên có thể xóa mép vật lý ngoài nhóm màu này.
Hough không thể khôi phục cạnh đã mất. Đây là giới hạn được ghi nhận, không phải lý do
để notebook tự thay đầu vào hoặc sửa module trước.

Đường có hình học hợp lệ vẫn có thể bám vạch làn, xe, bóng hoặc lề sai.
MAE chỉ có nghĩa trên số ảnh đánh giá được; luôn đọc kèm coverage và IoU toàn tập.
''')
code('''
fig, ax = plt.subplots(figsize=(13, 10))
ax.imshow(plt.imread(OUT / 'figures/figure_4_group_examples.png')); ax.axis('off')
plt.tight_layout(); plt.show(); plt.close(fig)
''')
md('''
## 5. Bàn giao
`detect_road_boundaries(edges, config) -> HoughResult`: hai biên hữu hạn, polygon hợp lệ
hoặc None, các đoạn ứng viên và ROI mask. Các dòng CSV ghi rõ availability,
polygon availability, số ảnh đủ ba mặt cắt, MAE, SR15 và IoU toàn tập.

Giới hạn: giả định biên trái slope âm/phải slope dương; ROI cố định và mô hình tuyến tính
không phù hợp mọi khúc cua/giao lộ. Consensus không tự phân biệt được mép vật lý với
vạch làn. Cần dữ liệu validation độc lập trước khi hiệu chỉnh hoặc kết luận khái quát.
''')
notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, nbformat=4, nbformat_minor=5)
path = ROOT / 'Hough_Colab.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(path)
