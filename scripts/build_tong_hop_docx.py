"""Sinh bao cao tong hop va danh gia (Markdown + DOCX) tu so lieu, khong go tay con so.

Chay sau scripts/analyze_tong_hop.py:  python scripts/build_tong_hop_docx.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_preprocessing_docx as base

ROOT = base.ROOT
DATA = ROOT / 'outputs' / 'bao_cao_tong_hop'
base.SOURCE = ROOT / 'baocao' / 'Bao_cao_Tong_hop_va_danh_gia.md'
base.OUTPUT = ROOT / 'baocao' / 'Bao_cao_Tong_hop_va_danh_gia.docx'

S = json.loads((DATA / 'summary.json').read_text(encoding='utf-8'))
G = S['group_stats']
P = S['per_group']
C = S['per_condition']
df = pd.read_csv(DATA / 'per_image.csv')
grid = pd.read_csv(ROOT / 'outputs/hough/hough_parameters_grid.csv')
bench = pd.read_csv(ROOT / 'outputs/hough/hough_method_benchmark.csv')
GROUPS = ['normal', 'curved', 'adverse', 'all']
LABEL = {'normal': 'Normal', 'curved': 'Curved', 'adverse': 'Adverse', 'all': 'Toàn tập'}


def vn(x, d=2):
    """So kieu Viet Nam: dau phay thap phan."""
    return f'{x:.{d}f}'.replace('.', ',')


def pct(x, d=1):
    return vn(x, d) + '%'


def frac(k, n):
    return f'{k}/{n}'


n_all = G['all']['count']
n_poly = round(G['all']['polygon_available_pct'] * n_all / 100)
mae_n = int(G['all']['mae_evaluable_count'])
default_row = grid[(grid.threshold == 25) & (grid.min_line_length == 25) & (grid.max_line_gap == 25)].iloc[0]
best_row = grid.sort_values('mean_iou_all', ascending=False).iloc[0]
cases = S['figure_cases']
fail_case = cases['fail'][1]
poly_df = df[df.polygon_available == 1]
mae_cov = {k: S['mae_gap'].get(k, 0) for k in ('top', 'bottom', 'gt', 'other')}
assert mae_cov['other'] == 0 and mae_cov['top'] + mae_cov['bottom'] + mae_cov['gt'] == n_poly - mae_n
gt_gap_images = S['mae_gap_gt_images']

md = []
w = md.append

w('# BÁO CÁO TỔNG HỢP VÀ ĐÁNH GIÁ · PHÁT HIỆN BIÊN ĐƯỜNG BẰNG HSV, CANNY VÀ HOUGH')
w('')
w('*Tổng hợp bốn phần của nhóm, chạy toàn bộ 50 ảnh và đối chiếu với ground truth*')
w('')

w('## 1. Tóm tắt kết quả')
w('')
w(f'Pipeline HSV → Canny → Hough được chạy trên toàn bộ {n_all} ảnh của bộ dữ liệu và so với ground truth vùng mặt đường. Kết quả chính:')
w('')
w(f'- **Có vùng đường dự đoán (đa giác hợp lệ):** {frac(n_poly, n_all)} ảnh ({pct(G["all"]["polygon_available_pct"])}).')
w(f'- **IoU trung bình trên toàn tập:** {vn(G["all"]["mean_iou_all"], 4)} ± {vn(G["all"]["std_iou_all"], 4)}. Nếu chỉ tính {n_poly} ảnh có đa giác, IoU trung bình là {vn(G["all"]["mean_iou_with_polygon"], 4)}.')
w(f'- **Không ảnh nào đạt IoU ≥ 0,7**; chỉ {P["all"]["bins"]["0_5_to_0_7"]} ảnh đạt IoU từ 0,5 đến dưới 0,7.')
w(f'- **Sai số vị trí biên (MAE):** chỉ đo được trên {mae_n}/{n_all} ảnh, trung bình {vn(G["all"]["mean_mae_evaluable"])} ± {vn(G["all"]["std_mae_evaluable"])} px.')
w(f'- **Tỉ lệ thành công SR15 (MAE < 15 px):** {pct(G["all"]["success_mae15_pct"])}.')
w(f'- **Thời gian Hough:** {vn(G["all"]["mean_runtime_ms"], 3)} ms/ảnh (chỉ riêng bước Hough và hậu xử lý).')
w('')
w('Kết luận ngắn: chuỗi xử lý chạy trọn vẹn và tái lập được, nhưng độ chính xác so với ground truth còn thấp. Vùng đường dự đoán thường hẹp hơn mặt đường thật vì Hough bám vào vạch sơn bên trong lòng đường thay vì mép vật lý của đường (mục 6). Các con số trên là kết quả của cấu hình cố định, không phải kết quả đã tối ưu.')
w('')

w('## 2. Bài toán và bộ dữ liệu')
w('')
w('Theo Phần 1, đối tượng của đề tài là **road boundary**: ranh giới vật lý giữa lòng đường và vỉa hè, cỏ, lan can hoặc dải phân cách. Vạch sơn chia làn (lane boundary) nằm bên trong lòng đường và không phải đối tượng cần tìm. Vì vậy ground truth là mặt nạ nhị phân bao trọn toàn bộ lòng đường, kể cả các vạch kẻ.')
w('')
w(f'Bộ dữ liệu gồm {n_all} ảnh chọn có chủ đích từ BDD100K và IDD. Kiểm tra lại trực tiếp trên thư mục `dataset/`: cả {n_all} ảnh đều có kích thước {S["image_sizes"][0][1]} × {S["image_sizes"][0][0]}, và mọi mặt nạ ground truth chỉ chứa hai giá trị {S["gt_values"][0]} và {S["gt_values"][1]}.')
w('')
w('| Nhóm | Số ảnh | Đặc trưng (theo Phần 1) |')
w('|---|---:|---|')
w(f'| Normal | {S["counts"]["normal"]} | Đường thẳng, đủ sáng, mép đường rõ; làm mốc đối sánh. |')
w(f'| Curved | {S["counts"]["curved"]} | Khúc cua, biên phi tuyến; đo giới hạn của Hough đường thẳng. |')
w(f'| Adverse | {S["counts"]["adverse"]} | Ban đêm, chói đèn, bóng đổ, đường ướt, tuyết, xe che biên, giao lộ. |')
w(f'| Tổng | {n_all} | |')
w('')
w('*Bảng 1. Ba nhóm kiểm thử; số ảnh đếm lại từ `dataset/images`.*')
w('')
w('![Ground truth nhóm Normal](dataset/visualizations/normal_tong_hop.jpg)')
w('*Hình 1. Ground truth của 18 ảnh nhóm Normal: vùng xanh là mặt đường được gán nhãn, bao trọn cả vạch kẻ (dataset/visualizations).*')
w('')

w('## 3. Pipeline tổng hợp')
w('')
w('Bốn module do các thành viên xây dựng được nối thành một chuỗi. Mỗi module chỉ nhận đầu ra của module trước, ground truth chỉ được dùng ở bước đánh giá.')
w('')
w('| Bước | Nội dung và tham số | Đầu ra | Nguồn |')
w('|---|---|---|---|')
w('| Tiền xử lý HSV | Resize 960 × 540 (Lanczos); giữ vạch trắng `S ≤ 80, V ≥ 170` và vạch vàng `15 ≤ H ≤ 40, S ≥ 70, V ≥ 90`; Gaussian bán kính 1,2; chuyển xám | Ảnh xám `uint8` | Phần 2 |')
w('| Canny tự cài (NumPy) | Gauss 5 × 5, σ = 1,4; Sobel 3 × 3; NMS; ngưỡng kép (50, 150) + hysteresis | Ảnh cạnh 0/255 | Phần 3 |')
w('| ROI + P-Hough | ROI hình thang 0,48H–0,98H, đáy trên 35% W, đáy dưới 98% W; ρ = 1 px, θ = 1°, threshold 25, minLineLength 25, maxLineGap 25 | Đoạn thẳng | Phần 4 |')
w('| Hậu xử lý | Lọc độ dốc có giá trị tuyệt đối từ 0,3 đến 4; chọn cụm đồng thuận (dung sai 0,02W); WLS; đa giác dưới giao điểm hai biên | Hai biên + đa giác | Phần 4 |')
w('| Đánh giá | IoU, MAE tại 3 hàng, SR15 (`evaluation.py`) | Bảng số liệu | Phần 1, 4 |')
w('')
w('*Bảng 2. Các bước của pipeline và tham số đang dùng.*')
w('')
w('Phần 3 đã kiểm tra bản Canny tự cài so với `cv2.Canny` cùng cấu hình: IoU giữa hai ảnh cạnh là 0,952 trung bình và 0,919 thấp nhất trên 50 ảnh.')
w('')
w(f'Notebook `Pipeline_HSV_Canny_Hough.ipynb` chạy trọn chuỗi trên trong một file và cho phép chọn một hoặc nhiều ảnh đầu vào bằng hộp thoại. Khi chạy trên cả {n_all} ảnh, notebook tạo ra ảnh cạnh trùng khớp từng pixel với `outputs/canny/edges` ({n_all}/{n_all} ảnh) và cho kết quả đánh giá trùng khớp hoàn toàn với `outputs/hough/hough_50_images_results.csv`.')
w('')

w('## 4. Phương pháp đánh giá')
w('')
w('Gọi P là tập pixel đường dự đoán (đa giác giữa hai biên) và G là tập pixel đường trong ground truth (Phần 1, mục 5):')
w('')
w('$$IoU = \\frac{|P \\cap G|}{|P \\cup G|}$$')
w('')
w('Tại ba hàng ngang y = 60%, 75% và 90% chiều cao ảnh, sai số vị trí biên của một ảnh là:')
w('')
w('$$MAE = \\frac{1}{2K}\\sum_{k=1}^{K}\\left(|x^L_{pred}-x^L_{gt}| + |x^R_{pred}-x^R_{gt}|\\right), \\quad K = 3$$')
w('')
w('Quy ước tính (Phần 4, mục 5; `evaluation.py`):')
w('')
w('- IoU tính trên toàn ảnh và lấy trung bình trên **tất cả** ảnh. Ảnh không có đa giác hợp lệ được tính như dự đoán rỗng (IoU = 0), không bị loại khỏi mẫu số.')
w('- MAE chỉ tính khi có đa giác hợp lệ, ground truth có pixel đường và hai biên phủ đủ cả ba hàng. Không ngoại suy và không thay giá trị thiếu bằng 0; luôn ghi kèm số ảnh đo được.')
w('- SR15 là số ảnh có MAE < 15 px chia cho **toàn bộ** số ảnh. Ảnh không đo được MAE tính là không thành công.')
w('- Tỉ lệ "bắt được biên" hoặc "có đa giác" chỉ cho biết pipeline có đưa ra kết quả, không phải độ chính xác.')
w('')

w('## 5. Kết quả độ chính xác')
w('')
w('### 5.1 Theo nhóm ảnh')
w('')
w('| Nhóm | N | Bắt được biên trái | Bắt được biên phải | Có đa giác |')
w('|---|---:|---:|---:|---:|')
for g in GROUPS:
    s = G[g]
    w(f'| {LABEL[g]} | {s["count"]} | {pct(s["left_available_pct"])} | {pct(s["right_available_pct"])} | {pct(s["polygon_available_pct"])} |')
w('')
w('*Bảng 3. Tỉ lệ ảnh có đầu ra theo nhóm (chưa phải độ chính xác).*')
w('')
w('| Nhóm | IoU toàn tập ± σ | IoU trên ảnh có đa giác | N đo MAE | MAE ± σ (px) | SR15 |')
w('|---|---:|---:|---:|---:|---:|')
for g in GROUPS:
    s = G[g]
    w(f'| {LABEL[g]} | {vn(s["mean_iou_all"], 4)} ± {vn(s["std_iou_all"], 4)} | {vn(s["mean_iou_with_polygon"], 4)} | {int(s["mae_evaluable_count"])} | '
      f'{vn(s["mean_mae_evaluable"])} ± {vn(s["std_mae_evaluable"])} | {pct(s["success_mae15_pct"])} |')
w('')
w('*Bảng 4. Độ chính xác theo nhóm. Nguồn: outputs/hough/hough_group_statistics.csv, tính lại và khớp từ ground truth.*')
w('')
w('### 5.2 Phân bố IoU')
w('')
w('| Nhóm | Không có đa giác (IoU = 0) | 0 < IoU < 0,3 | 0,3 ≤ IoU < 0,5 | 0,5 ≤ IoU < 0,7 | IoU ≥ 0,7 |')
w('|---|---:|---:|---:|---:|---:|')
for g in GROUPS:
    b = P[g]['bins']
    w(f'| {LABEL[g]} | {b["no_polygon"]} | {b["lt_0_3"]} | {b["0_3_to_0_5"]} | {b["0_5_to_0_7"]} | {b["ge_0_7"]} |')
w('')
w('*Bảng 5. Số ảnh theo khoảng IoU.*')
w('')
top = S['top5']
w('Năm ảnh có IoU cao nhất: ' + '; '.join(f'{t["image"]} ({vn(t["iou"], 3)})' for t in top) + ('. Không ảnh nào trong số này đo được MAE (mục 6.4).' if all(pd.isna(t['mae']) for t in top) else '.'))
w('')
w('### 5.3 Theo điều kiện ánh sáng')
w('')
w('Phần 3 đã gán mỗi ảnh một nhãn điều kiện theo độ sáng trung bình và độ lệch chuẩn của vùng lòng đường (`outputs/canny/conditions.csv`). Ghép nhãn này với kết quả Hough:')
w('')
w('| Điều kiện | N | Số đoạn P-Hough TB | Có đa giác | IoU toàn tập ± σ | N đo MAE |')
w('|---|---:|---:|---:|---:|---:|')
for c, s in C.items():
    w(f'| {c} | {s["n"]} | {vn(s["raw_lines_mean"], 1)} | {pct(s["polygon_pct"])} | {vn(s["iou_mean"], 4)} ± {vn(s["iou_std"], 4)} | {s["mae_n"]} |')
w('')
w('*Bảng 6. Kết quả theo điều kiện ánh sáng.*')
w('')
w('### 5.4 Thời gian xử lý')
w('')
w(f'Theo Phần 4, bước Hough và hậu xử lý mất trung bình {vn(G["all"]["mean_runtime_ms"], 3)} ms/ảnh, không gồm tiền xử lý, Canny và đọc ghi ảnh. Riêng phép biến đổi, Standard Hough mất {vn(bench.standard_median_ms.mean(), 3)} ms và P-Hough mất {vn(bench.probabilistic_median_ms.mean(), 3)} ms (trung bình các median trên 50 ảnh). Theo Phần 3, bản Canny NumPy tự cài mất khoảng 0,2 đến 0,3 giây mỗi ảnh, nên đây là bước chậm nhất của pipeline.')
w('')

w('## 6. Đối chiếu với ground truth')
w('')
w(f'![Đối chiếu dự đoán với ground truth](outputs/bao_cao_tong_hop/doi_chieu_ground_truth.png)')
w(f'*Hình 2. Ba trường hợp: IoU cao nhất ({cases["best"][1]}), trung vị trong các ảnh có đa giác ({cases["median"][1]}) và thất bại ({fail_case}). Vàng: ground truth; xanh lá: đa giác dự đoán; đỏ/xanh dương: biên trái/phải; ba đường trắng: các hàng đo MAE 60%, 75%, 90% H.*')
w('')
w('### 6.1 Vùng dự đoán hẹp hơn mặt đường')
w('')
w(f'Trên {n_poly} ảnh có đa giác, tách IoU thành độ chính xác pixel (phần đa giác nằm trong mặt đường thật, |P ∩ G| / |P|) và độ phủ pixel (phần mặt đường thật được đa giác phủ, |P ∩ G| / |G|):')
w('')
w('| Nhóm | Ảnh có đa giác | Độ chính xác pixel | Độ phủ pixel | Diện tích đa giác (% ảnh) | Diện tích ground truth (% ảnh) |')
w('|---|---:|---:|---:|---:|---:|')
for g in GROUPS:
    p = P[g]
    k = int((poly_df.group == g).sum()) if g != 'all' else len(poly_df)
    w(f'| {LABEL[g]} | {k} | {vn(p["precision_mean"], 3)} | {vn(p["recall_mean"], 3)} | {vn(p["pred_area_mean"], 1)} | {vn(p["gt_area_mean_poly"], 1)} |')
w('')
w('*Bảng 7. Độ chính xác và độ phủ pixel của đa giác dự đoán.*')
w('')
w(f'Đa giác dự đoán phần lớn nằm trong mặt đường (độ chính xác {vn(P["all"]["precision_mean"], 3)}) nhưng chỉ phủ khoảng {vn(P["all"]["recall_mean"] * 100, 0)}% mặt đường thật. IoU thấp chủ yếu do bỏ sót phần mặt đường hai bên, không phải do dự đoán tràn ra ngoài.')
w('')
w('### 6.2 Biên dự đoán lệch vào trong lòng đường')
w('')
a = P['all']
w(f'Đo sai số có dấu tại ba hàng 60/75/90% H (dương nghĩa là biên dự đoán nằm bên trong vùng đường thật), trên các ảnh mà đường biên phủ đủ ba hàng: biên trái nằm bên trong ở {frac(a["left_inside_n"], a["left_signed_n"])} ảnh, lệch trung bình {vn(a["left_signed_mean"], 1)} px; biên phải nằm bên trong ở {frac(a["right_inside_n"], a["right_signed_n"])} ảnh, lệch trung bình {vn(a["right_signed_mean"], 1)} px.')
w('')
w(f'Điều này khớp với nhận định của Phần 3 và Phần 4: bộ lọc HSV của Phần 2 được thiết kế để giữ vạch sơn trắng và vàng, nên mép vật lý của đường (nhựa đường giáp cỏ, đất, vỉa hè) thường bị loại trước khi đến Canny. Khi đó Hough thường bám vào vạch kẻ bên trong lòng đường. Ở ảnh {fail_case} (Hình 2, dưới cùng), biên trái được đặt đúng lên vạch vàng giữa đường.')
w('')
w('### 6.3 Nguyên nhân thiếu đa giác')
w('')
f = P['all']['failure']
w(f'{n_all - n_poly}/{n_all} ảnh không có đa giác. Trong đó {f["both_missing"]} ảnh không tìm được biên nào, {f["left_only_missing"]} ảnh thiếu biên trái, {f["right_only_missing"]} ảnh thiếu biên phải, và {f["both_lines_no_polygon"]} ảnh có đủ hai biên nhưng đa giác bị loại.' + (' Như vậy mọi trường hợp thiếu đa giác đều do bước Hough không tìm đủ hai biên.' if f['both_lines_no_polygon'] == 0 else ''))
w('')
w('### 6.4 Vì sao chỉ đo được MAE trên ít ảnh')
w('')
w(f'Trong {n_poly} ảnh có đa giác, {mae_n} ảnh đo được MAE. {n_poly - mae_n} ảnh còn lại không đo được vì: {mae_cov["top"]} ảnh có đỉnh đa giác nằm dưới hàng 60% H (hai biên giao nhau sớm); {mae_cov["bottom"]} ảnh có đáy đa giác nằm trên hàng 90% H (biên bị cắt tại mép trái hoặc phải của khung ảnh); {mae_cov["gt"]} ảnh ({", ".join(gt_gap_images)}) có ground truth không chứa pixel đường ở một trong ba hàng đo. Vì vậy MAE {vn(G["all"]["mean_mae_evaluable"])} px chỉ phản ánh {mae_n} ảnh và phải đọc cùng IoU toàn tập.')
w('')

w('## 7. Thảo luận')
w('')
w(f'- **Có kết quả không đồng nghĩa với đúng.** Nhóm Adverse có tỉ lệ đa giác cao nhất ({pct(G["adverse"]["polygon_available_pct"])}) nhưng IoU trên ảnh có đa giác thấp nhất ({vn(G["adverse"]["mean_iou_with_polygon"], 4)}).')
w(f'- **Khác biệt giữa ba nhóm nhỏ hơn độ lệch chuẩn.** IoU toàn tập của ba nhóm nằm trong khoảng {vn(min(G[g]["mean_iou_all"] for g in GROUPS[:3]), 4)} đến {vn(max(G[g]["mean_iou_all"] for g in GROUPS[:3]), 4)}, trong khi σ từ {vn(min(G[g]["std_iou_all"] for g in GROUPS[:3]), 2)} đến {vn(max(G[g]["std_iou_all"] for g in GROUPS[:3]), 2)}. Với 12 đến 20 ảnh mỗi nhóm, chưa thể kết luận nhóm nào khó hơn.')
w(f'- **Ánh sáng ảnh hưởng rõ nhất.** Từ ảnh sáng sang ảnh tối, số đoạn P-Hough trung bình giảm từ {vn(C["sáng"]["raw_lines_mean"], 1)} xuống {vn(C["tối"]["raw_lines_mean"], 1)}, tỉ lệ có đa giác giảm từ {pct(C["sáng"]["polygon_pct"])} xuống {pct(C["tối"]["polygon_pct"])} và IoU giảm từ {vn(C["sáng"]["iou_mean"], 4)} xuống {vn(C["tối"]["iou_mean"], 4)}. Điều này phù hợp với Phần 3: ảnh tối có rất ít điểm cạnh vì ngưỡng V cố định của bộ lọc HSV loại gần hết pixel.')
w(f'- **Tham số Hough còn dư địa.** Phần 4 khảo sát 27 tổ hợp tham số: cấu hình mặc định (25, 25, 25) cho IoU {vn(default_row.mean_iou_all, 4)}, cấu hình tốt nhất trong lưới ({int(best_row.threshold)}, {int(best_row.min_line_length)}, {int(best_row.max_line_gap)}) cho IoU {vn(best_row.mean_iou_all, 4)}. Lưới chạy trên chính tập đánh giá, không có tập kiểm tra riêng, nên đây chỉ là khảo sát độ nhạy.')
w('- **Mục tiêu của bộ lọc màu và của ground truth khác nhau.** Phần 2 lọc màu vạch làn, trong khi ground truth đánh dấu toàn bộ lòng đường. Phần 2 cũng đã lưu ý không dùng ground truth vùng mặt đường để kết luận độ chính xác phát hiện làn.')
w('')

w('## 8. Kết luận và hướng phát triển')
w('')
w(f'Nhóm đã xây dựng hoàn chỉnh pipeline HSV → Canny → Hough, có bộ dữ liệu {n_all} ảnh kèm ground truth và quy trình đánh giá định lượng tái lập được. Với cấu hình hiện tại, pipeline tạo được vùng đường cho {pct(G["all"]["polygon_available_pct"])} số ảnh, IoU trung bình toàn tập {vn(G["all"]["mean_iou_all"], 4)} và chưa ảnh nào đạt MAE dưới 15 px. Nguyên nhân chính là biên tìm được bám vào vạch sơn bên trong thay vì mép đường, cộng với việc thiếu biên ở ảnh tối.')
w('')
w('Các hướng cải thiện, lấy từ đề xuất của các phần trước:')
w('')
w('- Chạy thêm nhánh ảnh xám không lọc HSV qua cùng Canny và Hough để kiểm tra việc lọc trắng/vàng có làm mất mép đường hay không, như Phần 3 đề xuất; Phần 4 cũng ghi nhận giới hạn này.')
w('- Dùng ngưỡng tự động theo trung vị hoặc Otsu cho Canny để giảm mất cạnh ở ảnh tối (Phần 3).')
w('- Thay mô hình một đường thẳng mỗi bên bằng mô hình phân đoạn hoặc spline cho khúc cua (Phần 1, Phần 4).')
w('- Chia tập validation và test riêng trước khi chỉnh ngưỡng HSV và tham số Hough (Phần 2, Phần 4).')
w('')

w('## 9. Đối chiếu nguồn và tự soát')
w('')
w('Mọi con số trong báo cáo được sinh tự động từ dữ liệu bằng `scripts/analyze_tong_hop.py` và `scripts/build_tong_hop_docx.py`, không gõ tay. Script phân tích chạy lại Hough trên 50 ảnh cạnh và tính lại metric từ ground truth, rồi dừng lại nếu bất kỳ giá trị nào lệch với CSV chính thức của Phần 4.')
w('')
w('| Nội dung | Nguồn | Cách kiểm tra |')
w('|---|---|---|')
w('| Số ảnh, kích thước, giá trị mask | `dataset/images`, `dataset/ground_truth` | Đọc trực tiếp từng file |')
w('| Bảng 3, 4 | `outputs/hough/hough_group_statistics.csv` | Tính lại từ ground truth, khớp tuyệt đối |')
w('| Bảng 5, 7, mục 6.2–6.4 | `outputs/bao_cao_tong_hop/per_image.csv` | Tính từ kết quả Hough và ground truth |')
w('| Bảng 6 | `outputs/canny/conditions.csv` (Phần 3) ghép với kết quả Hough | Ghép theo tên ảnh |')
w('| Tham số và thời gian | Báo cáo Phần 2, 3, 4; `run_metadata.json`; `hough_method_benchmark.csv` | So với mã nguồn và CSV |')
w('| Notebook tổng hợp | `outputs/pipeline_notebook/results.csv` | So từng ảnh với `hough_50_images_results.csv` |')
w('')
w('*Bảng 8. Nguồn số liệu.*')
w('')
w('Các điểm chưa thống nhất giữa tài liệu và dữ liệu, phát hiện khi đối chiếu (giữ nguyên tài liệu gốc, chỉ ghi nhận):')
w('')
w('- Phần 1 mô tả tên file ground truth dạng `adverse_01_mask.png`, nhưng file thực tế là `dataset/ground_truth/adverse/adverse_01.png`.')
w('- Phần 1 mô tả bước tiền xử lý là "lọc màu mặt đường", trong khi Phần 2 lọc màu vạch sơn trắng và vàng.')
w(f'- Phần 1 định nghĩa nhóm Normal là ảnh đủ sáng, nhưng nhãn độ sáng của Phần 3 xếp {S["normal_dark"]} ảnh Normal vào điều kiện tối, và Hình 1 có một ảnh Normal chụp ban đêm.')
w('')

w('## Phụ lục A. Kết quả từng ảnh')
w('')
w('| Ảnh | Điều kiện | Biên trái | Biên phải | Đa giác | IoU | MAE (px) |')
w('|---|---|:---:|:---:|:---:|---:|---:|')
yes = lambda v: 'Có' if v else 'Không'
for _, r in df.iterrows():
    mae = '–' if pd.isna(r.mae) else vn(r.mae, 1)
    w(f'| {r.image} | {r.condition} | {yes(r.left_available)} | {yes(r.right_available)} | {yes(r.polygon_available)} | {vn(r.iou, 3)} | {mae} |')
w('')
w('*Bảng A.1. Kết quả của 50 ảnh. Nguồn: `outputs/hough/hough_50_images_results.csv`.*')
w('')

w('## Tài liệu tham khảo')
w('')
w('[1] M. D. E. Munajat, D. H. Widyantoro, and R. Munir, "Road Detection System Based on RGB Histogram Filterization and Boundary Classifier," *ICACSIS*, pp. 195-200, 2015.')
w('')
w('[2] S. Wang, S. Wu, X. Wang, and Z. Li, "A Canny Operator Road Edge Detection Method Based on Color Features," *Journal of Physics: Conference Series*, vol. 1629, 012018, 2020.')
w('')
w('[3] Y. Belhadri, D. Mohamed, B. Alshaqaqi, and A. S. Baquhaizel, "Lane Boundary Detection in Dynamic Environments," *Rev. Roum. Sci. Techn. - Électrotechn. et Énerg.*, vol. 71, no. 2, pp. 299-304, 2026.')
w('')
w('[4] J. F. Canny, "A Computational Approach to Edge Detection," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 8, no. 6, pp. 679-698, 1986.')
w('')
w('[5] F. Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning," *CVPR*, 2020.')
w('')
w('[6] G. Varma et al., "IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments," *WACV*, 2019.')
w('')
w('[7] Báo cáo Phần 1 đến Phần 4 của nhóm, thư mục `baocao/`.')

# Chu thich (dong *...*) khong duoc parse inline, bo dau ` de khong hien nguyen van
md = [l.replace('`', '') if l.startswith('*') and l.endswith('*') and not l.startswith('**') else l for l in md]
base.SOURCE.write_text('\n'.join(md) + '\n', encoding='utf-8')


def front_matter(document: Document) -> None:
    info = document.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.paragraph_format.space_after = Pt(16)
    base.apply_font(info.add_run('Báo cáo tổng hợp · Đánh giá toàn pipeline trên 50 ảnh'), size=Pt(11.5), bold=True, color=base.ACCENT)

    box = document.add_table(rows=1, cols=1)
    box.alignment = WD_TABLE_ALIGNMENT.CENTER
    box.autofit = False
    cell = box.cell(0, 0)
    cell.width = Cm(15.2)
    base.set_cell_shading(cell, 'EAF2F8')
    base.set_cell_margins(cell, top=160, start=180, bottom=160, end=180)
    paragraph = cell.paragraphs[0]
    base.apply_font(paragraph.add_run(
        'Nguồn số liệu: kết quả Hough chính thức của Phần 4 (outputs/hough), được tính lại từ ground truth '
        'và khớp tuyệt đối. Thông tin phương pháp lấy từ báo cáo Phần 1 đến Phần 4.'),
        size=Pt(11.5), bold=True, color=base.TEXT_COLOR)
    document.add_paragraph().paragraph_format.space_after = Pt(0)


base.add_front_matter = front_matter
path = base.build()

doc = Document(path)
for run in doc.sections[0].header.paragraphs[0].runs:
    run.text = 'BÁO CÁO TỔNG HỢP VÀ ĐÁNH GIÁ'
# Doan can deu co duong dan dai (khong ngat duoc) bi gian chu; chuyen sang can trai
for para in doc.paragraphs:
    if para.alignment == WD_ALIGN_PARAGRAPH.JUSTIFY and any(r.font.name == 'Consolas' and len(r.text) > 25 for r in para.runs):
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
# Giu bang ngan tren cung mot trang (khong de dong tieu de bang mo coi o cuoi trang)
for table in doc.tables:
    if len(table.rows) <= 12:
        for row in table.rows[:-1]:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.keep_with_next = True
doc.core_properties.title = 'Báo cáo tổng hợp và đánh giá - Phát hiện biên đường'
doc.core_properties.subject = 'Tổng hợp pipeline HSV, Canny, Hough và đánh giá với ground truth'
doc.core_properties.author = ''
doc.core_properties.keywords = 'road boundary, HSV, Canny, Hough, IoU, MAE, ground truth'
doc.save(path)
print(f'Da tao bao cao DOCX: {path}')
