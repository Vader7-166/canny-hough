"""Tu soat bao cao tong hop: doc nguoc file DOCX va doi chieu voi CSV, ground truth va bao cao Phan 1-4.

Chay: python scripts/verify_tong_hop.py   (thoat voi ma 1 neu co sai lech)
"""
import sys
from pathlib import Path

import pandas as pd
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / 'baocao' / 'Bao_cao_Tong_hop_va_danh_gia.docx'
errors, checks = [], 0


def check(ok, msg):
    global checks
    checks += 1
    if not ok:
        errors.append(msg)


def vn(x, d):
    return f'{x:.{d}f}'.replace('.', ',')


def docx_text(path):
    d = Document(path)
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for r in t.rows:
            parts.append(' | '.join(c.text for c in r.cells))
    return d, '\n'.join(parts)


doc, text = docx_text(DOCX)
tables = [[[c.text.strip() for c in r.cells] for r in t.rows] for t in doc.tables]


def table_with_header(first_cells):
    return next(t for t in tables if t[0][:len(first_cells)] == first_cells)


# 1. Bang 3, 4 so voi hough_group_statistics.csv
stats = pd.read_csv(ROOT / 'outputs/hough/hough_group_statistics.csv').set_index('group')
label = {'Normal': 'normal', 'Curved': 'curved', 'Adverse': 'adverse', 'Toàn tập': 'all'}
t3 = table_with_header(['Nhóm', 'N', 'Bắt được biên trái'])
for row in t3[1:]:
    s = stats.loc[label[row[0]]]
    check(row[1:] == [str(int(s['count'])), vn(s.left_available_pct, 1) + '%', vn(s.right_available_pct, 1) + '%',
                      vn(s.polygon_available_pct, 1) + '%'], f'Bang 3 lech: {row}')
t4 = table_with_header(['Nhóm', 'IoU toàn tập ± σ'])
for row in t4[1:]:
    s = stats.loc[label[row[0]]]
    check(row[1:] == [f'{vn(s.mean_iou_all, 4)} ± {vn(s.std_iou_all, 4)}', vn(s.mean_iou_with_polygon, 4),
                      str(int(s.mae_evaluable_count)), f'{vn(s.mean_mae_evaluable, 2)} ± {vn(s.std_mae_evaluable, 2)}',
                      vn(s.success_mae15_pct, 1) + '%'], f'Bang 4 lech: {row}')

# 2. Phu luc A so voi hough_50_images_results.csv
res = pd.read_csv(ROOT / 'outputs/hough/hough_50_images_results.csv').set_index('image')
cond = pd.read_csv(ROOT / 'outputs/canny/conditions.csv').set_index('name')
ta = table_with_header(['Ảnh', 'Điều kiện'])
check(len(ta) - 1 == len(res) == 50, 'Phu luc A khong du 50 anh')
yes = lambda v: 'Có' if v else 'Không'
for row in ta[1:]:
    r = res.loc[row[0]]
    mae = '–' if pd.isna(r.mae) else vn(r.mae, 1)
    check(row[1:] == [cond.loc[row[0], 'condition'], yes(r.left_available), yes(r.right_available),
                      yes(r.polygon_available), vn(r.iou, 3), mae], f'Phu luc A lech: {row}')

# 3. Bang 6 (dieu kien) tinh lai doc lap
df = res.reset_index().merge(cond.reset_index().rename(columns={'name': 'image'})[['image', 'condition']], on='image')
t6 = table_with_header(['Điều kiện', 'N'])
for row in t6[1:]:
    sub = df[df.condition == row[0]]
    check(row[1:] == [str(len(sub)), vn(sub.raw_lines.mean(), 1), vn(100 * sub.polygon_available.mean(), 1) + '%',
                      f'{vn(sub.iou.mean(), 4)} ± {vn(sub.iou.std(ddof=0), 4)}', str(int(sub.mae_evaluable.sum()))],
          f'Bang 6 lech: {row}')

# 4. Phan bo IoU (Bang 5) tinh lai doc lap
t5 = table_with_header(['Nhóm', 'Không có đa giác (IoU = 0)'])
for row in t5[1:]:
    sub = res if row[0] == 'Toàn tập' else res[res.group == label[row[0]]]
    p = sub.polygon_available == 1
    want = [(~p).sum(), (p & (sub.iou < .3)).sum(), ((sub.iou >= .3) & (sub.iou < .5)).sum(),
            ((sub.iou >= .5) & (sub.iou < .7)).sum(), (sub.iou >= .7).sum()]
    check(row[1:] == [str(int(x)) for x in want], f'Bang 5 lech: {row}')

# 5. Trich dan tu bao cao cac thanh vien phai co trong nguon goc
src = {f.stem: f.read_text(encoding='utf-8') for f in (ROOT / 'baocao').glob('Bao_cao_Phan_[234]*.md')}
p1 = docx_text(ROOT / 'baocao/Bao_cao_Phan_1_De_tai_va_du_lieu.docx')[1]
p2, p3, p4 = (next(v for k, v in src.items() if k.startswith(f'Bao_cao_Phan_{i}')) for i in (2, 3, 4))
quotes = [
    (p1, 'adverse_01_mask.png'), (p1, 'lọc màu mặt đường'), (p1, 'đủ sáng'), (p1, 'BDD100K'), (p1, 'IDD'),
    (p1, '60%, 75% và 90%'), (p1, 'spline'),
    (p2, '1280 × 720'), (p2, '`S ≤ 80` và `V ≥ 170`'), (p2, '`15 ≤ H ≤ 40`, `S ≥ 70` và `V ≥ 90`'), (p2, 'Lanczos'),
    (p2, 'không được dùng IoU giữa mặt nạ màu và ground truth hiện tại'), (p2, 'validation'),
    (p3, '0,952 trung bình và 0,919 thấp nhất'), (p3, '0,2 đến 0,3 giây'), (p3, '| Normal (18) | 4 | 5 | 2 | 7 |'),
    (p3, 'chỉ trả lời được khi chạy cả hai nhánh qua Hough'), (p3, 'trung vị'), (p3, 'apertureSize = 3'),
    (p4, 'Runtime Hough trung bình: 3.374 ms/ảnh'), (p4, '| Standard Hough | 2.886 | 50 |'), (p4, '| P-Hough | 2.111 | 50 |'),
    (p4, '| 25 | 25 | 25 | 0.1743 | 10 | 272.40 |'), (p4, '| 15 | 20 | 25 | 0.2579 | 14 | 239.86 |'),
    (p4, 'Lưới 27 tổ hợp'), (p4, '0,02W'), (p4, 'Lọc trắng/vàng có thể xóa mép nhựa đường'),
    (p4, 'mô hình phân đoạn theo chiều cao hoặc đường cong'), (p4, 'Cần tập validation/test độc lập'),
]
for source, q in quotes:
    check(q in source, f'Khong tim thay trich dan trong bao cao goc: {q!r}')
# va cac gia tri do phai xuat hien dung trong bao cao tong hop
for q in ['0,952', '0,919', '0,2 đến 0,3 giây', '3,374 ms', '2,886 ms', '2,111 ms', '0,1743', '0,2579', '(15, 20, 25)',
          'adverse_01_mask.png', '1280 × 720', 'S ≤ 80, V ≥ 170', '15 ≤ H ≤ 40, S ≥ 70, V ≥ 90', 'xếp 2 ảnh Normal']:
    check(q in text, f'Thieu trong bao cao tong hop: {q!r}')

# 6. Du lieu goc: ten file ground truth that su
check((ROOT / 'dataset/ground_truth/adverse/adverse_01.png').is_file(), 'adverse_01.png khong ton tai')
check(not (ROOT / 'dataset/ground_truth/adverse/adverse_01_mask.png').exists(), 'adverse_01_mask.png lai ton tai')

# 7. Notebook tong hop khop ket qua chinh thuc
nb = pd.read_csv(ROOT / 'outputs/pipeline_notebook/results.csv').set_index('image').sort_index()
cols = ['iou', 'mae', 'left_available', 'right_available', 'polygon_available', 'raw_lines']
check(((nb[cols].fillna(-1) - res.sort_index()[cols].fillna(-1)).abs().max() < 1e-9).all(), 'Notebook lech ket qua chinh thuc')

print(f'{checks} kiem tra, {len(errors)} sai lech')
for e in errors:
    print(' -', e)
sys.exit(1 if errors else 0)
