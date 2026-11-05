"""Thong ke danh gia tong hop (Phan 5) tu dau ra Hough chinh thuc va ground truth.

Chay: python scripts/analyze_tong_hop.py
Chi doc outputs/canny/edges, outputs/hough, outputs/canny/conditions.csv, dataset/;
chi ghi vao outputs/bao_cao_tong_hop/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import SAMPLE_ROWS, evaluate_road_result, summarize_records
from hough import detect_road_boundaries

OUT = ROOT / 'outputs' / 'bao_cao_tong_hop'
GROUPS = ['normal', 'curved', 'adverse']


def x_at(line, y):
    (xb, yb), (xt, yt) = line
    return xt + (xb - xt) * (y - yt) / (yb - yt)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    official = pd.read_csv(ROOT / 'outputs/hough/hough_50_images_results.csv')
    conditions = pd.read_csv(ROOT / 'outputs/canny/conditions.csv').rename(columns={'name': 'image'})

    # Kiem tra du lieu nguon
    sizes, gt_values = set(), set()
    for g in GROUPS:
        for p in sorted((ROOT / 'dataset/images' / g).glob('*.jpg')):
            sizes.add(cv2.imread(str(p)).shape[:2])
            gt_values |= set(np.unique(cv2.imread(str(ROOT / 'dataset/ground_truth' / g / f'{p.stem}.png'), 0)).tolist())

    rows, cache = [], {}
    for _, r in official.iterrows():
        g, n = r['group'], r['image']
        edges = cv2.imread(str(ROOT / 'outputs/canny/edges' / g / f'{n}.png'), 0)
        h, w = edges.shape
        gt = cv2.resize(cv2.imread(str(ROOT / 'dataset/ground_truth' / g / f'{n}.png'), 0), (w, h),
                        interpolation=cv2.INTER_NEAREST)
        res = detect_road_boundaries(edges)
        ev = evaluate_road_result(res, gt)
        cache[(g, n)] = (res, gt)
        # Doi chieu voi CSV chinh thuc: phai trung tung gia tri
        for k in ('iou', 'mae', 'left_available', 'right_available', 'polygon_available', 'success_mae15'):
            a, b = ev[k], r[k]
            assert (a is None and pd.isna(b)) or (a is not None and abs(a - b) < 1e-9), (g, n, k, a, b)

        row = dict(group=g, image=n, gt_area_pct=100 * np.count_nonzero(gt) / gt.size)
        if ev['polygon_available']:
            pred = np.zeros_like(gt)
            cv2.fillPoly(pred, [res.polygon], 1)
            inter = np.count_nonzero((pred > 0) & (gt > 0))
            row.update(pixel_precision=inter / np.count_nonzero(pred), pixel_recall=inter / np.count_nonzero(gt),
                       pred_area_pct=100 * np.count_nonzero(pred) / gt.size)
        ys = [min(h - 1, int(h * f)) for f in SAMPLE_ROWS]
        gt_rows = [np.flatnonzero(gt[y] > 0) for y in ys]
        if ev['polygon_available']:
            # Ly do khong do duoc MAE du co da giac
            top, bottom = res.polygon[1, 1], res.polygon[0, 1]
            row['mae_gap'] = ('ok' if ev['mae'] is not None else 'top' if top > ys[0] else 'bottom' if bottom < ys[-1]
                              else 'gt' if any(len(gr) == 0 for gr in gt_rows) else 'other')
        # Sai so co dau tai 3 hang: duong = bien du doan nam BEN TRONG vung duong GT
        for side, line in (('left', res.left_line), ('right', res.right_line)):
            if line is None or any(len(gr) == 0 for gr in gt_rows):
                continue
            (xb, yb), (xt, yt) = line
            if any(not yt <= y <= yb for y in ys):
                continue
            if side == 'left':
                d = [x_at(line, y) - gr[0] for y, gr in zip(ys, gt_rows)]
            else:
                d = [gr[-1] - x_at(line, y) for y, gr in zip(ys, gt_rows)]
            row[f'{side}_signed_err'] = float(np.mean(d))
        rows.append(row)

    extra = pd.DataFrame(rows)
    df = official.merge(extra, on=['group', 'image']).merge(conditions[['group', 'image', 'condition']],
                                                         on=['group', 'image'], how='left')
    df.to_csv(OUT / 'per_image.csv', index=False)

    # Kiem tra summarize tu CSV trung voi hough_group_statistics.csv
    stats = summarize_records(official)
    official_stats = pd.read_csv(ROOT / 'outputs/hough/hough_group_statistics.csv')
    assert np.allclose(stats.drop(columns='group').to_numpy(float),
                       official_stats.drop(columns='group').to_numpy(float), equal_nan=True)

    def bins(sub):
        iou = sub['iou']
        return {'no_polygon': int((sub['polygon_available'] == 0).sum()),
                'lt_0_3': int(((sub['polygon_available'] == 1) & (iou < 0.3)).sum()),
                '0_3_to_0_5': int(((iou >= 0.3) & (iou < 0.5)).sum()),
                '0_5_to_0_7': int(((iou >= 0.5) & (iou < 0.7)).sum()),
                'ge_0_7': int((iou >= 0.7).sum())}

    def failure(sub):
        nopoly = sub[sub['polygon_available'] == 0]
        return {'both_missing': int(((nopoly.left_available == 0) & (nopoly.right_available == 0)).sum()),
                'left_only_missing': int(((nopoly.left_available == 0) & (nopoly.right_available == 1)).sum()),
                'right_only_missing': int(((nopoly.left_available == 1) & (nopoly.right_available == 0)).sum()),
                'both_lines_no_polygon': int(((nopoly.left_available == 1) & (nopoly.right_available == 1)).sum())}

    per_group = {}
    for g in GROUPS + ['all']:
        sub = df if g == 'all' else df[df.group == g]
        poly = sub[sub.polygon_available == 1]
        per_group[g] = dict(
            n=len(sub), bins=bins(sub), failure=failure(sub),
            precision_mean=poly.pixel_precision.mean(), recall_mean=poly.pixel_recall.mean(),
            pred_area_mean=poly.pred_area_pct.mean(), gt_area_mean_poly=poly.gt_area_pct.mean(),
            left_signed_mean=sub.left_signed_err.mean(), left_signed_n=int(sub.left_signed_err.notna().sum()),
            left_inside_n=int((sub.left_signed_err > 0).sum()),
            right_signed_mean=sub.right_signed_err.mean(), right_signed_n=int(sub.right_signed_err.notna().sum()),
            right_inside_n=int((sub.right_signed_err > 0).sum()),
            left_mae_mean=sub.left_mae.mean(), left_mae_n=int(sub.left_mae.notna().sum()),
            right_mae_mean=sub.right_mae.mean(), right_mae_n=int(sub.right_mae.notna().sum()),
        )

    per_condition = {}
    for c in ['sáng', 'tương phản cao', 'vừa', 'tối']:
        sub = df[df.condition == c]
        per_condition[c] = dict(n=len(sub), polygon_pct=100 * sub.polygon_available.mean(),
                                iou_mean=sub.iou.mean(), iou_std=sub.iou.std(ddof=0),
                                mae_n=int(sub.mae_evaluable.sum()), raw_lines_mean=sub.raw_lines.mean())

    ranked = df.sort_values('iou', ascending=False)
    top5 = ranked.head(5)[['group', 'image', 'iou', 'mae', 'pixel_precision', 'pixel_recall']].to_dict('records')

    # Hinh doi chieu du doan voi ground truth: tot nhat, trung vi (trong anh co polygon), that bai
    poly_sorted = df[df.polygon_available == 1].sort_values('iou', ascending=False).reset_index(drop=True)
    best = poly_sorted.iloc[0]
    median = poly_sorted.iloc[len(poly_sorted) // 2]
    fail = df[(df.polygon_available == 0) & (df.left_available == 1)].iloc[0]
    panels = []
    for label, r in (('Tot nhat', best), ('Trung vi', median), ('That bai', fail)):
        res, gt = cache[(r.group, r.image)]
        rgb = cv2.imread(str(ROOT / 'outputs/preprocessed_for_canny/resized' / r.group / f'{r.image}.png'))
        over = rgb.copy()
        over[gt > 0] = (0.55 * over[gt > 0] + 0.45 * np.array([0, 215, 255])).astype(np.uint8)  # GT: vang (BGR)
        if res.polygon is not None:
            fill = over.copy()
            cv2.fillPoly(fill, [res.polygon], (0, 180, 0))
            over = cv2.addWeighted(fill, 0.35, over, 0.65, 0)
        for line, color in ((res.left_line, (0, 0, 255)), (res.right_line, (255, 0, 0))):
            if line is not None:
                cv2.line(over, line[0], line[1], color, 4, cv2.LINE_AA)
        for f in SAMPLE_ROWS:
            y = int(over.shape[0] * f)
            cv2.line(over, (0, y), (over.shape[1] - 1, y), (255, 255, 255), 1, cv2.LINE_AA)
        mae = 'khong do duoc' if pd.isna(r.mae) else f'{r.mae:.1f} px'
        text = f'{label}: {r.image} | IoU {r.iou:.3f} | MAE {mae}'
        cv2.rectangle(over, (0, 0), (over.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(over, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        crop = 190  # bo phan troi, giu vung duong va ba hang do
        panels.append(np.vstack([over[:44], over[crop:]]))
    cv2.imwrite(str(OUT / 'doi_chieu_ground_truth.png'), np.vstack(panels))

    summary = dict(
        image_sizes=sorted(map(list, sizes)), gt_values=sorted(gt_values),
        counts={g: int((df.group == g).sum()) for g in GROUPS},
        mae_gap={k: int(v) for k, v in df.mae_gap.value_counts().items()},
        mae_gap_gt_images=df[df.mae_gap == 'gt'].image.tolist(),
        normal_dark=int(((df.group == 'normal') & (df.condition == 'tối')).sum()),
        per_group=per_group, per_condition=per_condition, top5=top5,
        figure_cases={k: (r.group, r.image) for k, r in (('best', best), ('median', median), ('fail', fail))},
        group_stats=official_stats.set_index('group').to_dict('index'),
    )
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return summary


if __name__ == '__main__':
    s = main()
    print(json.dumps(s, ensure_ascii=False, indent=1, default=float))
