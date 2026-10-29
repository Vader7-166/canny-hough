"""Metrics for road-area masks, not lane-marking masks.

IoU uses the entire image; missing/invalid polygons predict an empty mask.
MAE requires both boundaries at ALL three prescribed rows (60/75/90% H).
Missing MAE is not zero and never counts as success. Ground truth is used
only here, never by Hough or by preprocessing.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from hough import HoughResult

SAMPLE_ROWS = (0.60, 0.75, 0.90)
SUCCESS_MAE_PX = 15.0


def evaluate_road_result(
    result: HoughResult,
    ground_truth: np.ndarray,
) -> dict:
    if ground_truth.ndim != 2 or ground_truth.shape != result.roi_mask.shape:
        raise ValueError("Ground truth va ket qua phai cung kich thuoc H x W")
    gt = ground_truth > 0
    height, width = gt.shape
    pred = np.zeros((height, width), dtype=np.uint8)
    polygon_valid = False
    if result.polygon is not None:
        p = result.polygon
        polygon_valid = bool(
            p.shape == (4, 2) and np.all(np.isfinite(p))
            and np.all((p[:, 0] >= 0) & (p[:, 0] < width))
            and np.all((p[:, 1] >= 0) & (p[:, 1] < height))
            and p[0, 1] == p[3, 1] and p[1, 1] == p[2, 1]
            and p[0, 1] > p[1, 1]
            and p[0, 0] < p[3, 0] and p[1, 0] < p[2, 0]
            and cv2.isContourConvex(p.astype(np.int32))
        )
        if polygon_valid:
            cv2.fillPoly(pred, [p.astype(np.int32)], 1)
    intersection = np.count_nonzero((pred > 0) & gt)
    union = np.count_nonzero((pred > 0) | gt)
    iou = float(intersection / union) if union else 0.0

    gt_rows = [np.flatnonzero(gt[min(height - 1, int(height * r))]) for r in SAMPLE_ROWS]
    sample_y = [min(height - 1, int(height * r)) for r in SAMPLE_ROWS]

    def boundary_error(line):
        if line is None or any(len(row) == 0 for row in gt_rows):
            return None
        (xb, yb), (xt, yt) = line
        if yb <= yt or any(not yt <= y <= yb for y in sample_y):
            return None
        return np.array([xt + (xb - xt) * (y - yt) / (yb - yt) for y in sample_y])

    left = boundary_error(result.left_line)
    right = boundary_error(result.right_line)
    left_mae = float(np.mean(np.abs(left - [row[0] for row in gt_rows]))) if left is not None else None
    right_mae = float(np.mean(np.abs(right - [row[-1] for row in gt_rows]))) if right is not None else None
    mae = (left_mae + right_mae) / 2 if polygon_valid and left_mae is not None and right_mae is not None else None
    return {
        "left_available": int(result.left_line is not None),
        "right_available": int(result.right_line is not None),
        "both_available": int(result.left_line is not None and result.right_line is not None),
        "polygon_available": int(polygon_valid),
        "iou": iou,
        "left_mae": left_mae,
        "right_mae": right_mae,
        "mae": mae,
        "mae_evaluable": int(mae is not None),
        "success_mae15": int(mae is not None and mae < SUCCESS_MAE_PX),
    }


def summarize_records(records: pd.DataFrame) -> pd.DataFrame:
    """All-image denominators for coverage, IoU and success; explicit MAE count."""
    rows = []
    for group in [*records["group"].drop_duplicates(), "all"]:
        sub = records if group == "all" else records[records["group"] == group]
        valid = sub[sub["mae_evaluable"] == 1]
        polygons = sub[sub["polygon_available"] == 1]
        rows.append({
            "group": group,
            "count": len(sub),
            "mean_raw_lines": sub["raw_lines"].mean(),
            "left_available_pct": sub["left_available"].mean() * 100,
            "right_available_pct": sub["right_available"].mean() * 100,
            "both_available_pct": sub["both_available"].mean() * 100,
            "polygon_available_pct": sub["polygon_available"].mean() * 100,
            "mean_iou_all": sub["iou"].mean(),
            "std_iou_all": sub["iou"].std(ddof=0),
            "mean_iou_with_polygon": polygons["iou"].mean(),
            "mae_evaluable_count": len(valid),
            "mean_mae_evaluable": valid["mae"].mean(),
            "std_mae_evaluable": valid["mae"].std(ddof=0),
            "success_mae15_pct": sub["success_mae15"].mean() * 100,
            "mean_runtime_ms": sub["runtime_ms"].mean(),
        })
    return pd.DataFrame(rows)
