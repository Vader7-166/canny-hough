"""Hinh minh hoa cho bao cao pipeline v2 -> outputs/pipeline_v2/figures/.

Can chay truoc: python road_pipeline.py --edge-maps
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from canny import canny_edges  # noqa: E402

OUT = ROOT / "outputs/pipeline_v2/figures"
TILE = (480, 270)


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def gt_contour(img, group, name):
    gt = cv2.imread(str(ROOT / "dataset/ground_truth" / group / f"{name}.png"), 0)
    gt = cv2.resize(gt, (960, 540), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 255), 3)
    return img


def tile(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, TILE, interpolation=cv2.INTER_AREA)


def fig_canny(names):
    rows = []
    for group, name in names:
        resized = cv2.imread(str(ROOT / "outputs/preprocessed_for_canny/resized" / group / f"{name}.png"))
        hsv_in = cv2.imread(str(ROOT / "outputs/preprocessed_for_canny/canny_input" / group / f"{name}.png"), 0)
        old = canny_edges(hsv_in, 50, 150, 1.4)
        new = cv2.imread(str(ROOT / "outputs/pipeline_v2/edge_map" / group / f"{name}.png"), 0)
        rows.append(np.hstack([label(tile(resized), f"{name}: anh goc"),
                               label(tile(old), "Cu: loc HSV + Canny (50,150)"),
                               label(tile(new), "Moi: chuan hoa + nguong phan vi")]))
    cv2.imwrite(str(OUT / "canny_low_contrast.jpg"), np.vstack(rows))


def fig_compare(names):
    rows = []
    for group, name in names:
        nb = gt_contour(cv2.imread(str(ROOT / "outputs/pipeline_notebook/hough" / group / f"{name}.png")), group, name)
        v2 = gt_contour(cv2.imread(str(ROOT / "outputs/pipeline_v2/overlay" / group / f"{name}.png")), group, name)
        r_nb = pd.read_csv(ROOT / "outputs/pipeline_notebook/results.csv").set_index("image").loc[name, "iou"]
        r_v2 = pd.read_csv(ROOT / "outputs/pipeline_v2/results.csv").set_index("image").loc[name, "iou"]
        rows.append(np.hstack([label(tile(nb), f"{name} notebook: IoU {r_nb:.2f}"),
                               label(tile(v2), f"{name} v2: IoU {r_v2:.2f}")]))
    cv2.imwrite(str(OUT / "notebook_vs_v2.jpg"), np.vstack(rows))


def fig_grid(names, filename, per_row=3):
    res = pd.read_csv(ROOT / "outputs/pipeline_v2/results.csv").set_index("image")
    tiles = []
    for group, name in names:
        img = gt_contour(cv2.imread(str(ROOT / "outputs/pipeline_v2/overlay" / group / f"{name}.png")), group, name)
        tiles.append(label(tile(img), f"{name}: IoU {res.loc[name, 'iou']:.2f}"))
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(OUT / filename), np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_canny([("curved", "curved_06"), ("normal", "normal_15"), ("adverse", "adverse_02")])
    fig_compare([("normal", "normal_08"), ("curved", "curved_08"), ("adverse", "adverse_06"), ("normal", "normal_06")])
    res = pd.read_csv(ROOT / "outputs/pipeline_v2/results.csv").sort_values("iou")
    worst = [(r.group, r.image) for r in res.head(6).itertuples()]
    best = [(r.group, r.image) for r in res.tail(6).itertuples()][::-1]
    fig_grid(best, "best_cases.jpg")
    fig_grid(worst, "failure_cases.jpg")
    print("Da ghi hinh vao", OUT)


if __name__ == "__main__":
    main()
