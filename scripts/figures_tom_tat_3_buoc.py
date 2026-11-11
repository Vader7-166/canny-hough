"""Hinh cho baocao/Bao_cao_Tom_tat_Tien_xu_ly_Canny_Hough.docx.

Hinh 1: mot anh toi (curved_06), hang tren la pipeline goc (Phan 2-4), hang duoi la pipeline v3.
Hinh 2: ba anh (normal / curved / adverse), cot trai la ket qua Hough goc, cot phai la v3.

Chay tu thu muc goc: python scripts/figures_tom_tat_3_buoc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from road_pipeline import RoadConfig, _read_rgb, draw_result, edge_map, normalize_contrast, refine_mask, \
    road_prior, segment_road  # noqa: E402
from preprocessing import resize_rgb  # noqa: E402

OUT = ROOT / "outputs" / "pipeline_v3" / "figures"
TILE = (480, 270)


def tile(img: np.ndarray, label: str) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, TILE, interpolation=cv2.INTER_AREA)
    img = img.copy()
    cv2.rectangle(img, (0, 0), (TILE[0], 26), (0, 0, 0), -1)
    cv2.putText(img, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def grid(rows: list[list[np.ndarray]], gap: int = 6) -> np.ndarray:
    h, w = TILE[1], TILE[0]
    n_r, n_c = len(rows), len(rows[0])
    canvas = np.full((n_r * h + (n_r - 1) * gap, n_c * w + (n_c - 1) * gap, 3), 255, np.uint8)
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            canvas[i * (h + gap):i * (h + gap) + h, j * (w + gap):j * (w + gap) + w] = im
    return canvas


def gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def gt_outline(img: np.ndarray, group: str, name: str) -> np.ndarray:
    gt = gray(ROOT / "dataset" / "ground_truth" / group / f"{name}.png")
    gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours((gt > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    cv2.drawContours(out, contours, -1, (255, 0, 255), 2, lineType=cv2.LINE_AA)
    return out


def figure_stages(group: str, name: str, iou_old: float, iou_new: float) -> Path:
    cfg = RoadConfig()
    src = next((ROOT / "dataset" / "images" / group).glob(f"{name}.*"))
    resized = resize_rgb(_read_rgb(src), cfg.width, cfg.height)
    enhanced, _ = normalize_contrast(resized, cfg)
    vp, hood_y, _ = road_prior(resized, cfg)
    grab = segment_road(enhanced, vp, hood_y, cfg)
    auto = edge_map(enhanced, vp)
    refined = refine_mask(grab, enhanced, vp, hood_y, auto, cfg)
    final = rgb(ROOT / "outputs" / "pipeline_v3" / "overlay" / group / f"{name}.png")

    mask_vis = resized.copy()
    mask_vis[grab & ~refined] = (220, 60, 60)          # bi cat bo boi hau xu ly
    mask_vis[refined & ~grab] = (60, 140, 255)          # duoc them vao (dong hinh thai)
    mask_vis[grab & refined] = (0.5 * mask_vis[grab & refined] + np.array([0, 120, 0])).astype(np.uint8)

    # Moi hang la mot buoc, cot trai: pipeline goc (Phan 2-4), cot phai: pipeline v3
    rows = [
        [tile(resized, "Anh goc (960x540)"),
         tile(enhanced, "Sau: gamma + CLAHE tren kenh L")],
        [tile(gray(ROOT / "outputs" / "preprocessed_for_canny" / "canny_input" / group / f"{name}.png"),
              "Truoc: loc HSV trang/vang -> Gauss -> xam"),
         tile(mask_vis, "Sau: GrabCut (xanh), do: bi cat, lam: lap")],
        [tile(gray(ROOT / "outputs" / "canny" / "edges" / group / f"{name}.png"), "Truoc: Canny (50, 150)"),
         tile(auto, "Sau: Canny nguong phan vi gradient")],
        [tile(gt_outline(rgb(ROOT / "outputs" / "hough" / "visualizations" / group / f"{name}.png"), group, name),
              f"Truoc: Hough + da giac, IoU {iou_old:.2f}"),
         tile(gt_outline(final, group, name), f"Sau: mat na v3 + bien Hough, IoU {iou_new:.2f}")],
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"stages_{name}.jpg"
    Image.fromarray(grid(rows)).save(path, quality=90)
    return path


def figure_compare(items: list[tuple[str, str]], old: pd.DataFrame, new: pd.DataFrame) -> Path:
    rows = []
    for group, name in items:
        o = rgb(ROOT / "outputs" / "hough" / "visualizations" / group / f"{name}.png")
        n = rgb(ROOT / "outputs" / "pipeline_v3" / "overlay" / group / f"{name}.png")
        io, iv = old.loc[name, "iou"], new.loc[name, "iou"]
        rows.append([tile(gt_outline(o, group, name), f"{name} - truoc (Phan 4): IoU {io:.2f}"),
                     tile(gt_outline(n, group, name), f"{name} - sau (v3): IoU {iv:.2f}")])
    path = OUT / "truoc_sau_3_anh.jpg"
    Image.fromarray(grid(rows)).save(path, quality=90)
    return path


def main() -> None:
    old = pd.read_csv(ROOT / "outputs" / "hough" / "hough_50_images_results.csv").set_index("image")
    new = pd.read_csv(ROOT / "outputs" / "pipeline_v3" / "results.csv").set_index("image")
    print(figure_stages("curved", "curved_06", old.loc["curved_06", "iou"], new.loc["curved_06", "iou"]))
    print(figure_compare([("normal", "normal_11"), ("curved", "curved_09"), ("adverse", "adverse_08")], old, new))


if __name__ == "__main__":
    main()
