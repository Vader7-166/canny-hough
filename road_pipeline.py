"""Pipeline v2: tien xu ly chuan hoa tuong phan -> phan vung long duong -> Canny -> Hough.

Khac voi pipeline goc (loc vach son HSV -> Canny (50, 150) -> Hough, IoU 0,174) va ban
notebook (mat na mau S/V -> Canny -> Hough, IoU 0,559, chi hon baseline "to kin dai chan
troi -> capo" 0,552 rat it), ban nay:

1. Chuan hoa do tuong phan cho anh toi: gamma tu dong + CLAHE tren kenh L cua Lab
   (chi doi do sang, giu sac do a, b). Chi ap dung khi do sang trung vi nua duoi anh
   < `dark_threshold`, vi voi anh du sang CLAHE lam nhieu mo hinh mau.
2. Uoc luong diem tu, mep capo va vach son theo notebook Pipeline_HSV_Canny_Hough
   (cac ham duoc chuyen nguyen sang day de khong con chep code giua notebook va .py).
3. Phan vung long duong bang GrabCut (Rother et al., 2004): GMM mau + cat do thi, khoi tao
   bang rang buoc hinh hoc: tren duong chan troi va duoi mep capo la nen chac chan, cua so
   ngay truoc xe la mat duong chac chan, nem tu diem tu la "co the la duong".
4. Canny (canny.py, NumPy) tren mat na long duong -> HoughLinesP + hoi quy WLS (hough.py)
   -> hai bien trai/phai de ve va bao cao. Vung long duong cuoi cung la mat na o buoc 3.

Ngoai ra `edge_map` dung Canny voi nguong theo phan vi gradient (canny.canny_edges_auto)
tren anh da chuan hoa: tren 50 anh, recall bien long duong cua anh canh tang tu 0,277
(ban goc) len 0,809, rieng anh toi tu 0,090 len 0,737.

Chay: python road_pipeline.py  (mac dinh: dataset/images -> outputs/pipeline_v2)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from canny import canny_edges, canny_edges_auto
from hough import HoughConfig, detect_hough_lines_p, fit_boundary_line, split_left_right_segments
from preprocessing import SUPPORTED_EXTENSIONS, resize_rgb, rgb_to_hsv_opencv


@dataclass(frozen=True)
class RoadConfig:
    width: int = 960
    height: int = 540
    # Chuan hoa tuong phan (anh toi)
    dark_threshold: float = 50.0          # trung vi L (0..255) cua nua duoi anh
    target_lightness: float = 0.45        # gamma dua trung vi L ve muc nay
    max_gamma: float = 2.5
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    # Vach son (white / yellow top-hat), diem tu, mo hinh mau S-V va quet hang (giong notebook)
    tophat_size: int = 31
    white_tophat_min: int = 25
    white_saturation_max: int = 110
    yellow_hue_min: int = 15
    yellow_hue_max: int = 40
    yellow_tophat_min: int = 20
    yellow_saturation_min: int = 60
    vp_min_votes: int = 8
    default_horizon_ratio: float = 0.45
    seed_window: tuple = (0.62, 0.70, 0.42, 0.58)
    road_k: float = 3.5
    road_s_min: float = 10.0
    road_v_min: float = 30.0
    shadow_ratio: float = 0.5
    perspective_grow: int = 2
    wedge: float = 7.0
    vp_margin: int = 10
    stop_rows: int = 6
    min_width: int = 12
    smooth_rows: int = 15
    # GrabCut
    gc_seed_rows: tuple = (0.60, 0.68)    # cua so mat duong chac chan (ti le H)
    gc_seed_cols: tuple = (0.42, 0.58)    # (ti le W)
    gc_seed_follow_vp: bool = True        # dich cua so mau xuong duoi diem tu (False: cua so co dinh nhu ban dau)
    gc_wedge: float = 3.0                 # |x - x_vp| <= 3 (y - y_vp): "co the la duong"
    gc_horizon_margin: int = 15           # tren y_vp - 15 px: nen chac chan
    gc_hood_margin: int = 4               # duoi mep capo + 4 px: nen chac chan
    gc_bottom_ratio: float = 0.03         # 3% hang cuoi: nen chac chan (nap capo)
    gc_color_k: float | None = None       # neu dat: "co the la duong" chi gom pixel trong nem co khoang cach Lab toi mau mau < k
    gc_scale: float = 0.5                 # chay GrabCut o nua do phan giai
    gc_iterations: int = 5
    gc_seed: int | None = 0               # seed RNG cua OpenCV (k-means khoi tao GMM) de ket qua tai lap
    gc_votes: int = 3                     # chay GrabCut voi seed, seed+1, ... roi bo phieu da so (chong suy bien)
    # Canny tren mat na + Hough
    canny_low: float = 50.0
    canny_high: float = 150.0
    canny_sigma: float = 1.4
    hough: HoughConfig = field(default_factory=lambda: HoughConfig(min_abs_slope=0.10))


@dataclass(frozen=True)
class RoadResult:
    resized_rgb: np.ndarray
    enhanced_rgb: np.ndarray
    contrast_applied: bool
    vp: tuple
    hood_y: int
    road_mask: np.ndarray                 # mat na long duong cuoi cung (0/255)
    mask_edges: np.ndarray                # Canny tren mat na (0/255)
    left_line: tuple | None
    right_line: tuple | None
    edge_map: np.ndarray | None = None    # Canny nguong tu dong tren anh da chuan hoa (tuy chon)


# ---------------------------------------------------------------------------
# 1. Chuan hoa do tuong phan
# ---------------------------------------------------------------------------

def normalize_contrast(rgb: np.ndarray, cfg: RoadConfig = RoadConfig(), force: bool = False) -> tuple[np.ndarray, bool]:
    """Gamma tu dong + CLAHE tren kenh L (Lab). Tra ve (anh, co_ap_dung_hay_khong)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0]
    median_low = float(np.median(lightness[lightness.shape[0] // 2:]))
    if not force and median_low >= cfg.dark_threshold:
        return rgb, False
    level = float(np.median(lightness)) / 255.0
    if 0.01 < level < cfg.target_lightness:
        gamma = float(np.clip(np.log(cfg.target_lightness) / np.log(level), 1.0 / cfg.max_gamma, 1.0))
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        lightness = cv2.LUT(lightness, lut)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
    lab = lab.copy()
    lab[..., 0] = clahe.apply(lightness)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), True


# ---------------------------------------------------------------------------
# 2. Vach son, diem tu, mo hinh mau S-V, quet hang (chuyen tu notebook Pipeline_HSV_Canny_Hough)
# ---------------------------------------------------------------------------

def lane_marking_mask(hsv, v_clahe, road_s, cfg):
    # Vach son la cau truc sang va manh hon kernel top-hat; bau troi, tuong, than xe (vung lon) bi loai.
    hue, sat = hsv[..., 0], hsv[..., 1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.tophat_size, cfg.tophat_size))
    tophat_v = cv2.morphologyEx(v_clahe, cv2.MORPH_TOPHAT, kernel)
    tophat_s = cv2.morphologyEx(sat, cv2.MORPH_TOPHAT, kernel)
    white = (tophat_v >= cfg.white_tophat_min) & (sat <= min(cfg.white_saturation_max, road_s + 50))
    yellow = ((hue >= cfg.yellow_hue_min) & (hue <= cfg.yellow_hue_max)
              & (tophat_s >= cfg.yellow_tophat_min) & (sat >= cfg.yellow_saturation_min))
    return np.where(white | yellow, 255, 0).astype(np.uint8)


def estimate_vanishing_point(value):
    # Diem tu = noi cac doan nghieng trai (m < 0) va nghieng phai (m > 0) cat nhau nhieu nhat.
    # Moi cap doan bo mot phieu co trong so = tich chieu dai; lay o 20x20 px nhieu phieu nhat.
    h, w = value.shape
    edges = cv2.Canny(cv2.GaussianBlur(value, (5, 5), 0), 50, 150)
    edges[:int(0.25 * h)] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=10)
    if lines is None:
        return w / 2, 0.45 * h, 0
    x1, y1, x2, y2 = lines.reshape(-1, 4).astype(np.float64).T
    dx, dy = x2 - x1, y2 - y1
    slope = np.divide(dy, dx, out=np.zeros_like(dy), where=dx != 0)
    keep = (dx != 0) & (np.abs(slope) >= 0.25) & (np.abs(slope) <= 3.0)
    x1, y1, slope, length = x1[keep], y1[keep], slope[keep], np.hypot(dx, dy)[keep]
    k = 1 / slope                    # x = k*y + b
    b = x1 - k * y1
    neg, pos = slope < 0, slope > 0
    if not neg.any() or not pos.any():
        return w / 2, 0.45 * h, 0
    with np.errstate(divide='ignore', invalid='ignore'):
        yi = (b[pos][None] - b[neg][:, None]) / (k[neg][:, None] - k[pos][None])
    xi = k[neg][:, None] * yi + b[neg][:, None]
    weight = length[neg][:, None] * length[pos][None]
    ok = np.isfinite(yi) & (yi >= 0.25 * h) & (yi <= 0.70 * h) & (xi >= 0.1 * w) & (xi <= 0.9 * w)
    if not ok.any():
        return w / 2, 0.45 * h, 0
    xi, yi, weight = xi[ok], yi[ok], weight[ok]
    hist, xe, ye = np.histogram2d(xi, yi, bins=(w // 20, h // 20), range=((0, w), (0, h)), weights=weight)
    ix, iy = np.unravel_index(cv2.GaussianBlur(hist.astype(np.float32), (5, 5), 0).argmax(), hist.shape)
    cx, cy = (xe[ix] + xe[ix + 1]) / 2, (ye[iy] + ye[iy + 1]) / 2
    near = (np.abs(xi - cx) < 40) & (np.abs(yi - cy) < 30)
    if not near.any():
        return w / 2, 0.45 * h, 0
    return (float(np.average(xi[near], weights=weight[near])),
            float(np.average(yi[near], weights=weight[near])), int(near.sum()))


def road_color_mask(hsv, lane_mask, cfg):
    # Pixel "giong mat duong": S gan S0, V gan V0 (trung vi + MAD trong cua so mau), cho phep bong do.
    # Vach son duoc gop vao roi dong hinh thai theo chieu ngang de lap khe giua cac vach.
    h, w = hsv.shape[:2]
    sat, val = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    y0, y1, x0, x1 = cfg.seed_window
    win = (slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w)))
    s0, v0 = np.median(sat[win]), np.median(val[win])
    s_tol = max(cfg.road_s_min, cfg.road_k * 1.4826 * np.median(np.abs(sat[win] - s0)))
    v_tol = max(cfg.road_v_min, cfg.road_k * 1.4826 * np.median(np.abs(val[win] - v0)))
    road = (np.abs(sat - s0) <= s_tol) & (val <= v0 + v_tol) & (val >= min(v0 - v_tol, cfg.shadow_ratio * v0))
    mask = cv2.morphologyEx((road * 255).astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = np.maximum(mask, lane_mask)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)))


def scan_road_rows(like, vp, cfg):
    # Quet tung hang tu day cua so mau: giu doan "giong mat duong" chua tam duong (tam cap nhat dan).
    # Di len: dung o duong chan troi, bien chi noi ra toi da perspective_grow px/hang, nam trong nem tu diem tu.
    # Di xuong: dung khi tam duong mat stop_rows hang lien tiep (nap capo).
    h, w = like.shape
    xv, yv = vp
    cols = np.arange(w)
    seed_row = int(cfg.seed_window[1] * h)
    lo, hi, ok = np.zeros(h, int), np.zeros(h, int), np.zeros(h, bool)
    for going_up, rows in ((True, range(seed_row, -1, -1)), (False, range(seed_row + 1, h))):
        c, miss, prev = w / 2.0, 0, None
        for y in rows:
            if going_up and y < yv - cfg.vp_margin:
                break
            r = like[y] > 0
            if going_up:
                r = r & (np.abs(cols - xv) <= (y - yv + cfg.vp_margin) * cfg.wedge)
            ci = int(round(c))
            if not r[ci]:
                near = np.flatnonzero(r[max(0, ci - 15):ci + 16])
                if len(near) == 0:
                    miss += 1
                    if miss >= cfg.stop_rows:
                        break
                    continue
                ci = max(0, ci - 15) + near[np.argmin(np.abs(near - min(ci, 15)))]
            left, right = np.flatnonzero(~r[:ci]), np.flatnonzero(~r[ci:])
            a = left[-1] + 1 if len(left) else 0
            b = ci + right[0] - 1 if len(right) else w - 1
            if going_up and prev is not None:
                a, b = max(a, prev[0] - cfg.perspective_grow), min(b, prev[1] + cfg.perspective_grow)
            if b - a < cfg.min_width:
                miss += 1
                if miss >= cfg.stop_rows:
                    break
                continue
            miss, prev = 0, (a, b)
            lo[y], hi[y], ok[y] = a, b, True
            c = 0.7 * c + 0.3 * (a + b) / 2
    return lo, hi, ok


def road_prior(rgb: np.ndarray, cfg: RoadConfig = RoadConfig()) -> tuple[tuple, int, np.ndarray]:
    """Diem tu (x, y), hang mep capo va mat na vach son, theo cach cua notebook."""
    hsv = rgb_to_hsv_opencv(rgb)
    h, w = hsv.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(hsv[..., 2])
    y0, y1, x0, x1 = cfg.seed_window
    road_s = float(np.median(hsv[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w), 1]))
    lane = lane_marking_mask(hsv, v_clahe, road_s, cfg)
    xv, yv, votes = estimate_vanishing_point(hsv[..., 2])
    if votes < cfg.vp_min_votes:
        xv, yv = w / 2, cfg.default_horizon_ratio * h
    like = road_color_mask(hsv, lane, cfg)
    _, _, ok = scan_road_rows(like, (xv, yv), cfg)
    hood_y = int(np.flatnonzero(ok).max()) if ok.any() else h - 1
    return (float(xv), float(yv)), hood_y, lane


# ---------------------------------------------------------------------------
# 3. Phan vung long duong bang GrabCut voi rang buoc hinh hoc
# ---------------------------------------------------------------------------

def _seed_component(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Chi giu thanh phan lien thong chua nhieu pixel cua so mau nhat."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8))
    ids, hits = np.unique(labels[seed & mask], return_counts=True)
    hits, ids = hits[ids > 0], ids[ids > 0]
    return labels == ids[np.argmax(hits)] if len(ids) else mask


def seed_rows(vp: tuple, hood_y: int, h: int, cfg: RoadConfig = RoadConfig()) -> slice:
    """Hang cua cua so mau mat duong: mac dinh gc_seed_rows, nhung luon nam duoi diem tu it nhat
    4% H va tren mep capo. Khi diem tu thap (y_vp ~ 0,67 H) cua so co dinh se rong va GrabCut that bai."""
    if not cfg.gc_seed_follow_vp:
        return slice(max(int(cfg.gc_seed_rows[0] * h), int(vp[1]) + 11), int(cfg.gc_seed_rows[1] * h))
    y0 = max(int(cfg.gc_seed_rows[0] * h), int(vp[1] + 0.04 * h))
    y1 = max(int(cfg.gc_seed_rows[1] * h), y0 + int(0.06 * h))
    y1 = min(y1, hood_y - 5, h)
    return slice(y0, max(y1, y0 + 1))


def lab_distance(rgb: np.ndarray, seed: np.ndarray, floor=(8.0, 3.0, 3.0)) -> np.ndarray:
    """Khoang cach chuan hoa trong Lab toi mau mat duong (trung vi + MAD cua cua so mau)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    sample = lab[seed]
    median = np.median(sample, axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(sample - median), axis=0), floor)
    return np.sqrt((((lab - median) / spread) ** 2).sum(axis=2))


def segment_road(rgb: np.ndarray, vp: tuple, hood_y: int, cfg: RoadConfig = RoadConfig()) -> np.ndarray:
    """GrabCut khoi tao bang rang buoc hinh hoc. Tra ve mat na bool H x W."""
    h, w = rgb.shape[:2]
    xv, yv = vp
    yy, xx = np.mgrid[0:h, 0:w]
    init = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    wedge = (yy > yv + 5) & (np.abs(xx - xv) <= (yy - yv) * cfg.gc_wedge)
    seed = np.zeros((h, w), bool)
    seed[seed_rows(vp, hood_y, h, cfg), int(cfg.gc_seed_cols[0] * w):int(cfg.gc_seed_cols[1] * w)] = True
    if cfg.gc_color_k is not None and seed.any():
        wedge &= lab_distance(rgb, seed) < cfg.gc_color_k
    init[wedge] = cv2.GC_PR_FGD
    init[:int(max(yv - cfg.gc_horizon_margin, 0))] = cv2.GC_BGD
    init[min(hood_y + cfg.gc_hood_margin, h):] = cv2.GC_BGD
    if cfg.gc_bottom_ratio > 0:
        init[int((1 - cfg.gc_bottom_ratio) * h):] = cv2.GC_BGD
    init[seed] = cv2.GC_FGD

    small = cv2.resize(init, None, fx=cfg.gc_scale, fy=cfg.gc_scale, interpolation=cv2.INTER_NEAREST)
    image = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (small.shape[1], small.shape[0]),
                       interpolation=cv2.INTER_AREA)
    if not (small == cv2.GC_FGD).any() or not np.isin(small, (cv2.GC_BGD, cv2.GC_PR_BGD)).any():
        return np.zeros((h, w), bool)
    votes = np.zeros(small.shape, np.int32)
    for k in range(max(1, cfg.gc_votes)):
        if cfg.gc_seed is not None:
            cv2.setRNGSeed(cfg.gc_seed + k)
        labels = small.copy()
        bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
        cv2.grabCut(image, labels, None, bgd, fgd, cfg.gc_iterations, cv2.GC_INIT_WITH_MASK)
        votes += np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD))
    road = (2 * votes > max(1, cfg.gc_votes)).astype(np.uint8)
    road = cv2.resize(road, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return _seed_component(road, seed)


# ---------------------------------------------------------------------------
# 4. Canny tren mat na -> Hough -> hai bien
# ---------------------------------------------------------------------------

def boundary_lines(mask: np.ndarray, vp: tuple, hood_y: int, cfg: RoadConfig = RoadConfig()):
    """Canny tren mat na long duong -> P-Hough rieng cho mep trai va mep phai -> WLS moi ben.

    Chi giu canh Canny nam sat diem ngoai cung ben trai / ben phai cua mat na o tung hang
    (bo vien tren do xe / chan troi, vien duoi do nap capo va doan cham mep khung anh),
    vi cac vien nay cung co do doc va lam Hough chon nham bien.
    """
    h, w = mask.shape
    # Lap lo (xe, vach son) de Canny chi con vien ngoai cua long duong
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    edges = canny_edges(cv2.GaussianBlur(filled, (5, 5), 1.2), cfg.canny_low, cfg.canny_high, cfg.canny_sigma)
    y_top = int(np.clip(vp[1] + 0.05 * h, 0.30 * h, 0.70 * h))
    y_bottom = max(min(h - 1, hood_y - 10), y_top + 40)
    near = {"left": np.zeros((h, w), bool), "right": np.zeros((h, w), bool)}
    margin = 4
    for y in range(y_top, y_bottom + 1):
        xs = np.flatnonzero(filled[y])
        if len(xs) == 0:
            continue
        if xs[0] > margin:
            near["left"][y, max(0, xs[0] - margin):xs[0] + margin + 1] = True
        if xs[-1] < w - 1 - margin:
            near["right"][y, xs[-1] - margin:min(w, xs[-1] + margin + 1)] = True
    result = []
    for side in ("left", "right"):
        side_edges = np.where(near[side], edges, 0).astype(np.uint8)
        segments = detect_hough_lines_p(side_edges, cfg.hough)
        left, right = split_left_right_segments(segments, w, cfg.hough)
        result.append(fit_boundary_line(left if side == "left" else right, y_bottom, y_top,
                                        side=side, width=w, config=cfg.hough))
    return edges, result[0], result[1]


def edge_map(rgb_enhanced: np.ndarray, vp: tuple) -> np.ndarray:
    """Anh canh Canny nguong tu dong (phan vi gradient duoi duong chan troi) tren anh da chuan hoa."""
    gray = cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2GRAY)
    region = np.zeros(gray.shape, bool)
    region[int(vp[1]):] = True
    edges, _ = canny_edges_auto(gray, region=region)
    return edges


def run_pipeline(rgb: np.ndarray, cfg: RoadConfig = RoadConfig(), with_edge_map: bool = False) -> RoadResult:
    resized = resize_rgb(rgb, cfg.width, cfg.height)
    enhanced, applied = normalize_contrast(resized, cfg)
    vp, hood_y, _ = road_prior(resized, cfg)
    road = segment_road(enhanced, vp, hood_y, cfg)
    edges, left, right = boundary_lines(road, vp, hood_y, cfg)
    return RoadResult(resized, enhanced, applied, vp, hood_y, (road * 255).astype(np.uint8), edges, left, right,
                      edge_map(enhanced, vp) if with_edge_map else None)


def draw_result(result: RoadResult) -> np.ndarray:
    """Ve long duong (xanh la), bien trai (do), bien phai (xanh duong), diem tu (vang). Tra ve RGB."""
    canvas = result.resized_rgb.copy()
    road = result.road_mask > 0
    canvas[road] = (0.6 * canvas[road] + np.array([0, 100, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(result.road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    for line, color in ((result.left_line, (255, 0, 0)), (result.right_line, (0, 0, 255))):
        if line is not None:
            cv2.line(canvas, line[0], line[1], color, 3, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (int(result.vp[0]), int(result.vp[1])), 6, (255, 255, 0), -1)
    return canvas


# ---------------------------------------------------------------------------
# Chay hang loat + danh gia
# ---------------------------------------------------------------------------

def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))


def process_dataset(input_dir="dataset/images", gt_dir="dataset/ground_truth", output_dir="outputs/pipeline_v2",
                    cfg: RoadConfig = RoadConfig(), save_edge_maps: bool = False):
    import pandas as pd
    from evaluation import evaluate_road_mask

    src, gts, out = Path(input_dir), Path(gt_dir), Path(output_dir)
    paths = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    rows = []
    for path in paths:
        rel = path.relative_to(src).with_suffix(".png")
        t0 = time.perf_counter()
        res = run_pipeline(_read_rgb(path), cfg, with_edge_map=save_edge_maps)
        runtime = (time.perf_counter() - t0) * 1000
        for sub, arr in (("road_mask", res.road_mask), ("mask_edges", res.mask_edges),
                         ("overlay", draw_result(res)), ("edge_map", res.edge_map)):
            if arr is None:
                continue
            target = out / sub / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(target)
        row = {"group": rel.parent.as_posix(), "image": path.stem, "contrast_applied": int(res.contrast_applied),
               "left_line": int(res.left_line is not None), "right_line": int(res.right_line is not None),
               "runtime_ms": runtime}
        gt_path = gts / rel
        if gt_path.is_file():
            gt = np.asarray(Image.open(gt_path).convert("L").resize((cfg.width, cfg.height), Image.Resampling.NEAREST))
            row.update(evaluate_road_mask(res.road_mask, gt))
            # Baseline tam thuong cung gioi han tren / duoi: to kin dai [y_vp + 2% H, mep capo]
            band = np.zeros_like(gt)
            band[int(np.clip(res.vp[1] + 0.02 * cfg.height, 0.3 * cfg.height, 0.65 * cfg.height)):res.hood_y + 1] = 255
            row["iou_band_baseline"] = evaluate_road_mask(band, gt)["iou"]
        rows.append(row)
        print(f"{rel.as_posix()}: IoU={row.get('iou', float('nan')):.3f}  {runtime:.0f} ms")
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    if "iou" in df:
        metrics = ["iou", "precision", "recall", "f1", "mae", "mae_evaluable", "success_mae15",
                   "iou_band_baseline", "contrast_applied", "runtime_ms"]
        summary = df.groupby("group")[metrics].mean()
        summary.loc["all"] = df[metrics].mean()
        summary.insert(0, "count", list(df.groupby("group").size()) + [len(df)])
        summary.to_csv(out / "summary.csv")
        print(summary.round(3).to_string())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline v2: chuan hoa tuong phan -> phan vung -> Canny -> Hough")
    parser.add_argument("--input", default="dataset/images")
    parser.add_argument("--gt", default="dataset/ground_truth")
    parser.add_argument("--output", default="outputs/pipeline_v2")
    parser.add_argument("--edge-maps", action="store_true", help="Luu them anh canh Canny nguong tu dong")
    args = parser.parse_args()
    process_dataset(args.input, args.gt, args.output, save_edge_maps=args.edge_maps)


if __name__ == "__main__":
    main()
