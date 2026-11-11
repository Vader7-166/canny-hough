"""PDF minh hoa tung buoc cua pipeline v3 (road_pipeline.py) tren 3 anh mau.

Moi trang la mot buoc: giai thich buoc do lam gi + anh ket qua tren 3 anh (moi hang mot anh).
Cac anh trung gian duoc tinh lai bang chinh cac ham trong road_pipeline.py; script kiem tra
mat na cuoi cung trung khop voi run_pipeline() de hinh minh hoa khong lech so voi code.

Chay tu thu muc goc: python scripts/pdf_tung_buoc_pipeline.py
Ket qua: baocao/Minh_hoa_tung_buoc_pipeline_v3.pdf
"""
from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import road_pipeline as rp  # noqa: E402
from canny import canny_edges_auto  # noqa: E402
from evaluation import evaluate_road_mask  # noqa: E402
from hough import detect_hough_lines_p  # noqa: E402
from preprocessing import resize_rgb, rgb_to_hsv_opencv  # noqa: E402

OUTPUT = ROOT / "baocao" / "Minh_hoa_tung_buoc_pipeline_v3.pdf"
SAMPLES = [("normal", "normal_11", "Sáng, đường cao tốc"),
           ("curved", "curved_06", "Ban đêm, khúc cua"),
           ("adverse", "adverse_08", "Trời mưa, nhiều xe")]

FONT_CANDIDATES = [
    ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),   # co du ky tu → ≥ ≤
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
]

RED, BLUE, GREEN, YELLOW, MAGENTA, CYAN = (230, 40, 40), (40, 90, 255), (0, 200, 0), (255, 220, 0), \
    (255, 0, 255), (0, 220, 220)


# ---------------------------------------------------------------------------
# Tinh lai cac buoc trung gian
# ---------------------------------------------------------------------------

@dataclass
class Steps:
    name: str
    label: str
    resized: np.ndarray
    enhanced: np.ndarray
    applied: bool
    median_low: float
    value: np.ndarray
    vp_edges: np.ndarray
    vp_segments: np.ndarray
    vp: tuple
    road_like: np.ndarray
    lane: np.ndarray
    rows: tuple
    hood_y: int
    trimap: np.ndarray
    seed: np.ndarray
    single_run: np.ndarray
    voted: np.ndarray
    grab: np.ndarray
    hood_edges: np.ndarray
    curve: np.ndarray | None
    after_hood: np.ndarray
    after_close: np.ndarray
    auto_edges: np.ndarray
    auto_thresholds: tuple
    ray_segments: np.ndarray
    rays: list
    chosen_rays: list
    final: np.ndarray
    mask_edges: np.ndarray
    side_edges: dict
    side_segments: dict
    left: tuple | None
    right: tuple | None
    gt: np.ndarray
    metrics: dict


def vp_internals(value: np.ndarray):
    """Lap lai phan dau cua rp.estimate_vanishing_point de ve canh va doan Hough."""
    h = value.shape[0]
    edges = cv2.Canny(cv2.GaussianBlur(value, (5, 5), 0), 50, 150)
    edges[:int(0.25 * h)] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=10)
    segs = np.zeros((0, 4)) if lines is None else lines.reshape(-1, 4).astype(np.float64)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    slope = np.divide(dy, dx, out=np.zeros_like(dy), where=dx != 0)
    keep = (dx != 0) & (np.abs(slope) >= 0.25) & (np.abs(slope) <= 3.0)
    return edges, segs[keep]


def grabcut_internals(rgb, vp, hood_y, cfg):
    """Lap lai rp.segment_road, giu them trimap, ket qua 1 lan chay va ket qua bo phieu."""
    h, w = rgb.shape[:2]
    xv, yv = vp
    yy, xx = np.mgrid[0:h, 0:w]
    init = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    wedge = (yy > yv + 5) & (np.abs(xx - xv) <= (yy - yv) * cfg.gc_wedge)
    seed = np.zeros((h, w), bool)
    seed[rp.seed_rows(vp, hood_y, h, cfg), int(cfg.gc_seed_cols[0] * w):int(cfg.gc_seed_cols[1] * w)] = True
    init[wedge] = cv2.GC_PR_FGD
    init[:int(max(yv - cfg.gc_horizon_margin, 0))] = cv2.GC_BGD
    init[min(hood_y + cfg.gc_hood_margin, h):] = cv2.GC_BGD
    init[int((1 - cfg.gc_bottom_ratio) * h):] = cv2.GC_BGD
    init[seed] = cv2.GC_FGD
    small = cv2.resize(init, None, fx=cfg.gc_scale, fy=cfg.gc_scale, interpolation=cv2.INTER_NEAREST)
    image = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (small.shape[1], small.shape[0]),
                       interpolation=cv2.INTER_AREA)
    votes = np.zeros(small.shape, np.int32)
    runs = []
    for k in range(cfg.gc_votes):
        cv2.setRNGSeed(cfg.gc_seed + k)
        labels = small.copy()
        cv2.grabCut(image, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), cfg.gc_iterations,
                    cv2.GC_INIT_WITH_MASK)
        fg = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD))
        runs.append(fg)
        votes += fg
    up = lambda m: cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0  # noqa: E731
    voted = up(2 * votes > cfg.gc_votes)
    return init, seed, up(runs[0]), voted, rp._seed_component(voted, seed)


def hood_edge_internals(rgb, vp, hood_y, cfg):
    """Lap lai phan loc canh ngang trong rp.detect_hood_curve (truoc khi khop parabol)."""
    h, w = rgb.shape[:2]
    seed = np.zeros((h, w), bool)
    rows = rp.seed_rows(vp, hood_y if hood_y < h - 1 else h, h, cfg)
    seed[rows, int(cfg.gc_seed_cols[0] * w):int(cfg.gc_seed_cols[1] * w)] = True
    dist = np.minimum(rp.lab_distance(rgb, seed), 30).astype(np.float32)
    b0, b1 = cfg.hood_band
    above, below = rp._band_mean(dist, -b1, -b0), rp._band_mean(dist, b0, b1)
    raw = rp._horizontal_edges(rgb, cfg.hood_edge_pct)
    kept = raw & (above < cfg.hood_road_k) & (below > above + cfg.hood_gap)
    kept[:rows.stop + 15] = False
    return raw, kept


def ray_segments(edges, vp, hood_y):
    """Lap lai phan chon doan Hough gan diem tu trong rp._vp_rays."""
    xv, yv = vp
    e = edges.copy()
    e[:int(yv) + 6] = 0
    e[max(hood_y - 3, 0):] = 0
    segs = cv2.HoughLinesP(e, 1, np.pi / 180, 25, minLineLength=35, maxLineGap=8)
    if segs is None:
        return np.zeros((0, 4)), np.zeros((0, 4))
    s = segs.reshape(-1, 4).astype(np.float64)
    x1, y1, x2, y2 = s.T
    dx, dy = x2 - x1, y2 - y1
    dist = np.abs(dx * (y1 - yv) - dy * (x1 - xv)) / np.hypot(dx, dy)
    phi = np.degrees(np.arctan2((x1 + x2) / 2 - xv, (y1 + y2) / 2 - yv))
    near = (dist < 22.0) & (np.abs(phi) < 86) & (np.abs(phi) > 4)
    return s, s[near]


def compute(group: str, name: str, label: str, cfg=rp.RoadConfig()) -> Steps:
    src = next((ROOT / "dataset" / "images" / group).glob(f"{name}.*"))
    resized = resize_rgb(rp._read_rgb(src), cfg.width, cfg.height)
    h, w = resized.shape[:2]
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    median_low = float(np.median(lab[h // 2:, :, 0]))
    enhanced, applied = rp.normalize_contrast(resized, cfg)

    # 3. Hinh hoc: diem tu, mau mat duong, quet hang -> mep capo
    hsv = rgb_to_hsv_opencv(resized)
    vp_edges, vp_segs = vp_internals(hsv[..., 2])
    vp, hood_y, lane = rp.road_prior(resized, cfg)
    v_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(hsv[..., 2])
    y0, y1, x0, x1 = cfg.seed_window
    road_s = float(np.median(hsv[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w), 1]))
    like = rp.road_color_mask(hsv, rp.lane_marking_mask(hsv, v_clahe, road_s, cfg), cfg)
    rows = rp.scan_road_rows(like, vp, cfg)

    # 4. GrabCut
    trimap, seed, single, voted, grab = grabcut_internals(enhanced, vp, hood_y, cfg)
    assert np.array_equal(grab, rp.segment_road(enhanced, vp, hood_y, cfg)), "GrabCut lap lai khong khop"

    # 5. Hau xu ly
    yy, xx = np.mgrid[0:h, 0:w]
    _, hood_edges = hood_edge_internals(enhanced, vp, hood_y, cfg)
    curve = rp.detect_hood_curve(enhanced, vp, hood_y, cfg)
    after_hood = grab & (yy <= curve[None] - 1) if curve is not None else grab.copy()
    after_close = rp.refine_mask(grab, enhanced, vp, hood_y, None,
                                 rp.RoadConfig(ray_cut=False))
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    region = np.zeros(gray.shape, bool)
    region[int(vp[1]):] = True
    auto_edges, thr = canny_edges_auto(gray, region=region)
    assert np.array_equal(auto_edges, rp.edge_map(enhanced, vp))
    _, near_segs = ray_segments(auto_edges, vp, hood_y)
    rays = rp._vp_rays(auto_edges, vp, hood_y, cfg.ray_min_support)
    lab_f = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB).astype(np.float32)
    rays = [(phi, sup, rp._ray_color_step(lab_f, vp, hood_y, phi)) for phi, sup in rays]
    chosen = []
    for sign in (-1, 1):
        cand = [phi for phi, _, de in rays if np.sign(phi) == sign and abs(phi) >= cfg.ray_min_angle
                and de >= cfg.ray_min_delta_e]
        if cand:
            chosen.append(min(cand, key=abs))
    final = rp.refine_mask(grab, enhanced, vp, hood_y, auto_edges, cfg)

    # 6. Bien trai / phai
    mask_edges, left, right = rp.boundary_lines(final, vp, hood_y, cfg)
    ref = rp.run_pipeline(rp._read_rgb(src), cfg)
    assert np.array_equal(ref.road_mask > 0, final), "mat na cuoi khong khop run_pipeline"
    contours, _ = cv2.findContours(final.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    y_top = int(np.clip(vp[1] + 0.05 * h, 0.30 * h, 0.70 * h))
    y_bottom = max(min(h - 1, hood_y - 10), y_top + 40)
    near = {"left": np.zeros((h, w), bool), "right": np.zeros((h, w), bool)}
    for y in range(y_top, y_bottom + 1):
        xs = np.flatnonzero(filled[y])
        if len(xs) == 0:
            continue
        if xs[0] > 4:
            near["left"][y, max(0, xs[0] - 4):xs[0] + 5] = True
        if xs[-1] < w - 5:
            near["right"][y, xs[-1] - 4:min(w, xs[-1] + 5)] = True
    side_edges = {k: np.where(v, mask_edges, 0).astype(np.uint8) for k, v in near.items()}
    side_segments = {k: detect_hough_lines_p(e, cfg.hough) for k, e in side_edges.items()}

    gt = np.asarray(Image.open(ROOT / "dataset" / "ground_truth" / group / f"{name}.png").convert("L")
                    .resize((w, h), Image.Resampling.NEAREST))
    metrics = evaluate_road_mask((final * 255).astype(np.uint8), gt)
    return Steps(name, label, resized, enhanced, applied, median_low, hsv[..., 2], vp_edges, vp_segs, vp, like, lane,
                 rows, hood_y, trimap, seed, single, voted, grab, hood_edges, curve, after_hood, after_close,
                 auto_edges, thr, near_segs, rays, chosen, final, mask_edges, side_edges, side_segments, left, right,
                 gt, metrics)


# ---------------------------------------------------------------------------
# Ve anh minh hoa
# ---------------------------------------------------------------------------

def rgb3(a):
    return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB) if a.ndim == 2 else a.copy()


def tint(img, mask, color, alpha=0.45):
    out = img.copy()
    out[mask] = ((1 - alpha) * out[mask] + alpha * np.array(color)).astype(np.uint8)
    return out


def outline(img, mask, color, thick=2):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thick, lineType=cv2.LINE_AA)
    return img


def draw_vp(img, vp, horizon=True):
    if horizon:
        cv2.line(img, (0, int(vp[1])), (img.shape[1], int(vp[1])), YELLOW, 2, cv2.LINE_AA)
    cv2.circle(img, (int(vp[0]), int(vp[1])), 9, YELLOW, -1, cv2.LINE_AA)
    cv2.circle(img, (int(vp[0]), int(vp[1])), 9, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def draw_segments(img, segs, color=None, thick=2):
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4).astype(int):
        c = color or (RED if (x2 - x1) * (y2 - y1) < 0 else BLUE)
        cv2.line(img, (x1, y1), (x2, y2), c, thick, cv2.LINE_AA)
    return img


def panels(s: Steps) -> dict[str, list[tuple[np.ndarray, str]]]:
    h, w = s.resized.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    gt = s.gt > 127
    p: dict[str, list] = {}

    # Buoc 1-2
    p["contrast"] = [
        (s.resized, "Ảnh sau resize 960 × 540"),
        (s.enhanced, "Sau chuẩn hóa" if s.applied else "Không chuẩn hóa (ảnh đủ sáng)"),
        (rgb3(cv2.cvtColor(s.enhanced, cv2.COLOR_RGB2LAB)[..., 0]),
         f"Kênh L dùng cho quyết định: trung vị nửa dưới = {s.median_low:.0f}"),
    ]

    # Buoc 3a diem tu
    seg_img = draw_segments(rgb3(s.vp_edges // 3), s.vp_segments)
    p["vp"] = [
        (rgb3(s.value), "Kênh V (độ sáng) của HSV"),
        (rgb3(s.vp_edges), "Canny (50, 150) trên kênh V, bỏ 25% trên"),
        (draw_vp(seg_img, s.vp), f"Đoạn Hough (đỏ: nghiêng trái, lam: phải) → điểm tụ y = {s.vp[1]:.0f}"),
    ]

    # Buoc 3b mep capo
    lo, hi, ok = s.rows
    scan = s.resized.copy()
    for y in np.flatnonzero(ok)[::3]:
        cv2.line(scan, (int(lo[y]), int(y)), (int(hi[y]), int(y)), GREEN, 1)
    cv2.line(scan, (0, s.hood_y), (w, s.hood_y), MAGENTA, 3, cv2.LINE_AA)
    y0, y1, x0, x1 = rp.RoadConfig().seed_window
    cv2.rectangle(scan, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), YELLOW, 2)
    like = tint(s.resized, s.road_like > 0, GREEN, 0.55)
    like = tint(like, s.lane > 0, RED, 0.8)
    p["hood"] = [
        (like, "Giống màu mặt đường (xanh), vạch sơn (đỏ)"),
        (scan, f"Quét hàng → mép capo y = {s.hood_y}" + (" (đáy ảnh)" if s.hood_y >= h - 1 else "")),
    ]

    # Buoc 4 GrabCut
    tri = s.enhanced.copy()
    colors = {cv2.GC_BGD: (40, 40, 40), cv2.GC_PR_BGD: (150, 110, 60), cv2.GC_PR_FGD: (90, 200, 90),
              cv2.GC_FGD: (0, 255, 0)}
    alpha = {cv2.GC_BGD: 0.75, cv2.GC_PR_BGD: 0.35, cv2.GC_PR_FGD: 0.45, cv2.GC_FGD: 0.9}
    for k, c in colors.items():
        tri = tint(tri, s.trimap == k, c, alpha[k])
    draw_vp(tri, s.vp, horizon=False)
    vote = tint(s.enhanced, s.voted & ~s.grab, RED, 0.6)
    vote = tint(vote, s.grab, GREEN, 0.5)
    p["grabcut"] = [
        (tri, "Đánh dấu ban đầu (xem chú giải)"),
        (tint(s.enhanced, s.single_run, GREEN, 0.5), "Một lần chạy GrabCut (seed 0)"),
        (vote, "Đa số 3 lần chạy; đỏ: mảng rời bị loại"),
    ]

    # Buoc 5a capo cong + dong hinh thai
    hood_img = s.enhanced.copy()
    hood_img[cv2.dilate(s.hood_edges.astype(np.uint8), np.ones((3, 3))) > 0] = CYAN
    if s.curve is not None:
        pts = np.stack([np.arange(w), np.clip(s.curve, 0, h - 1)], 1).astype(np.int32)
        cv2.polylines(hood_img, [pts], False, MAGENTA, 3, cv2.LINE_AA)
    close = tint(s.enhanced, s.after_close & s.grab, GREEN, 0.45)
    close = tint(close, s.grab & ~s.after_close, RED, 0.7)
    close = tint(close, s.after_close & ~s.grab, BLUE, 0.7)
    p["refine1"] = [
        (hood_img, "Cạnh ngang (lam ngọc) → parabol capo (tím)"
         if s.curve is not None else "Cạnh ngang: không đủ tin cậy → không cắt"),
        (close, "Sau cắt capo + đóng. Đỏ: cắt, lam: lấp"),
    ]

    # Buoc 5b cat theo tia
    ray_img = draw_segments(rgb3(s.auto_edges // 3), s.ray_segments, color=YELLOW)
    for phi, sup, de in s.rays:
        ok_ray = abs(phi) >= 62 and de >= 30
        x_end = s.vp[0] + np.tan(np.radians(phi)) * (h - s.vp[1])
        color = GREEN if phi in s.chosen_rays else ((255, 140, 0) if ok_ray else (120, 120, 120))
        cv2.line(ray_img, (int(s.vp[0]), int(s.vp[1])), (int(x_end), h), color, 3 if phi in s.chosen_rays else 1,
                 cv2.LINE_AA)
    draw_vp(ray_img, s.vp, horizon=False)
    cut = tint(s.enhanced, s.final, GREEN, 0.45)
    cut = tint(cut, s.after_close & ~s.final, RED, 0.7)
    p["refine2"] = [
        (rgb3(s.auto_edges), f"Canny ngưỡng phân vị: thấp {s.auto_thresholds[0]:.0f}, cao {s.auto_thresholds[1]:.0f}"),
        (ray_img, f"Đoạn Hough gần điểm tụ (vàng); {len(s.chosen_rays)} tia được chọn (xanh)"),
        (cut, "Mặt nạ cuối. Đỏ: phần bị cắt theo tia"),
    ]

    # Buoc 6 bien
    thick = lambda m: cv2.dilate((m > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # noqa: E731 (de nhin)
    side = rgb3(np.where(thick(s.mask_edges), 90, 0).astype(np.uint8))
    side[thick(s.side_edges["left"])] = RED
    side[thick(s.side_edges["right"])] = BLUE
    seg = rgb3(np.zeros((h, w), np.uint8))
    seg = draw_segments(seg, s.side_segments["left"], color=RED, thick=4)
    seg = draw_segments(seg, s.side_segments["right"], color=BLUE, thick=4)
    final = tint(s.resized, s.final, GREEN, 0.45)
    outline(final, gt, MAGENTA, 3)
    for line, c in ((s.left, RED), (s.right, BLUE)):
        if line is not None:
            cv2.line(seg, line[0], line[1], (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(final, line[0], line[1], c, 4, cv2.LINE_AA)
    draw_vp(final, s.vp, horizon=False)
    p["boundary"] = [
        (side, "Canny trên mặt nạ; giữ viền sát mép trái (đỏ) / phải (lam)"),
        (seg, "Đoạn Hough mỗi bên; đường WLS (trắng)"),
        (final, f"Kết quả: IoU {s.metrics['iou']:.2f}, F1 {s.metrics['f1']:.2f}. Tím: ground truth"),
    ]
    return p


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

PAGES = [
    ("contrast", "Bước 1–2. Resize và chuẩn hóa tương phản",
     "Ảnh được resize về 960 × 540. Sau đó chuyển sang không gian Lab và lấy trung vị kênh độ sáng L ở nửa dưới ảnh. "
     "<b>Nếu trung vị &lt; 50 (ảnh tối)</b>: áp gamma để kéo độ sáng trung vị về khoảng 0,45, rồi áp CLAHE "
     "(cân bằng histogram cục bộ, clip 2,0, lưới 8 × 8) <b>chỉ trên kênh L</b>; hai kênh màu a, b giữ nguyên nên "
     "màu mặt đường không bị lệch. Ảnh đủ sáng giữ nguyên, vì CLAHE làm đổi màu cục bộ và làm GrabCut kém đi. "
     "Ảnh đã chuẩn hóa được dùng cho GrabCut, hậu xử lý và Canny ngưỡng phân vị; bước 3 dùng ảnh gốc.",
     "Hàm: normalize_contrast. Áp dụng cho 13/50 ảnh."),
    ("vp", "Bước 3a. Ước lượng điểm tụ (Canny lần 1)",
     "Trong ảnh, hai mép đường song song ngoài thực tế sẽ chụm lại tại <b>điểm tụ</b> gần đường chân trời. "
     "Cách tìm: chuyển ảnh sang HSV, lấy kênh V (V = max(R, G, B), tức độ sáng), làm mờ Gauss rồi chạy "
     "<b>Canny (50, 150)</b>; bỏ 25% hàng trên cùng. HoughLinesP (ngưỡng 40, dài ≥ 40 px, khe ≤ 10 px) cho các "
     "đoạn thẳng; chỉ giữ đoạn có độ dốc 0,25–3. Mỗi cặp (đoạn nghiêng trái, đoạn nghiêng phải) được kéo dài tới "
     "giao điểm và bỏ một phiếu, trọng số = tích chiều dài. Ô 20 × 20 px nhiều phiếu nhất là điểm tụ (chấm vàng). "
     "Hàng ngang qua điểm tụ được coi là <b>đường chân trời</b> (vạch vàng). Nếu có dưới 8 phiếu, dùng mặc định "
     "(giữa ảnh, 45% H).",
     "Hàm: estimate_vanishing_point (trong road_prior). Điểm tụ dùng cho GrabCut, cắt theo tia và tìm biên."),
    ("hood", "Bước 3b. Ước lượng mép capo",
     "Từ cửa sổ mẫu ngay trước xe (khung vàng, 62–70% H, 42–58% W) lấy trung vị và độ phân tán của S, V. "
     "Pixel có S, V gần với mẫu (cho phép tối hơn do bóng đổ) được coi là \"giống mặt đường\"; vạch sơn tìm bằng "
     "top-hat được gộp vào rồi đóng hình thái để lấp khe. Sau đó quét từng hàng từ cửa sổ mẫu: đi lên tới đường "
     "chân trời, đi xuống cho tới khi mất mặt đường 6 hàng liên tiếp (gặp nắp capo). <b>Hàng thấp nhất còn "
     "mặt đường = mép capo</b> (vạch tím). Nếu quét chạm đáy ảnh (y = 539) tức là không tìm thấy capo; khi đó GrabCut dựa vào 3% hàng cuối luôn là nền, và bước 5a có thể cắt thêm. Mặt nạ màu ở đây chỉ dùng để tìm mép capo, không dùng làm kết quả.",
     "Hàm: lane_marking_mask, road_color_mask, scan_road_rows (trong road_prior), chuyển từ notebook."),
    ("grabcut", "Bước 4. Phân vùng lòng đường bằng GrabCut",
     "GrabCut tách ảnh thành \"đường\" và \"nền\" bằng hai mô hình màu (mỗi mô hình 5 Gauss) và phép cắt đồ thị: "
     "pixel nhận nhãn phù hợp với màu của nó, ranh giới ưu tiên đi qua chỗ đổi màu mạnh. Bản đánh dấu ban đầu "
     "tạo tự động từ bước 3: <b>xám đậm</b> = nền chắc chắn (trên chân trời − 15 px, dưới mép capo + 4 px, 3% hàng "
     "cuối); <b>xanh đậm</b> = đường chắc chắn (cửa sổ trước xe, đi theo điểm tụ); <b>xanh nhạt</b> = có thể là "
     "đường (nêm từ điểm tụ, |x − x_vp| ≤ 3·(y − y_vp)); <b>nâu nhạt</b> = có thể là nền. GrabCut "
     "chạy ở nửa độ phân giải, 5 vòng lặp, <b>3 lần với seed 0, 1, 2</b> rồi lấy đa số (chống lần chạy hỏng do "
     "k-means ngẫu nhiên). Cuối cùng chỉ giữ vùng liền khối chứa cửa sổ trước xe.",
     "Hàm: segment_road. Đây là bước quyết định vùng đường."),
    ("refine1", "Bước 5a. Hậu xử lý: cắt theo mép capo cong, đóng hình thái",
     "<b>Mép capo cong</b>: lấy cạnh ngang mạnh (|gy| trên phân vị 92% và |gy| ≥ 1,5·|gx|, với gx, gy là gradient Sobel ngang, dọc) "
     "mà dải 4–14 px phía trên giống màu mặt đường (khoảng cách Lab tới mẫu &lt; 4) còn dải phía dưới khác hơn ít "
     "nhất 3. Khớp parabol y = y0 + a·u² + t·u phủ nhiều cạnh nhất; chỉ cắt khi parabol trùng cạnh ở ≥ 40% số cột "
     "(12/50 ảnh). <b>Đóng hình thái</b> bằng ellipse 141 px để lấp chỗ xe che và khuyết ở mép, nhưng không cho "
     "mặt nạ vượt quá mép capo hoặc hàng thấp nhất đã có.",
     "Hàm: detect_hood_curve, refine_mask."),
    ("refine2", "Bước 5b. Hậu xử lý: cắt theo tia biên (Canny lần 2 + Hough)",
     "Chạy Canny trên ảnh xám đã chuẩn hóa với <b>ngưỡng theo phân vị gradient</b>: ngưỡng cao = phân vị 90% độ "
     "lớn gradient dưới đường chân trời, ngưỡng thấp = 0,4 × ngưỡng cao. HoughLinesP (ngưỡng 25, dài ≥ 35, khe ≤ 8) "
     "rồi chỉ giữ đoạn mà đường kéo dài đi qua gần điểm tụ (&lt; 22 px). Các đoạn bỏ phiếu theo góc tạo thành tia "
     "qua điểm tụ. Mỗi bên chỉ cắt bằng tia trong cùng thỏa cả ba điều kiện: nghiêng ≥ 62° so với phương thẳng đứng, "
     "được ủng hộ ≥ 60 px, màu Lab hai bên tia lệch ΔE ≥ 30. Tia xám: không đạt; cam: đạt nhưng không phải tia "
     "trong cùng; xanh: được dùng để cắt. Phần mặt nạ nằm ngoài tia bị bỏ.",
     "Hàm: edge_map, _vp_rays, _ray_color_step, refine_mask. Kết quả của bước này là vùng đường để chấm IoU."),
    ("boundary", "Bước 6. Tìm hai biên trái/phải (Canny lần 3 + Hough)",
     "Mặt nạ được lấp lỗ, làm mờ, rồi chạy <b>Canny (50, 150) trên mặt nạ</b> để lấy đường viền vùng đường. "
     "Ở mỗi hàng từ chân trời + 5% H tới mép capo − 10 px, chỉ giữ cạnh trong ±4 px quanh điểm ngoài cùng bên trái và "
     "bên phải; bỏ hàng mà mặt nạ chạm mép khung ảnh. P-Hough chạy riêng cho mỗi bên (|độ dốc| ≥ 0,10), sau đó chọn "
     "cụm đồng thuận và khớp WLS (hàm của Phần 4). Hai đường biên chỉ dùng để vẽ và đo MAE, <b>không thay đổi mặt nạ</b>.",
     "Hàm: boundary_lines. Thiếu biên ở một bên thường do mặt nạ chạm mép khung ảnh ở bên đó."),
]

LEGENDS = {
    "grabcut": [((40, 40, 40), "Nền chắc chắn"), ((150, 110, 60), "Có thể là nền"),
                ((90, 200, 90), "Có thể là đường"), ((0, 255, 0), "Đường chắc chắn")],
}


def register_fonts():
    for regular, bold in FONT_CANDIDATES:
        if Path(regular).is_file() and Path(bold).is_file():
            pdfmetrics.registerFont(TTFont("VN", regular))
            pdfmetrics.registerFont(TTFont("VN-Bold", bold))
            pdfmetrics.registerFontFamily("VN", normal="VN", bold="VN-Bold", italic="VN", boldItalic="VN-Bold")
            return
    raise FileNotFoundError("Khong tim thay font TTF co dau tieng Viet")


def para(c, text, x, y_top, width, size=10, color="#222222", leading=None):
    style = ParagraphStyle("p", fontName="VN", fontSize=size, leading=leading or size * 1.35,
                           textColor=HexColor(color))
    p = Paragraph(text, style)
    _, h = p.wrap(width, 1000)
    p.drawOn(c, x, y_top - h)
    return h


def image_reader(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=88)   # JPEG de PDF nhe (PNG ~20 MB)
    buf.seek(0)
    return ImageReader(buf)


def draw_page(c, page_w, page_h, number, title, text, note, rows, legend=None):
    m = 1.2 * cm
    c.setFillColor(HexColor("#1F4E79"))
    c.rect(0, page_h - 1.5 * cm, page_w, 1.5 * cm, stroke=0, fill=1)
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("VN-Bold", 15)
    c.drawString(m, page_h - 1.0 * cm, title)
    c.setFont("VN", 9)
    c.drawRightString(page_w - m, page_h - 1.0 * cm, f"Trang {number}")
    y = page_h - 1.5 * cm - 0.35 * cm
    y -= para(c, text, m, y, page_w - 2 * m, size=9.6)
    y -= 0.1 * cm
    y -= para(c, f"<b>Code:</b> {note}", m, y, page_w - 2 * m, size=8.6, color="#555555")
    if legend:
        y -= 0.15 * cm
        x = m
        for color, label in legend:
            c.setFillColorRGB(*[v / 255 for v in color])
            c.rect(x, y - 0.32 * cm, 0.4 * cm, 0.3 * cm, stroke=0, fill=1)
            c.setFillColor(HexColor("#222222"))
            c.setFont("VN", 8.6)
            c.drawString(x + 0.5 * cm, y - 0.28 * cm, label)
            x += 0.9 * cm + c.stringWidth(label, "VN", 8.6)
        y -= 0.4 * cm
    y -= 0.25 * cm

    n_rows, n_cols = len(rows), max(len(r[1]) for r in rows)
    label_w = 2.6 * cm
    gap = 0.2 * cm
    cap_h = 0.42 * cm
    avail_w = page_w - 2 * m - label_w
    avail_h = y - m
    img_w = min((avail_w - (n_cols - 1) * gap) / n_cols, ((avail_h / n_rows) - cap_h - gap) * 16 / 9)
    img_h = img_w * 9 / 16
    total_w = label_w + n_cols * img_w + (n_cols - 1) * gap
    x0 = (page_w - total_w) / 2
    for i, (row_label, items) in enumerate(rows):
        top = y - i * (img_h + cap_h + gap)
        para(c, row_label, x0, top - cap_h - img_h / 2 + 0.6 * cm, label_w - 0.25 * cm, size=8.6, color="#1F4E79")
        for j, (arr, caption) in enumerate(items):
            x = x0 + label_w + j * (img_w + gap)
            c.setFillColor(HexColor("#333333"))
            c.setFont("VN", 7.4)
            cap = caption
            while c.stringWidth(cap, "VN", 7.4) > img_w and len(cap) > 4:
                cap = cap[:-2]
            if cap != caption:
                cap = cap[:-1] + "…"
            c.drawString(x, top - cap_h + 0.12 * cm, cap)
            c.drawImage(image_reader(arr), x, top - cap_h - img_h, img_w, img_h)


def draw_cover(c, page_w, page_h, steps: list[Steps]):
    m = 1.6 * cm
    c.setFillColor(HexColor("#1F4E79"))
    c.rect(0, page_h - 3.2 * cm, page_w, 3.2 * cm, stroke=0, fill=1)
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("VN-Bold", 22)
    c.drawString(m, page_h - 1.7 * cm, "Minh họa từng bước của pipeline v3")
    c.setFont("VN", 11)
    c.drawString(m, page_h - 2.5 * cm, "Phát hiện lòng đường: tiền xử lý → GrabCut → hậu xử lý → Canny + Hough (road_pipeline.py)")
    y = page_h - 4.0 * cm
    flow = (
        "<b>Luồng xử lý</b><br/>"
        "1. Resize 960 × 540 &nbsp;→&nbsp; 2. Chuẩn hóa tương phản (chỉ ảnh tối) &nbsp;→&nbsp; "
        "3a. Điểm tụ (Canny lần 1 + Hough) &nbsp;→&nbsp; 3b. Mép capo (màu S, V) &nbsp;→&nbsp; "
        "4. GrabCut &nbsp;→&nbsp; 5a. Cắt theo mép capo cong, đóng hình thái &nbsp;→&nbsp; "
        "5b. Cắt theo tia biên (Canny lần 2 + Hough) &nbsp;→&nbsp; <b>vùng đường</b> &nbsp;→&nbsp; "
        "6. Hai biên trái/phải (Canny lần 3 trên mặt nạ + Hough)<br/><br/>"
        "Mỗi trang sau là một bước: phần trên giải thích bước đó làm gì, phần dưới là ảnh kết quả trên ba ảnh mẫu "
        "(mỗi hàng một ảnh). Các ảnh trung gian được tính lại bằng chính các hàm trong road_pipeline.py; script kiểm "
        "tra mặt nạ cuối trùng khớp với run_pipeline(). Quy ước màu chung: <b>xanh lá</b> = vùng đường dự đoán, "
        "<b>đỏ</b> = biên trái hoặc phần bị cắt bỏ, <b>xanh lam</b> = biên phải hoặc phần được lấp, "
        "<b>vàng</b> = điểm tụ / đường chân trời, <b>tím</b> = mép capo hoặc ground truth."
    )
    y -= para(c, flow, m, y, page_w - 2 * m, size=11)
    y -= 0.6 * cm
    w = (page_w - 2 * m - 2 * 0.4 * cm) / 3
    for i, s in enumerate(steps):
        x = m + i * (w + 0.4 * cm)
        h = w * 9 / 16
        c.drawImage(image_reader(s.resized), x, y - h, w, h)
        para(c, f"<b>{s.name}</b> · {s.label}<br/>Chuẩn hóa tương phản: {'có' if s.applied else 'không'} · "
                f"IoU {s.metrics['iou']:.2f} · F1 {s.metrics['f1']:.2f}", x, y - h - 0.15 * cm, w, size=9.5)
    y -= w * 9 / 16 + 1.6 * cm
    para(c, "Kết quả trên toàn bộ 50 ảnh (outputs/pipeline_v3/summary.csv): IoU 0,689 · F1 0,807 · 25/50 ảnh IoU ≥ 0,7 "
            "· khoảng 6,2 s/ảnh. Pipeline gốc (Phần 2–4): IoU 0,174.", m, y, page_w - 2 * m, size=10, color="#444444")


def main():
    register_fonts()
    steps = []
    for group, name, label in SAMPLES:
        print("dang tinh", name)
        steps.append(compute(group, name, label))
    all_panels = [panels(s) for s in steps]
    page_w, page_h = landscape(A4)
    c = canvas.Canvas(str(OUTPUT), pagesize=(page_w, page_h))
    c.setTitle("Minh họa từng bước pipeline v3")
    draw_cover(c, page_w, page_h, steps)
    c.showPage()
    for number, (key, title, text, note) in enumerate(PAGES, start=2):
        rows = [(f"<b>{s.name}</b><br/>{s.label}", p[key]) for s, p in zip(steps, all_panels)]
        draw_page(c, page_w, page_h, number, title, text, note, rows, LEGENDS.get(key))
        c.showPage()
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
