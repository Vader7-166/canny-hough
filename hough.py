"""Module bien doi Hough va hau xu ly phat hien bien duong (Phan 4).

Cung cap cac ham cat ROI hinh thang, HoughLinesP, loc do doc,
phan cum trai/phai va hoi quy tuyen tinh co trong so de xac dinh hai bien duong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


@dataclass(frozen=True)
class HoughConfig:
    """Cau hinh tham so cho ROI hinh thang va bien doi Hough."""

    # Tham so ROI hinh thang (ty le so voi W va H cua anh)
    roi_top_width_ratio: float = 0.35      # Be rong day tren ROI (~35% W)
    roi_bottom_width_ratio: float = 0.98   # Be rong day duoi ROI (~98% W)
    roi_top_y_ratio: float = 0.48          # Do cao canh tren ROI (~48% H, sat duong chan troi thuc te)
    roi_bottom_y_ratio: float = 0.98       # Do cao canh duoi ROI (~98% H)

    # Tham so HoughLinesP
    rho: float = 1.0                       # Do phan giai ban kinh (pixel)
    theta: float = np.pi / 180.0           # Do phan giai goc (rad, 1 do)
    threshold: int = 25                    # Nguong bo phieu tich luy toi thieu
    min_line_length: float = 25.0          # Chieu dai doan thang toi thieu (pixel)
    max_line_gap: float = 25.0             # Khoang cach toi da de noi cac doan dut quang (pixel)

    # Loc do doc m = dy / dx trong he toa do pixel (y huong xuong)
    # Bien trai: m < 0, bien phai: m > 0
    min_abs_slope: float = 0.30            # Loai bo cac duong qua nam ngang (~17 do)
    max_abs_slope: float = 4.00            # Loai bo cac duong gan nhu thang dung (~76 do)

    # Dung sai vuong goc khi chon cum dong thuan, ty le theo be rong anh.
    fit_distance_ratio: float = 0.02

    def __post_init__(self) -> None:
        values = np.array(list(self.__dict__.values()), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Tham so Hough phai huu han")
        if not (0.0 < self.roi_top_y_ratio < self.roi_bottom_y_ratio <= 1.0):
            raise ValueError("roi_top_y_ratio va roi_bottom_y_ratio khong hop le")
        if not (0.0 < self.roi_top_width_ratio <= self.roi_bottom_width_ratio <= 1.0):
            raise ValueError("Be rong ROI phai thoa 0 < top <= bottom <= 1")
        if not (0.0 < self.min_abs_slope < self.max_abs_slope):
            raise ValueError("Do doc phai thoa 0 < min < max")
        if self.rho <= 0 or not (0 < self.theta <= np.pi) or self.threshold <= 0:
            raise ValueError("rho, theta va threshold khong hop le")
        if self.min_line_length <= 0 or self.max_line_gap < 0:
            raise ValueError("Chieu dai phai duong, khoang dut doan khong am")
        if not (0 < self.fit_distance_ratio < 1):
            raise ValueError("fit_distance_ratio phai nam trong (0, 1)")


@dataclass(frozen=True)
class HoughResult:
    """Ket qua phat hien bien duong tu module Hough."""

    left_line: tuple[tuple[int, int], tuple[int, int]] | None    # ((x_bottom, y_bottom), (x_top, y_top))
    right_line: tuple[tuple[int, int], tuple[int, int]] | None   # ((x_bottom, y_bottom), (x_top, y_top))
    polygon: np.ndarray | None                                   # Toa do da giac (4, 2) mat duong giua 2 bien
    raw_lines: np.ndarray                                        # Mang tat ca doan thang HoughLinesP tim thay (N, 4)
    left_segments: np.ndarray                                    # Ung vien trai sau loc slope; fit chi dung inlier (K, 4)
    right_segments: np.ndarray                                   # Ung vien phai sau loc slope; fit chi dung inlier (M, 4)
    roi_mask: np.ndarray                                         # Mat na ROI da giac (H, W)


def get_trapezoid_vertices(height: int, width: int, config: HoughConfig) -> np.ndarray:
    """Tinh toa do 4 dinh cua ROI hinh thang can doi qua truc giua."""
    if height < 2 or width < 2:
        raise ValueError("Anh phai co it nhat 2 hang va 2 cot")
    center_x = width / 2.0
    top_y = min(height - 1, int(height * config.roi_top_y_ratio))
    bottom_y = min(height - 1, int(height * config.roi_bottom_y_ratio))

    half_top_w = (width * config.roi_top_width_ratio) / 2.0
    half_bottom_w = (width * config.roi_bottom_width_ratio) / 2.0

    top_left = (int(max(0, center_x - half_top_w)), top_y)
    top_right = (int(min(width - 1, center_x + half_top_w)), top_y)
    bottom_right = (int(min(width - 1, center_x + half_bottom_w)), bottom_y)
    bottom_left = (int(max(0, center_x - half_bottom_w)), bottom_y)

    return np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)


def apply_roi(edges: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ap dung mat na ROI hinh thang len anh canh Canny bang cv2.fillPoly.

    Tra ve: (roi_edges, roi_mask)
    """
    mask = np.zeros_like(edges, dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges, mask


def detect_hough_lines_p(roi_edges: np.ndarray, config: HoughConfig) -> np.ndarray:
    """Chay Probabilistic Hough Transform (cv2.HoughLinesP) tren vung ROI.

    Tra ve mang cac doan thang dang (N, 4) voi [x1, y1, x2, y2].
    Neu khong tim thay, tra ve mang rong kich thuoc (0, 4).
    """
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=config.rho,
        theta=config.theta,
        threshold=config.threshold,
        minLineLength=config.min_line_length,
        maxLineGap=config.max_line_gap,
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    return lines.reshape(-1, 4).astype(np.int32)


def detect_hough_lines_standard(
    roi_edges: np.ndarray,
    config: HoughConfig,
) -> np.ndarray:
    """Chay Standard Hough Transform (cv2.HoughLines) phuc vu so sanh va nghien cuu.

    Tra ve mang cac duong thang dang (N, 2) voi [rho, theta].
    """
    lines = cv2.HoughLines(
        roi_edges,
        rho=config.rho,
        theta=config.theta,
        threshold=config.threshold,
    )
    if lines is None:
        return np.empty((0, 2), dtype=np.float32)
    return lines.reshape(-1, 2)


def split_left_right_segments(
    lines: np.ndarray,
    width: int,
    config: HoughConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Loc do doc va phan chia cac doan thang thanh 2 cum bien trai va bien phai.

    He toa do anh (goc (0,0) o goc tren-trai):
    - Bien trai huong ve vanishing point: dx > 0 khi dy < 0 => slope = dy / dx < 0.
    - Bien phai huong ve vanishing point: dx < 0 khi dy < 0 => slope = dy / dx > 0.
    - Cac doan thang nam ngang (|slope| < min_slope) hoac gan thang dung (|slope| > max_slope) bi loai.
    - Rang buoc khong gian: doan bien trai phai nam o phia ben trai (x_mid < 0.65 W),
      doan bien phai phai nam o phia ben phai (x_mid > 0.35 W).
    """
    if len(lines) == 0:
        return np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=np.int32)

    left_list: list[np.ndarray] = []
    right_list: list[np.ndarray] = []

    for seg in lines:
        x1, y1, x2, y2 = seg
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            # Duong thang dung hoan toan
            continue

        slope = dy / dx
        abs_slope = abs(slope)

        if not (config.min_abs_slope <= abs_slope <= config.max_abs_slope):
            continue

        x_mid = (x1 + x2) / 2.0

        if slope < 0 and x_mid < width * 0.65:
            left_list.append(seg)
        elif slope > 0 and x_mid > width * 0.35:
            right_list.append(seg)

    left_arr = np.array(left_list, dtype=np.int32) if left_list else np.empty((0, 4), dtype=np.int32)
    right_arr = np.array(right_list, dtype=np.int32) if right_list else np.empty((0, 4), dtype=np.int32)

    return left_arr, right_arr


def fit_boundary_line(
    segments: np.ndarray,
    y_bottom: int,
    y_top: int,
    *,
    side: Literal["left", "right"],
    width: int,
    config: HoughConfig,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Chon cum dong thuan tat dinh roi WLS x(y), khong tron moi doan cung dau.

    Moi doan de xuat mot mo hinh. Ca hai dau mut cua inlier phai cach mo
    hinh khong qua fit_distance_ratio * width pixel (khoang cach vuong goc).
    Chon tong chieu dai inlier lon nhat, hoa thi chon sai so nho nhat.
    """
    if side not in ("left", "right"):
        raise ValueError("side phai la left hoac right")
    if y_bottom <= y_top or width < 2:
        return None
    data = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    if len(data) == 0:
        return None
    dx = data[:, 2] - data[:, 0]
    dy = data[:, 3] - data[:, 1]
    slopes = np.divide(dy, dx, out=np.zeros_like(dy), where=dx != 0)
    valid = (
        np.all(np.isfinite(data), axis=1)
        & (dx != 0) & (dy != 0)
        & (np.abs(slopes) >= config.min_abs_slope)
        & (np.abs(slopes) <= config.max_abs_slope)
        & ((slopes < 0) if side == "left" else (slopes > 0))
    )
    data = data[valid]
    if len(data) == 0:
        return None
    lengths = np.hypot(data[:, 2] - data[:, 0], data[:, 3] - data[:, 1])
    # Sap xep de hoa diem khong phu thuoc thu tu cac doan Hough.
    order = np.lexsort(data.T[::-1])
    data, lengths = data[order], lengths[order]
    best_score = (-1.0, -np.inf)
    best_inliers = np.zeros(len(data), dtype=bool)
    for x1, y1, x2, y2 in data:
        k = (x2 - x1) / (y2 - y1)
        b = x1 - k * y1
        residuals = np.maximum(
            np.abs(data[:, 0] - k * data[:, 1] - b),
            np.abs(data[:, 2] - k * data[:, 3] - b),
        ) / np.hypot(1.0, k)
        inliers = residuals <= config.fit_distance_ratio * width
        support = float(lengths[inliers].sum())
        error = float(np.average(residuals[inliers], weights=lengths[inliers]))
        score = (support, -error)
        if score > best_score:
            best_score, best_inliers = score, inliers

    selected = data[best_inliers]
    weights = np.tile(lengths[best_inliers], 2)
    x_coords = np.concatenate((selected[:, 0], selected[:, 2]))
    y_coords = np.concatenate((selected[:, 1], selected[:, 3]))
    mean_y = np.average(y_coords, weights=weights)
    mean_x = np.average(x_coords, weights=weights)
    var_y = np.sum(weights * (y_coords - mean_y) ** 2)
    if var_y < 1e-5:
        return None
    k = np.sum(weights * (y_coords - mean_y) * (x_coords - mean_x)) / var_y
    if not np.isfinite(k) or k == 0:
        return None
    slope = 1.0 / k
    if not (config.min_abs_slope <= abs(slope) <= config.max_abs_slope):
        return None
    if (side == "left" and k >= 0) or (side == "right" and k <= 0):
        return None
    b = mean_x - k * mean_y
    return (
        (int(round(k * y_bottom + b)), y_bottom),
        (int(round(k * y_top + b)), y_top),
    )


def create_road_polygon(
    left_line: tuple[tuple[int, int], tuple[int, int]] | None,
    right_line: tuple[tuple[int, int], tuple[int, int]] | None,
) -> np.ndarray | None:
    """Tao vung chung hop le; chi giu phan nam duoi giao diem hai bien.

    Khong ngoai suy qua mien y cua mot trong hai bien. Hai bien dao thu tu
    tai day, khong co mien chung, hoac khong con dien tich thi tra ve None.
    """
    if left_line is None or right_line is None:
        return None
    if any(line[0][1] <= line[1][1] for line in (left_line, right_line)):
        return None

    def x_at(line, y):
        (xb, yb), (xt, yt) = line
        return xt + (xb - xt) * (y - yt) / (yb - yt)

    y_bottom = min(left_line[0][1], right_line[0][1])
    y_top = max(left_line[1][1], right_line[1][1])
    if y_top >= y_bottom:
        return None
    gap_bottom = x_at(right_line, y_bottom) - x_at(left_line, y_bottom)
    gap_top = x_at(right_line, y_top) - x_at(left_line, y_top)
    if gap_bottom < 1:
        return None
    if gap_top < 1:
        # Gap la ham bac nhat theo y. Giu it nhat 1 pixel o dinh.
        y_top += int(np.ceil((1 - gap_top) * (y_bottom - y_top) / (gap_bottom - gap_top)))
    if y_top >= y_bottom:
        return None
    polygon = np.array([
        [round(x_at(left_line, y_bottom)), y_bottom],
        [round(x_at(left_line, y_top)), y_top],
        [round(x_at(right_line, y_top)), y_top],
        [round(x_at(right_line, y_bottom)), y_bottom],
    ], dtype=np.int32)
    if polygon[0, 0] >= polygon[3, 0] or polygon[1, 0] >= polygon[2, 0]:
        return None
    return polygon


def detect_road_boundaries(
    edges: np.ndarray,
    config: HoughConfig | None = None,
) -> HoughResult:
    """Pipeline hoan chinh cua Phan 4: nhan anh canh Canny, tra ve 2 bien duong va da giac."""
    cfg = config or HoughConfig()
    if edges.ndim != 2 or edges.dtype != np.uint8:
        raise ValueError("Can anh canh uint8 mot kenh")
    height, width = edges.shape[:2]

    # 1. Tao va ap dung mat na ROI hinh thang
    vertices = get_trapezoid_vertices(height, width, cfg)
    roi_edges, roi_mask = apply_roi(edges, vertices)

    # 2. P-Hough phat hien tap doan thang
    raw_lines = detect_hough_lines_p(roi_edges, cfg)

    # 3. Loc do doc va phan chia cum trai / phai
    left_segs, right_segs = split_left_right_segments(raw_lines, width, cfg)

    # 4. Hoi quy tuyen tinh co trong so de tim 2 duong bien
    y_bottom = int(vertices[0, 0, 1])
    y_top = int(vertices[0, 1, 1])

    left_line = fit_boundary_line(
        left_segs, y_bottom, y_top, side="left", width=width, config=cfg,
    )
    right_line = fit_boundary_line(
        right_segs, y_bottom, y_top, side="right", width=width, config=cfg,
    )

    def clip_line(line):
        if line is None:
            return None
        visible, bottom, top = cv2.clipLine((0, 0, width, height), *line)
        return (bottom, top) if visible and bottom[1] > top[1] else None

    left_line, right_line = clip_line(left_line), clip_line(right_line)

    # 5. Tao da giac vung duong giua 2 bien
    polygon = create_road_polygon(left_line, right_line)
    if polygon is not None:
        # Cung mien y, dung thu tu trai/phai, toa do trong khung anh.
        left_line = (tuple(map(int, polygon[0])), tuple(map(int, polygon[1])))
        right_line = (tuple(map(int, polygon[3])), tuple(map(int, polygon[2])))

    return HoughResult(
        left_line=left_line,
        right_line=right_line,
        polygon=polygon,
        raw_lines=raw_lines,
        left_segments=left_segs,
        right_segments=right_segs,
        roi_mask=roi_mask,
    )


def draw_hough_visualization(
    background_rgb: np.ndarray,
    result: HoughResult,
    draw_raw_lines: bool = False,
    draw_roi: bool = True,
) -> np.ndarray:
    """Ve cac ket qua phat hien len anh RGB goc de truc quan hoa va phan tich."""
    canvas = background_rgb.copy()

    # Ve da giac mat duong trong suot mau xanh la
    if result.polygon is not None:
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [result.polygon], (0, 200, 0))
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # Ve vien ROI hinh thang mau vang nhat
    if draw_roi:
        contours, _ = cv2.findContours(result.roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 0), 1, lineType=cv2.LINE_AA)

    # Ve ung vien sau loc slope (RGB: trai do nhat, phai xanh lam nhat).
    if draw_raw_lines:
        for x1, y1, x2, y2 in result.left_segments:
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 100, 100), 1, lineType=cv2.LINE_AA)
        for x1, y1, x2, y2 in result.right_segments:
            cv2.line(canvas, (x1, y1), (x2, y2), (100, 100, 255), 1, lineType=cv2.LINE_AA)

    # Ve hai bien duong chinh (bien trai do dam, bien phai xanh duong dam)
    if result.left_line is not None:
        pt1, pt2 = result.left_line
        cv2.line(canvas, pt1, pt2, (255, 0, 0), 3, lineType=cv2.LINE_AA)

    if result.right_line is not None:
        pt1, pt2 = result.right_line
        cv2.line(canvas, pt1, pt2, (0, 0, 255), 3, lineType=cv2.LINE_AA)

    return canvas
