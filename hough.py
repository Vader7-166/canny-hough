"""Module bien doi Hough va hau xu ly phat hien bien duong (Phan 4).

Cung cap cac ham cat ROI hinh thang, HoughLinesP, loc do doc,
phan cum trai/phai va hoi quy tuyen tinh co trong so de xac dinh hai bien duong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

    def __post_init__(self) -> None:
        if not (0.0 < self.roi_top_y_ratio < self.roi_bottom_y_ratio <= 1.0):
            raise ValueError("roi_top_y_ratio va roi_bottom_y_ratio khong hop le")
        if self.min_abs_slope >= self.max_abs_slope:
            raise ValueError("min_abs_slope phai nho hon max_abs_slope")


@dataclass(frozen=True)
class HoughResult:
    """Ket qua phat hien bien duong tu module Hough."""

    left_line: tuple[tuple[int, int], tuple[int, int]] | None    # ((x_bottom, y_bottom), (x_top, y_top))
    right_line: tuple[tuple[int, int], tuple[int, int]] | None   # ((x_bottom, y_bottom), (x_top, y_top))
    polygon: np.ndarray | None                                   # Toa do da giac (4, 2) mat duong giua 2 bien
    raw_lines: np.ndarray                                        # Mang tat ca doan thang HoughLinesP tim thay (N, 4)
    left_segments: np.ndarray                                    # Tap cac doan thang thuoc bien trai (K, 4)
    right_segments: np.ndarray                                   # Tap cac doan thang thuoc bien phai (M, 4)
    roi_mask: np.ndarray                                         # Mat na ROI da giac (H, W)


def get_trapezoid_vertices(height: int, width: int, config: HoughConfig) -> np.ndarray:
    """Tinh toa do 4 dinh cua ROI hinh thang can doi qua truc giua."""
    center_x = width / 2.0
    top_y = int(height * config.roi_top_y_ratio)
    bottom_y = int(height * config.roi_bottom_y_ratio)

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
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Hoi quy tuyen tinh co trong so (Weighted Least Squares) de tim duong bien dai dien.

    Giai quyet tinh don le cua do doc bang cach hoi quy x theo y: x = k * y + b.
    Vi y nam trong doan huu han [y_top, y_bottom], dao ham dx / dy luon xac dinh tot
    ngay ca khi duong bien gan nhu thang dung.

    Trong so w_i la chieu dai doan do doan thang Hough tao ra: w_i = sqrt(dx^2 + dy^2).
    Tra ve ((x_bottom, y_bottom), (x_top, y_top)) hoac None neu khong the fit.
    """
    if len(segments) == 0:
        return None

    # Tinh chieu dai moi doan
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    lengths = np.hypot(dx, dy)

    # Bo qua cac doan co chieu dai bang 0
    valid = lengths > 1e-3
    if not np.any(valid):
        return None

    valid_segments = segments[valid]
    weights = lengths[valid]

    # Moi doan thang gop 2 diem (x1, y1) va (x2, y2) voi cung trong so weight
    x_coords = np.concatenate([valid_segments[:, 0], valid_segments[:, 2]])
    y_coords = np.concatenate([valid_segments[:, 1], valid_segments[:, 3]])
    w = np.concatenate([weights, weights])

    total_w = np.sum(w)
    if total_w <= 0:
        return None

    mean_y = np.sum(w * y_coords) / total_w
    mean_x = np.sum(w * x_coords) / total_w

    var_y = np.sum(w * (y_coords - mean_y) ** 2)
    cov_xy = np.sum(w * (y_coords - mean_y) * (x_coords - mean_x))

    if var_y < 1e-5:
        return None

    k = cov_xy / var_y
    b = mean_x - k * mean_y

    x_bottom = int(round(k * y_bottom + b))
    x_top = int(round(k * y_top + b))

    return ((x_bottom, y_bottom), (x_top, y_top))


def create_road_polygon(
    left_line: tuple[tuple[int, int], tuple[int, int]] | None,
    right_line: tuple[tuple[int, int], tuple[int, int]] | None,
) -> np.ndarray | None:
    """Tao da giac 4 dinh bieu dien vung mat duong giua 2 bien trai va phai."""
    if left_line is None or right_line is None:
        return None

    (x_l_bot, y_l_bot), (x_l_top, y_l_top) = left_line
    (x_r_bot, y_r_bot), (x_r_top, y_r_top) = right_line

    polygon = np.array(
        [
            [x_l_bot, y_l_bot],
            [x_l_top, y_l_top],
            [x_r_top, y_r_top],
            [x_r_bot, y_r_bot],
        ],
        dtype=np.int32,
    )
    return polygon


def detect_road_boundaries(
    edges: np.ndarray,
    config: HoughConfig | None = None,
) -> HoughResult:
    """Pipeline hoan chinh cua Phan 4: nhan anh canh Canny, tra ve 2 bien duong va da giac."""
    cfg = config or HoughConfig()
    height, width = edges.shape[:2]

    # 1. Tao va ap dung mat na ROI hinh thang
    vertices = get_trapezoid_vertices(height, width, cfg)
    roi_edges, roi_mask = apply_roi(edges, vertices)

    # 2. P-Hough phat hien tap doan thang
    raw_lines = detect_hough_lines_p(roi_edges, cfg)

    # 3. Loc do doc va phan chia cum trai / phai
    left_segs, right_segs = split_left_right_segments(raw_lines, width, cfg)

    # 4. Hoi quy tuyen tinh co trong so de tim 2 duong bien
    y_bottom = int(height * cfg.roi_bottom_y_ratio)
    y_top = int(height * cfg.roi_top_y_ratio)

    left_line = fit_boundary_line(left_segs, y_bottom, y_top)
    right_line = fit_boundary_line(right_segs, y_bottom, y_top)

    # 5. Tao da giac vung duong giua 2 bien
    polygon = create_road_polygon(left_line, right_line)

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

    # Ve cac doan thang tho neu duoc yeu cau (doan trai mau xanh lam, doan phai mau do)
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
