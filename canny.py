"""Canny: ban tu cai bang NumPy (Gauss, Sobel, NMS, nguong kep + hysteresis) va ban OpenCV.

Pipeline (road_pipeline.py) dung canny_cv2 / canny_cv2_auto. Ban NumPy giu lai de kiem chung
(khop cv2.Canny, xem bao cao Phan 3) va de trinh bay 4 buoc cua thuat toan.
"""

import cv2
import numpy as np


def gaussian_kernel(size, sigma):
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def convolve_separable(img, k):
    r = len(k) // 2
    p = np.pad(img, r, mode="reflect")
    tmp = sum(k[i] * p[:, i:i + img.shape[1]] for i in range(len(k)))
    return sum(k[i] * tmp[i:i + img.shape[0], :] for i in range(len(k)))


def gaussian_smooth(img, size=5, sigma=1.4):
    return convolve_separable(img.astype(np.float64), gaussian_kernel(size, sigma))


def sobel_gradient(img, l2=True):
    p = np.pad(img, 1, mode="reflect")
    h, w = img.shape
    a = lambda dy, dx: p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    gx = (a(-1, 1) + 2 * a(0, 1) + a(1, 1)) - (a(-1, -1) + 2 * a(0, -1) + a(1, -1))
    gy = (a(1, -1) + 2 * a(1, 0) + a(1, 1)) - (a(-1, -1) + 2 * a(-1, 0) + a(-1, 1))
    mag = np.hypot(gx, gy) if l2 else np.abs(gx) + np.abs(gy)
    return mag, np.arctan2(gy, gx)


def non_max_suppression(mag, theta):
    h, w = mag.shape
    p = np.pad(mag, 1)
    a = lambda dy, dx: p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    ang = np.rad2deg(theta) % 180
    n1 = np.zeros_like(mag)
    n2 = np.zeros_like(mag)
    for lo, hi, dy, dx in [(0, 22.5, 0, 1), (22.5, 67.5, 1, 1), (67.5, 112.5, 1, 0), (112.5, 157.5, 1, -1), (157.5, 180.01, 0, 1)]:
        m = (ang >= lo) & (ang < hi)
        n1[m] = a(dy, dx)[m]
        n2[m] = a(-dy, -dx)[m]
    return np.where((mag > n1) & (mag >= n2), mag, 0)


def double_threshold_hysteresis(nms, low, high):
    strong = nms >= high
    weak = nms >= low
    edges = strong.copy()
    while True:
        p = np.pad(edges, 1)
        near = np.zeros_like(edges)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                near |= p[1 + dy:1 + dy + edges.shape[0], 1 + dx:1 + dx + edges.shape[1]]
        new = weak & near
        if (new == edges).all():
            return (edges * 255).astype(np.uint8)
        edges = new


def canny_edges(img, low, high, sigma, size=5, l2=True):
    smooth = gaussian_smooth(img, size, sigma)
    mag, theta = sobel_gradient(smooth, l2)
    return double_threshold_hysteresis(non_max_suppression(mag, theta), low, high)


def gradient_percentile_thresholds(mag, high_percentile=90.0, low_ratio=0.4, region=None, min_high=8.0):
    """Nguong kep theo phan vi do lon gradient thay vi gia tri co dinh.

    Nguong (50, 150) co dinh lam anh toi / tuong phan thap gan nhu khong con canh
    (recall bien long duong 0,09 o anh toi). Lay nguong cao bang phan vi `high_percentile`
    cua do lon gradient trong `region` (vi du phan duoi duong chan troi), nguong thap bang
    `low_ratio` lan nguong cao, nen luong canh on dinh khi do tuong phan thay doi.
    """
    values = mag[region] if region is not None else mag
    if values.size == 0:
        values = mag.ravel()
    high = max(float(np.percentile(values, high_percentile)), min_high)
    return low_ratio * high, high


def canny_edges_auto(img, sigma=1.4, size=5, l2=True, high_percentile=90.0, low_ratio=0.4, region=None):
    """Canny voi nguong tu dong theo phan vi gradient. Tra ve (anh canh 0/255, (low, high))."""
    smooth = gaussian_smooth(img, size, sigma)
    mag, theta = sobel_gradient(smooth, l2)
    low, high = gradient_percentile_thresholds(mag, high_percentile, low_ratio, region)
    return double_threshold_hysteresis(non_max_suppression(mag, theta), low, high), (low, high)


# ---------------------------------------------------------------------------
# Ban OpenCV (pipeline dung ban nay; ban NumPy o tren giu lai de kiem chung va giai thich 4 buoc)
# ---------------------------------------------------------------------------

def _cv2_smooth(img, size, sigma):
    # cv2.Canny khong tu lam mo Gauss nen phai lam mo truoc; dau vao cua cv2.Canny phai la uint8
    return cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (size, size), sigma)


def canny_cv2(img, low, high, sigma=1.4, size=5, l2=True):
    """Gauss (cv2.GaussianBlur) + cv2.Canny (Sobel 3x3, NMS, nguong kep + hysteresis). Tra ve anh canh 0/255."""
    return cv2.Canny(_cv2_smooth(img, size, sigma), low, high, apertureSize=3, L2gradient=l2)


def canny_cv2_auto(img, sigma=1.4, size=5, l2=True, high_percentile=90.0, low_ratio=0.4, region=None):
    """cv2.Canny voi nguong theo phan vi gradient. Tra ve (anh canh 0/255, (low, high)).

    Do lon gradient duoc tinh bang cv2.Sobel 3x3 tren cung anh da lam mo ma cv2.Canny dung ben trong,
    nen nguong phan vi cung thang do voi gradient cua cv2.Canny.
    """
    smooth = _cv2_smooth(img, size, sigma)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy) if l2 else np.abs(gx) + np.abs(gy)
    low, high = gradient_percentile_thresholds(mag, high_percentile, low_ratio, region)
    return cv2.Canny(smooth, low, high, apertureSize=3, L2gradient=l2), (low, high)
