"""Canny cai bang NumPy: Gauss, Sobel, NMS, nguong kep + hysteresis."""

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
