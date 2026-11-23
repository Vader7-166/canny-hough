"""Hinh cho baocao/Bao_cao_Khung_Abstract_den_3.1.docx (Abstract, muc 1, 2, 3.1, Conclusion).

- fig_2_2_kieu_anh.png    : anh RGB, kenh H, S, V, anh xam, anh nhi phan (mat na long duong v3)
- fig_2_3_pipeline.png    : so do pipeline v3
- fig_3_1_chuan_hoa.png   : anh toi -> gamma -> gamma + CLAHE, kem histogram kenh L

Chay tu thu muc goc: python scripts/figures_khung_bao_cao.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import road_pipeline as rp  # noqa: E402
from preprocessing import resize_rgb, rgb_to_hsv_opencv  # noqa: E402

OUT = ROOT / "outputs" / "khung_bao_cao"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def load(group: str, name: str) -> np.ndarray:
    src = next((ROOT / "dataset" / "images" / group).glob(f"{name}.*"))
    return resize_rgb(rp._read_rgb(src), 960, 540)


def fig_kieu_anh(group="normal", name="normal_11") -> Path:
    rgb = load(group, name)
    hsv = rgb_to_hsv_opencv(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.asarray(Image.open(ROOT / "outputs" / "pipeline_v3" / "road_mask" / group / f"{name}.png"))
    panels = [
        (rgb, None, "Ảnh màu RGB (3 kênh, 960 × 540)"),
        (hsv[..., 0], "hsv", "Kênh H: sắc độ (0–179)"),
        (hsv[..., 1], "gray", "Kênh S: độ bão hòa (0–255)"),
        (hsv[..., 2], "gray", "Kênh V = max(R, G, B): độ sáng"),
        (gray, "gray", "Ảnh xám: 0,299R + 0,587G + 0,114B"),
        (mask, "gray", "Ảnh nhị phân: mặt nạ lòng đường (0/255)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 4.9))
    for ax, (img, cmap, title) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=179 if cmap == "hsv" else 255) if cmap else ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    path = OUT / "fig_2_2_kieu_anh.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_pipeline() -> Path:
    stages = [
        ("Ảnh đầu vào", "1280 × 720, RGB", "#EEEEEE"),
        ("Resize", "960 × 540, Lanczos", "#D9EAF7"),
        ("Chuẩn hóa tương phản", "gamma + CLAHE trên kênh L\n(chỉ ảnh tối)", "#D9EAF7"),
        ("Ước lượng hình học", "điểm tụ: Canny + Hough (kênh V)\nmép capo: màu S, V", "#FFF2CC"),
        ("Phân vùng GrabCut", "khởi tạo từ điểm tụ, mép capo\n3 lần chạy, lấy đa số", "#E2F0D9"),
        ("Hậu xử lý mặt nạ", "mép capo cong, đóng hình thái,\ncắt theo tia (Canny + Hough)", "#E2F0D9"),
        ("Hai biên trái/phải", "Canny trên mặt nạ\n→ Hough + WLS", "#FCE4D6"),
        ("Đánh giá", "IoU, F1 (vùng đường)\nMAE, SR15 (biên)", "#EEEEEE"),
    ]
    sections = ["", "3.1", "3.1", "", "", "", "", ""]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.axis("off")
    w, h = 0.86, 0.72
    pos = [(0.07 + i * 1.0, 1.15) for i in range(4)] + [(0.07 + (3 - i) * 1.0, 0.12) for i in range(4)]
    for (title, sub, color), (x, y), sec in zip(stages, pos, sections):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc=color, ec="#1F4E79", lw=1.2))
        ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x + w / 2, y + 0.27, sub, ha="center", va="center", fontsize=8.6, linespacing=1.3)
        if sec:
            ax.text(x + w - 0.03, y + h - 0.06, sec, ha="right", va="top", fontsize=8, color="#1F4E79")
    for i in range(len(pos) - 1):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        if y0 == y1:
            a = (x0 + w, y0 + h / 2) if x1 > x0 else (x0, y0 + h / 2)
            b = (x1, y1 + h / 2) if x1 > x0 else (x1 + w, y1 + h / 2)
        else:
            a, b = (x0 + w / 2, y0), (x1 + w / 2, y1 + h)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color="#1F4E79", lw=1.4))
    ax.text(0.07, 1.98, "Tiền xử lý  ·  Phân vùng  ·  Canny + Hough  ·  Đánh giá", fontsize=9, color="#555555",
            va="top")
    ax.text(3.93 + 0.0, 0.03, "vùng đường = mặt nạ sau hậu xử lý", fontsize=8.2, color="#555555", ha="right")
    fig.tight_layout()
    path = OUT / "fig_2_3_pipeline.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def gamma_only(rgb, cfg=rp.RoadConfig()):
    """Lap lai buoc gamma cua rp.normalize_contrast (chua CLAHE); tra ve (anh, gamma)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    level = float(np.median(lab[..., 0])) / 255.0
    gamma = float(np.clip(np.log(cfg.target_lightness) / np.log(level), 1.0 / cfg.max_gamma, 1.0))
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    lab = lab.copy()
    lab[..., 0] = cv2.LUT(lab[..., 0], lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), gamma


def fig_chuan_hoa(items=(("curved", "curved_06"), ("normal", "normal_15"))) -> Path:
    fig, axes = plt.subplots(len(items), 4, figsize=(13, 3.0 * len(items)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    for row, (group, name) in zip(axes, items):
        rgb = load(group, name)
        full, applied = rp.normalize_contrast(rgb)
        assert applied
        mid, gamma = gamma_only(rgb)
        L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
        med_low = float(np.median(L[L.shape[0] // 2:]))
        for ax, img, title in ((row[0], rgb, f"{name}: ảnh gốc\ntrung vị L nửa dưới = {med_low:.0f} < 50"),
                               (row[1], mid, f"Sau gamma (γ = {gamma:.2f})".replace(".", ",")),
                               (row[2], full, "Sau gamma + CLAHE (clip 2,0, lưới 8 × 8)")):
            ax.imshow(img)
            ax.set_title(title, fontsize=9.5)
            ax.axis("off")
        ax = row[3]
        for img, label, color in ((rgb, "gốc", "#7F7F7F"), (mid, "gamma", "#2E75B6"),
                                  (full, "gamma + CLAHE", "#C00000")):
            l_chan = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0]
            hist = np.bincount(l_chan.ravel(), minlength=256) / l_chan.size
            ax.plot(np.convolve(hist, np.ones(5) / 5, mode="same"), color=color, lw=1.4, label=label)
        ax.set_title("Histogram kênh L", fontsize=9.5)
        ax.set_xlim(0, 255)
        ax.set_yticks([])
        ax.set_xlabel("độ sáng L (0–255)")
        ax.legend(fontsize=8, frameon=False)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = OUT / "fig_3_1_chuan_hoa.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_kieu_anh, fig_pipeline, fig_chuan_hoa):
        print(f())
