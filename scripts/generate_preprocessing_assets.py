"""Sinh hinh minh hoa va thong ke cho bao cao Phan 2."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing import PreprocessConfig, preprocess_rgb, rgb_to_gray


DATASET = ROOT / "dataset"
OUTPUT = ROOT / "outputs" / "preprocessing_figures"
SAMPLES = [
    ("normal", "normal_12", "Normal"),
    ("curved", "curved_09", "Curved"),
    ("adverse", "adverse_03", "Adverse"),
]
CONFIG = PreprocessConfig()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def to_rgb(array: np.ndarray) -> Image.Image:
    data = np.asarray(array)
    if data.ndim == 2:
        data = np.repeat(data[..., None], 3, axis=2)
    return Image.fromarray(data.astype(np.uint8), mode="RGB")


def hsv_visualization(hsv: np.ndarray) -> np.ndarray:
    pillow_hsv = hsv.copy()
    pillow_hsv[..., 0] = np.rint(hsv[..., 0].astype(np.float32) * 255.0 / 179.0).astype(np.uint8)
    return np.asarray(Image.fromarray(pillow_hsv, mode="HSV").convert("RGB"))


def make_grid(
    panels: list[tuple[Image.Image, str]],
    columns: int,
    panel_size: tuple[int, int],
    title: str,
) -> Image.Image:
    panel_width, panel_height = panel_size
    label_height = 48
    title_height = 76
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * panel_width, title_height + rows * (panel_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), title, fill="black", font=load_font(32, bold=True))
    for index, (image, label) in enumerate(panels):
        row, column = divmod(index, columns)
        x = column * panel_width
        y = title_height + row * (panel_height + label_height)
        fitted = image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        canvas.paste(fitted, (x, y))
        draw.rectangle((x, y + panel_height, x + panel_width, y + panel_height + label_height), fill="#F2F2F2")
        draw.text((x + 16, y + panel_height + 10), label, fill="black", font=load_font(24, bold=True))
    return canvas


def high_frequency_energy(gray: np.ndarray) -> float:
    source = gray.astype(np.float32)
    horizontal = np.abs(np.diff(source, axis=1)).mean()
    vertical = np.abs(np.diff(source, axis=0)).mean()
    return float((horizontal + vertical) / 2.0)


def split_lane_masks(hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (saturation <= CONFIG.white_saturation_max) & (value >= CONFIG.white_value_min)
    yellow = (
        (hue >= CONFIG.yellow_hue_min)
        & (hue <= CONFIG.yellow_hue_max)
        & (saturation >= CONFIG.yellow_saturation_min)
        & (value >= CONFIG.yellow_value_min)
    )
    return white, yellow


def evaluate_mask(hsv: np.ndarray, color_mask: np.ndarray) -> dict[str, float]:
    white, yellow = split_lane_masks(hsv)
    return {
        "retained_pixel_ratio": float((color_mask > 0).mean()),
        "white_pixel_ratio": float(white.mean()),
        "yellow_pixel_ratio": float(yellow.mean()),
    }


def load_sample(group: str, stem: str) -> tuple[np.ndarray, object]:
    with Image.open(DATASET / "images" / group / f"{stem}.jpg") as image:
        rgb = np.asarray(image.convert("RGB"))
    return rgb, preprocess_rgb(rgb, CONFIG)


def build_figures(sample_results: list[tuple[str, str, str, np.ndarray, np.ndarray, object]]) -> None:
    group, stem, label, original, _, result = sample_results[0]
    figure_one_panels = [
        (to_rgb(original), "A  Ảnh gốc 1280 x 720"),
        (to_rgb(result.resized_rgb), "B  Resize 960 x 540"),
        (to_rgb(hsv_visualization(result.hsv)), "C  Biểu diễn ảnh trong HSV"),
        (to_rgb(result.color_mask), "D  Mặt nạ vạch trắng và vàng"),
        (to_rgb(result.color_filtered_rgb), "E  Ảnh sau lọc màu vạch làn"),
        (to_rgb(result.canny_input), "F  Gaussian và grayscale cho Canny"),
    ]
    make_grid(
        figure_one_panels,
        columns=3,
        panel_size=(640, 360),
        title=f"Quy trình tiền xử lý đề xuất trên mẫu {stem}",
    ).save(OUTPUT / "figure_1_pipeline_normal_12.png", quality=95)

    before_after = []
    gaussian_compare = []
    for group, stem, label, original, _, result in sample_results:
        before_after.extend(
            [
                (to_rgb(original), f"{label}  Trước lọc màu"),
                (to_rgb(result.color_filtered_rgb), f"{label}  Sau lọc màu"),
            ]
        )
        gaussian_compare.extend(
            [
                (to_rgb(result.color_filtered_rgb), f"{label}  Trước Gaussian"),
                (to_rgb(result.gaussian_rgb), f"{label}  Sau Gaussian"),
            ]
        )
    make_grid(
        before_after,
        columns=2,
        panel_size=(720, 405),
        title="So sánh trước và sau lọc màu vạch làn trên ba điều kiện",
    ).save(OUTPUT / "figure_2_before_after_color_filter.png", quality=95)
    make_grid(
        gaussian_compare,
        columns=2,
        panel_size=(720, 405),
        title="So sánh trước và sau làm mờ Gaussian",
    ).save(OUTPUT / "figure_3_before_after_gaussian.png", quality=95)

    hsv_result = sample_results[0][-1]
    white, yellow = split_lane_masks(hsv_result.hsv)
    hsv_panels = [
        (to_rgb(np.rint(hsv_result.hsv[..., 0].astype(np.float32) * 255.0 / 179.0).astype(np.uint8)), "A  Kênh H sắc độ"),
        (to_rgb(hsv_result.hsv[..., 1]), "B  Kênh S độ bão hòa"),
        (to_rgb(hsv_result.hsv[..., 2]), "C  Kênh V độ sáng"),
        (to_rgb(np.where(white, 255, 0).astype(np.uint8)), "D  Mặt nạ vạch trắng"),
        (to_rgb(np.where(yellow, 255, 0).astype(np.uint8)), "E  Mặt nạ vạch vàng"),
        (to_rgb(hsv_result.color_mask), "F  Mặt nạ kết hợp"),
    ]
    make_grid(
        hsv_panels,
        columns=3,
        panel_size=(640, 360),
        title="Các kênh HSV và mặt nạ màu trên mẫu normal_12",
    ).save(OUTPUT / "figure_4_hsv_channels_and_masks.png", quality=95)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sample_results = []
    rows = []
    group_values: dict[str, list[dict[str, float]]] = {"normal": [], "curved": [], "adverse": []}

    for group_dir in sorted((DATASET / "images").iterdir()):
        if not group_dir.is_dir() or group_dir.name not in group_values:
            continue
        for image_path in sorted(group_dir.glob("*.jpg")):
            stem = image_path.stem
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            result = preprocess_rgb(rgb, CONFIG)
            metrics = evaluate_mask(result.hsv, result.color_mask)
            before_energy = high_frequency_energy(rgb_to_gray(result.color_filtered_rgb))
            after_energy = high_frequency_energy(result.canny_input)
            metrics.update(
                {
                    "high_frequency_before": before_energy,
                    "high_frequency_after": after_energy,
                    "high_frequency_reduction": 1.0 - after_energy / max(before_energy, 1e-9),
                }
            )
            group_values[group_dir.name].append(metrics)
            if any(group_dir.name == group and stem == expected for group, expected, _ in SAMPLES):
                label = next(label for group, expected, label in SAMPLES if group == group_dir.name and expected == stem)
                sample_results.append((group_dir.name, stem, label, rgb, None, result))
                rows.append({"group": group_dir.name, "image": stem, **metrics})

    sample_order = {stem: index for index, (_, stem, _) in enumerate(SAMPLES)}
    sample_results.sort(key=lambda item: sample_order[item[1]])
    rows.sort(key=lambda item: sample_order[item["image"]])
    build_figures(sample_results)

    group_summary = {}
    for group, values in group_values.items():
        group_summary[group] = {
            key: float(np.mean([entry[key] for entry in values]))
            for key in values[0]
        }
        group_summary[group]["count"] = len(values)

    statistics = {
        "config": asdict(CONFIG),
        "base_paper": "Belhadri et al. (2026), Lane Boundary Detection in Dynamic Environments",
        "proposed_variant": "Replace direct RGB-to-grayscale conversion with RGB-to-HSV white/yellow lane-color filtering before Gaussian and Canny.",
        "dataset_label_warning": "Existing ground truth labels drivable road area, not lane markings; it is not used to score lane-boundary accuracy here.",
        "interpretation": {
            "retained_pixel_ratio": "Tỷ lệ pixel trắng/vàng được giữ trên toàn ảnh.",
            "white_pixel_ratio": "Tỷ lệ pixel thỏa ngưỡng vạch trắng.",
            "yellow_pixel_ratio": "Tỷ lệ pixel thỏa ngưỡng vạch vàng.",
            "high_frequency_reduction": "Mức giảm biến thiên cục bộ sau Gaussian.",
        },
        "representative_samples": rows,
        "group_mean": group_summary,
    }
    (OUTPUT / "preprocessing_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (OUTPUT / "preprocessing_statistics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Da tao {len(list(OUTPUT.glob('figure_*.png')))} hinh va thong ke tai {OUTPUT}")


if __name__ == "__main__":
    main()
