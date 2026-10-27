"""Tien xu ly anh dau vao cho pipeline phat hien bien duong.

Day la bien the de xuat cua pipeline Belhadri et al. (2026): thay buoc chuyen
thang sang grayscale bang RGB -> HSV -> loc mau vach trang/vang. Module chi
phu thuoc NumPy va Pillow de co the chay ca tren may local va Google Colab.
Anh dau ra ``canny_input`` la anh xam uint8 da loc mau va lam mo Gaussian.
Module dung tai day va khong thuc hien Canny, ROI hay Hough.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PreprocessConfig:
    """Cau hinh mac dinh da dong bo voi quy uoc HSV cua OpenCV."""

    target_width: int = 960
    target_height: int = 540
    white_saturation_max: int = 80
    white_value_min: int = 170
    yellow_hue_min: int = 15
    yellow_hue_max: int = 40
    yellow_saturation_min: int = 70
    yellow_value_min: int = 90
    gaussian_radius: float = 1.2

    def __post_init__(self) -> None:
        if self.target_width <= 0 or self.target_height <= 0:
            raise ValueError("Kich thuoc dau ra phai lon hon 0")
        if not 0 <= self.white_saturation_max <= 255:
            raise ValueError("white_saturation_max phai nam trong [0, 255]")
        if not 0 <= self.white_value_min <= 255:
            raise ValueError("white_value_min phai nam trong [0, 255]")
        if not 0 <= self.yellow_hue_min <= self.yellow_hue_max <= 179:
            raise ValueError("Nguong H mau vang phai nam trong [0, 179]")
        if not 0 <= self.yellow_saturation_min <= 255:
            raise ValueError("yellow_saturation_min phai nam trong [0, 255]")
        if not 0 <= self.yellow_value_min <= 255:
            raise ValueError("yellow_value_min phai nam trong [0, 255]")
        if self.gaussian_radius < 0:
            raise ValueError("gaussian_radius khong duoc am")


@dataclass(frozen=True)
class PreprocessResult:
    resized_rgb: np.ndarray
    hsv: np.ndarray
    color_mask: np.ndarray
    color_filtered_rgb: np.ndarray
    gaussian_rgb: np.ndarray
    canny_input: np.ndarray


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Anh dau vao phai co dang H x W x 3")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size:
            if float(array.min()) >= 0.0 and float(array.max()) <= 1.0:
                array = np.rint(array * 255.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize ve kich thuoc co dinh bang Lanczos."""

    image = Image.fromarray(_validate_rgb(rgb), mode="RGB")
    return np.asarray(image.resize((width, height), Image.Resampling.LANCZOS))


def rgb_to_hsv_opencv(rgb: np.ndarray) -> np.ndarray:
    """Chuyen RGB sang HSV voi H in [0,179], S,V in [0,255]."""

    source = _validate_rgb(rgb).astype(np.float32) / 255.0
    red, green, blue = source[..., 0], source[..., 1], source[..., 2]
    maximum = source.max(axis=2)
    minimum = source.min(axis=2)
    delta = maximum - minimum

    hue_degrees = np.zeros_like(maximum)
    non_gray = delta > 1e-7
    red_max = non_gray & (maximum == red)
    green_max = non_gray & (maximum == green)
    blue_max = non_gray & (maximum == blue)
    hue_degrees[red_max] = (60.0 * ((green[red_max] - blue[red_max]) / delta[red_max])) % 360.0
    hue_degrees[green_max] = 60.0 * ((blue[green_max] - red[green_max]) / delta[green_max] + 2.0)
    hue_degrees[blue_max] = 60.0 * ((red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0)

    saturation = np.zeros_like(maximum)
    non_black = maximum > 1e-7
    saturation[non_black] = delta[non_black] / maximum[non_black]

    hue = np.clip(np.rint(hue_degrees / 2.0), 0, 179).astype(np.uint8)
    sat = np.clip(np.rint(saturation * 255.0), 0, 255).astype(np.uint8)
    value = np.clip(np.rint(maximum * 255.0), 0, 255).astype(np.uint8)
    return np.stack((hue, sat, value), axis=2)


def lane_marking_mask(
    hsv: np.ndarray,
    white_saturation_max: int = 80,
    white_value_min: int = 170,
    yellow_hue_min: int = 15,
    yellow_hue_max: int = 40,
    yellow_saturation_min: int = 70,
    yellow_value_min: int = 90,
) -> np.ndarray:
    """Tao mat na ket hop vach trang va vach vang trong khong gian HSV."""

    array = np.asarray(hsv)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Anh HSV phai co dang H x W x 3")
    hue = array[..., 0]
    saturation = array[..., 1]
    value = array[..., 2]
    white = (saturation <= white_saturation_max) & (value >= white_value_min)
    yellow = (
        (hue >= yellow_hue_min)
        & (hue <= yellow_hue_max)
        & (saturation >= yellow_saturation_min)
        & (value >= yellow_value_min)
    )
    selected = white | yellow
    return np.where(selected, 255, 0).astype(np.uint8)


def apply_binary_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    source = _validate_rgb(rgb)
    if mask.shape != source.shape[:2]:
        raise ValueError("Mask va anh phai cung kich thuoc")
    return np.where(mask[..., None] > 0, source, 0).astype(np.uint8)


def gaussian_blur_rgb(rgb: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(_validate_rgb(rgb), mode="RGB")
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius=radius)))


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    source = _validate_rgb(rgb).astype(np.float32)
    gray = 0.299 * source[..., 0] + 0.587 * source[..., 1] + 0.114 * source[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def preprocess_rgb(rgb: np.ndarray, config: PreprocessConfig | None = None) -> PreprocessResult:
    """Chay toan bo tien xu ly va tra ve cac buoc trung gian."""

    cfg = config or PreprocessConfig()
    resized = resize_rgb(rgb, cfg.target_width, cfg.target_height)
    hsv = rgb_to_hsv_opencv(resized)
    color_mask = lane_marking_mask(
        hsv,
        cfg.white_saturation_max,
        cfg.white_value_min,
        cfg.yellow_hue_min,
        cfg.yellow_hue_max,
        cfg.yellow_saturation_min,
        cfg.yellow_value_min,
    )
    filtered = apply_binary_mask(resized, color_mask)
    gaussian = gaussian_blur_rgb(filtered, cfg.gaussian_radius)
    canny_input = rgb_to_gray(gaussian)
    return PreprocessResult(resized, hsv, color_mask, filtered, gaussian, canny_input)


def iter_image_files(input_dir: Path) -> Iterable[Path]:
    """Chi doc anh trong nhanh input duoc chi dinh, khong lan sang visualizations."""

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def process_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    config: PreprocessConfig | None = None,
    save_intermediate: bool = False,
) -> dict:
    """Xu ly hang loat va giu nguyen cau truc nhom normal/curved/adverse."""

    cfg = config or PreprocessConfig()
    source_root = Path(input_dir)
    destination_root = Path(output_dir)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Khong tim thay thu muc anh: {source_root}")

    records = []
    for source_path in iter_image_files(source_root):
        relative = source_path.relative_to(source_root).with_suffix(".png")
        with Image.open(source_path) as opened:
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
        result = preprocess_rgb(rgb, cfg)

        _save_array(destination_root / "canny_input" / relative, result.canny_input)
        if save_intermediate:
            _save_array(destination_root / "resized" / relative, result.resized_rgb)
            _save_array(destination_root / "color_masks" / relative, result.color_mask)
            _save_array(destination_root / "color_filtered" / relative, result.color_filtered_rgb)
            _save_array(destination_root / "gaussian_rgb" / relative, result.gaussian_rgb)

        records.append(
            {
                "source": source_path.relative_to(source_root).as_posix(),
                "output": (Path("canny_input") / relative).as_posix(),
                "retained_pixel_ratio": round(float(np.mean(result.color_mask > 0)), 6),
            }
        )

    summary = {
        "processed_images": len(records),
        "config": asdict(cfg),
        "method": "Proposed Belhadri et al. (2026) variant: RGB-to-HSV white/yellow lane-color filtering before Gaussian; output stops before Canny.",
        "handoff": "Use canny_input/*.png as the input of the separate Canny module.",
        "files": records,
    }
    destination_root.mkdir(parents=True, exist_ok=True)
    (destination_root / "preprocessing_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tien xu ly anh cho buoc Canny")
    parser.add_argument("--input", default="dataset/images", help="Thu muc anh goc")
    parser.add_argument("--output", default="outputs/preprocessed_for_canny", help="Thu muc dau ra")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--white-saturation-max", type=int, default=80)
    parser.add_argument("--white-value-min", type=int, default=170)
    parser.add_argument("--yellow-hue-min", type=int, default=15)
    parser.add_argument("--yellow-hue-max", type=int, default=40)
    parser.add_argument("--yellow-saturation-min", type=int, default=70)
    parser.add_argument("--yellow-value-min", type=int, default=90)
    parser.add_argument("--gaussian-radius", type=float, default=1.2)
    parser.add_argument("--save-intermediate", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = PreprocessConfig(
        target_width=args.width,
        target_height=args.height,
        white_saturation_max=args.white_saturation_max,
        white_value_min=args.white_value_min,
        yellow_hue_min=args.yellow_hue_min,
        yellow_hue_max=args.yellow_hue_max,
        yellow_saturation_min=args.yellow_saturation_min,
        yellow_value_min=args.yellow_value_min,
        gaussian_radius=args.gaussian_radius,
    )
    summary = process_dataset(args.input, args.output, config, args.save_intermediate)
    print(f"Da xu ly {summary['processed_images']} anh. Ket qua: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
