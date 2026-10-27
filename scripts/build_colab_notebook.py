"""Build the self-contained Google Colab notebook for preprocessing."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Tien_xu_ly_HSV_Colab.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "colab": {"name": OUTPUT.name, "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
}

notebook["cells"] = [
    markdown(
        """
# Phần 2  Tiền xử lý ảnh bằng không gian màu HSV

Notebook triển khai phần việc của **Thế** dựa trên bài *Lane Boundary Detection in Dynamic Environments* của Belhadri và cộng sự (2026).

Điểm cải tiến đề xuất là thay bước `RGB → grayscale` của bài base bằng:

`Resize → RGB sang HSV → lọc vạch trắng và vàng → Gaussian → grayscale đầu ra`

**Phạm vi dừng trước Canny.** Notebook không chạy Canny, ROI, Hough hoặc chọn hai biên làn. Thư mục `canny_input` là dữ liệu bàn giao cho thành viên phụ trách bước tiếp theo.
"""
    ),
    markdown(
        """
## Cách dùng trên Google Colab

1. Tải toàn bộ thư mục dự án lên Google Drive, ví dụ `MyDrive/Bai_Tap_Lon`.
2. Bỏ dấu `#` ở hai dòng mount Drive trong ô dưới nếu dùng Drive.
3. Sửa `PROJECT_DIR` nếu notebook không tự tìm thấy `dataset/images`.
4. Chạy lần lượt tất cả các ô. Kết quả 50 ảnh được ghi vào `outputs/preprocessed_for_canny`.

Ngưỡng HSV bên dưới là cấu hình thực nghiệm ban đầu, không phải tham số được bài báo base công bố.
"""
    ),
    code(
        """
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Nếu dùng Google Drive, bỏ dấu # ở hai dòng sau.
# from google.colab import drive
# drive.mount('/content/drive')

@dataclass(frozen=True)
class PreprocessConfig:
    target_width: int = 960
    target_height: int = 540
    white_saturation_max: int = 80
    white_value_min: int = 170
    yellow_hue_min: int = 15
    yellow_hue_max: int = 40
    yellow_saturation_min: int = 70
    yellow_value_min: int = 90
    gaussian_radius: float = 1.2

CONFIG = PreprocessConfig()
print(asdict(CONFIG))
"""
    ),
    code(
        """
def validate_rgb(rgb):
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError('Ảnh đầu vào phải có dạng H x W x 3')
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size:
            if float(array.min()) >= 0.0 and float(array.max()) <= 1.0:
                array = np.rint(array * 255.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def resize_rgb(rgb, width, height):
    image = Image.fromarray(validate_rgb(rgb), mode='RGB')
    return np.asarray(image.resize((width, height), Image.Resampling.LANCZOS))


def rgb_to_hsv_opencv(rgb):
    source = validate_rgb(rgb).astype(np.float32) / 255.0
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


def lane_marking_masks(hsv, config=CONFIG):
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (saturation <= config.white_saturation_max) & (value >= config.white_value_min)
    yellow = (
        (hue >= config.yellow_hue_min)
        & (hue <= config.yellow_hue_max)
        & (saturation >= config.yellow_saturation_min)
        & (value >= config.yellow_value_min)
    )
    combined = white | yellow
    return (
        np.where(white, 255, 0).astype(np.uint8),
        np.where(yellow, 255, 0).astype(np.uint8),
        np.where(combined, 255, 0).astype(np.uint8),
    )


def rgb_to_gray(rgb):
    source = validate_rgb(rgb).astype(np.float32)
    gray = 0.299 * source[..., 0] + 0.587 * source[..., 1] + 0.114 * source[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def preprocess_rgb(rgb, config=CONFIG):
    resized = resize_rgb(rgb, config.target_width, config.target_height)
    hsv = rgb_to_hsv_opencv(resized)
    white_mask, yellow_mask, color_mask = lane_marking_masks(hsv, config)
    color_filtered = np.where(color_mask[..., None] > 0, resized, 0).astype(np.uint8)
    gaussian_rgb = np.asarray(
        Image.fromarray(color_filtered, mode='RGB').filter(
            ImageFilter.GaussianBlur(radius=config.gaussian_radius)
        )
    )
    canny_input = rgb_to_gray(gaussian_rgb)
    return {
        'resized_rgb': resized,
        'hsv': hsv,
        'white_mask': white_mask,
        'yellow_mask': yellow_mask,
        'color_mask': color_mask,
        'color_filtered_rgb': color_filtered,
        'gaussian_rgb': gaussian_rgb,
        'canny_input': canny_input,
    }
"""
    ),
    markdown(
        """
## Chọn thư mục dự án và kiểm tra dữ liệu

Ô dưới chỉ quét `dataset/images`; không quét `dataset/visualizations`, vì thư mục đó chứa ảnh dẫn xuất có overlay ground truth.
"""
    ),
    code(
        """
candidates = [
    Path.cwd(),
    Path('/content/Bai_Tap_Lon'),
    Path('/content/drive/MyDrive/Bai_Tap_Lon'),
]
PROJECT_DIR = next((path for path in candidates if (path / 'dataset' / 'images').is_dir()), None)
if PROJECT_DIR is None:
    raise FileNotFoundError(
        'Không tìm thấy dataset/images. Hãy mount Drive hoặc sửa PROJECT_DIR theo vị trí dự án.'
    )

IMAGE_DIR = PROJECT_DIR / 'dataset' / 'images'
OUTPUT_DIR = PROJECT_DIR / 'outputs' / 'preprocessed_for_canny'
image_paths = sorted(
    path for path in IMAGE_DIR.rglob('*')
    if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}
)
print('PROJECT_DIR =', PROJECT_DIR)
print('Số ảnh tìm thấy =', len(image_paths))
assert len(image_paths) == 50, 'Bộ dữ liệu hiện tại phải có đúng 50 ảnh nguồn.'
"""
    ),
    markdown("## Minh họa từng bước trên ảnh normal_12"),
    code(
        """
sample_path = IMAGE_DIR / 'normal' / 'normal_12.jpg'
with Image.open(sample_path) as image:
    sample_rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
result = preprocess_rgb(sample_rgb)

panels = [
    (sample_rgb, 'Ảnh gốc'),
    (result['resized_rgb'], 'Resize 960 × 540'),
    (result['hsv'][..., 0], 'Kênh H'),
    (result['hsv'][..., 1], 'Kênh S'),
    (result['hsv'][..., 2], 'Kênh V'),
    (result['color_mask'], 'Mặt nạ trắng và vàng'),
    (result['color_filtered_rgb'], 'Sau lọc màu'),
    (result['gaussian_rgb'], 'Sau Gaussian'),
    (result['canny_input'], 'Ảnh bàn giao cho Canny'),
]

fig, axes = plt.subplots(3, 3, figsize=(18, 11))
for axis, (panel, title) in zip(axes.ravel(), panels):
    axis.imshow(panel, cmap='gray' if panel.ndim == 2 else None)
    axis.set_title(title)
    axis.axis('off')
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## So sánh trước và sau lọc màu trên ba điều kiện"),
    code(
        """
representatives = [
    ('Normal', IMAGE_DIR / 'normal' / 'normal_12.jpg'),
    ('Curved', IMAGE_DIR / 'curved' / 'curved_09.jpg'),
    ('Adverse', IMAGE_DIR / 'adverse' / 'adverse_03.jpg'),
]

fig, axes = plt.subplots(3, 2, figsize=(14, 13))
for row, (label, path) in enumerate(representatives):
    with Image.open(path) as image:
        original = np.asarray(image.convert('RGB'))
    processed = preprocess_rgb(original)
    axes[row, 0].imshow(original)
    axes[row, 0].set_title(f'{label}  Trước lọc màu')
    axes[row, 1].imshow(processed['color_filtered_rgb'])
    axes[row, 1].set_title(f'{label}  Sau lọc màu')
    axes[row, 0].axis('off')
    axes[row, 1].axis('off')
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## Xử lý toàn bộ dữ liệu

Mỗi ảnh đầu ra giữ nguyên stem và nhóm `normal`, `curved` hoặc `adverse`. `canny_input` là đầu ra chính; các thư mục khác dùng để kiểm tra và minh họa.
"""
    ),
    code(
        """
def save_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


records = []
for source_path in image_paths:
    relative = source_path.relative_to(IMAGE_DIR).with_suffix('.png')
    with Image.open(source_path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    processed = preprocess_rgb(rgb)

    save_array(OUTPUT_DIR / 'canny_input' / relative, processed['canny_input'])
    save_array(OUTPUT_DIR / 'resized' / relative, processed['resized_rgb'])
    save_array(OUTPUT_DIR / 'color_masks' / relative, processed['color_mask'])
    save_array(OUTPUT_DIR / 'color_filtered' / relative, processed['color_filtered_rgb'])
    save_array(OUTPUT_DIR / 'gaussian_rgb' / relative, processed['gaussian_rgb'])

    records.append({
        'source': source_path.relative_to(IMAGE_DIR).as_posix(),
        'output': (Path('canny_input') / relative).as_posix(),
        'retained_pixel_ratio': round(float(np.mean(processed['color_mask'] > 0)), 6),
    })

manifest = {
    'processed_images': len(records),
    'config': asdict(CONFIG),
    'base_paper': 'Belhadri et al. (2026), Lane Boundary Detection in Dynamic Environments',
    'proposed_variant': 'RGB to HSV white/yellow filtering before Gaussian; stop before Canny.',
    'handoff': 'Read canny_input/*.png in the separate Canny module.',
    'files': records,
}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'preprocessing_manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
)

print(f'Đã xử lý {len(records)} ảnh vào {OUTPUT_DIR}')
"""
    ),
    code(
        """
# Kiểm tra hợp đồng bàn giao. Không có edge map hoặc đường Hough trong đầu ra.
canny_files = sorted((OUTPUT_DIR / 'canny_input').rglob('*.png'))
assert len(canny_files) == 50
with Image.open(canny_files[0]) as image:
    assert image.mode == 'L'
    assert image.size == (CONFIG.target_width, CONFIG.target_height)

archive_path = shutil.make_archive(str(OUTPUT_DIR), 'zip', root_dir=OUTPUT_DIR)
print('Kiểm tra đạt: 50 ảnh PNG grayscale 960 × 540.')
print('File ZIP bàn giao:', archive_path)
"""
    ),
    markdown(
        """
## Ghi chú đánh giá

HSV là một cải tiến **đề xuất** vì dùng được thông tin màu mà grayscale loại bỏ. Để chứng minh tốt hơn bài base, nhóm phải so sánh baseline và phương pháp HSV trên cùng nhãn lane boundary, giữ nguyên Canny, ROI và Hough. Ground truth hiện tại là mask vùng mặt đường, nên không được dùng để kết luận độ chính xác lane boundary.
"""
    ),
]

nbf.write(notebook, OUTPUT)
print(OUTPUT)
