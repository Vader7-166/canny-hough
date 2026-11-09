"""So sanh anh canh Canny truoc / sau khi chuan hoa tuong phan, theo dieu kien anh sang.

Chi so giong bao cao Phan 3: recall = ti le diem tren bien mat na long duong (GT) co canh
Canny trong ban kinh 5 px; density = ti le pixel canh. Nhan dieu kien lay tu
outputs/canny/conditions.csv (chi dung de nhom ket qua, khong dung trong pipeline).

Chay tu thu muc goc: python scripts/compare_canny_contrast.py
Ket qua: outputs/pipeline_v2/canny_contrast_by_condition.csv
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from canny import canny_edges, canny_edges_auto  # noqa: E402
from road_pipeline import _read_rgb, normalize_contrast, road_prior  # noqa: E402
from preprocessing import resize_rgb  # noqa: E402

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


def boundary(gt):
    b = cv2.morphologyEx(gt.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    b[:8], b[-8:], b[:, :8], b[:, -8:] = False, False, False, False
    return b


def recall_density(edges, b):
    near = cv2.dilate((edges > 0).astype(np.uint8), KERNEL) > 0
    return (b & near).sum() / max(b.sum(), 1), float((edges > 0).mean())


def main():
    cond = pd.read_csv(ROOT / "outputs/canny/conditions.csv").set_index("name")["condition"]
    rows = []
    for path in sorted((ROOT / "dataset/images").rglob("*.jpg")):
        g, n = path.parent.name, path.stem
        gt = cv2.imread(str(ROOT / "dataset/ground_truth" / g / f"{n}.png"), 0)
        b = boundary(cv2.resize(gt, (960, 540), interpolation=cv2.INTER_NEAREST) > 0)
        rgb = resize_rgb(_read_rgb(path), 960, 540)
        vp, _, _ = road_prior(rgb)
        region = np.zeros((540, 960), bool)
        region[int(vp[1]):] = True
        hsv_input = cv2.imread(str(ROOT / "outputs/preprocessed_for_canny/canny_input" / g / f"{n}.png"), 0)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.cvtColor(normalize_contrast(rgb, force=True)[0], cv2.COLOR_RGB2GRAY)
        variants = {
            "goc: loc HSV + (50,150)": canny_edges(hsv_input, 50, 150, 1.4),
            "xam + (50,150)": canny_edges(gray, 50, 150, 1.4),
            "chuan hoa + (50,150)": canny_edges(enhanced, 50, 150, 1.4),
            "chuan hoa + nguong phan vi": canny_edges_auto(enhanced, region=region)[0],
        }
        for name, edges in variants.items():
            r, d = recall_density(edges, b)
            rows.append(dict(image=n, condition=cond[n], variant=name, recall=r, density=d))
        print(n)
    df = pd.DataFrame(rows)
    table = df.pivot_table(index="condition", columns="variant", values=["recall", "density"], aggfunc="mean")
    overall = df.groupby("variant")[["recall", "density"]].mean().T.stack().rename("tat ca")
    table.loc["tat ca"] = overall.reindex(table.columns)
    out = ROOT / "outputs/pipeline_v2"
    out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / "canny_contrast_by_condition.csv")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
