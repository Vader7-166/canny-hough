"""Ablation cho pipeline v2: bat / tat tung cai tien, cung danh gia tren mat na.

Chay tu thu muc goc: python scripts/ablation_pipeline_v2.py
Ket qua: outputs/pipeline_v2/ablation.csv (tung anh) va ablation_summary.csv
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import evaluate_road_mask  # noqa: E402
from road_pipeline import RoadConfig, _read_rgb, normalize_contrast, road_prior, segment_road  # noqa: E402
from preprocessing import resize_rgb  # noqa: E402

BASE = RoadConfig()
VARIANTS = {
    "v2 day du": BASE,
    "bo chuan hoa tuong phan": BASE,
    "bo bo phieu (1 lan GrabCut)": replace(BASE, gc_votes=1),
    "cua so mau co dinh": replace(BASE, gc_seed_follow_vp=False),
    "bo nen chac chan 3% day anh": replace(BASE, gc_bottom_ratio=0.0),
    "chi GrabCut (bo ca 4 cai tien)": replace(BASE, gc_votes=1, gc_seed_follow_vp=False, gc_bottom_ratio=0.0),
}
NO_CONTRAST = {"bo chuan hoa tuong phan", "chi GrabCut (bo ca 4 cai tien)"}


def main():
    cond = pd.read_csv(ROOT / "outputs/canny/conditions.csv").set_index("name")["condition"]
    rows = []
    for path in sorted((ROOT / "dataset/images").rglob("*.jpg")):
        g, n = path.parent.name, path.stem
        gt = np.asarray(Image.open(ROOT / "dataset/ground_truth" / g / f"{n}.png").convert("L")
                        .resize((960, 540), Image.Resampling.NEAREST))
        rgb = resize_rgb(_read_rgb(path), 960, 540)
        enhanced, _ = normalize_contrast(rgb)
        vp, hood, _ = road_prior(rgb)
        for name, cfg in VARIANTS.items():
            image = rgb if name in NO_CONTRAST else enhanced
            mask = segment_road(image, vp, hood, cfg).astype(np.uint8)
            rows.append(dict(group=g, image=n, condition=cond[n], variant=name, **evaluate_road_mask(mask, gt)))
        print(n)
    df = pd.DataFrame(rows)
    out = ROOT / "outputs/pipeline_v2"
    df.to_csv(out / "ablation.csv", index=False)
    summary = df.groupby("variant", sort=False)[["iou", "precision", "recall", "f1"]].mean()
    by_cond = df.pivot_table(index="variant", columns="condition", values="iou", aggfunc="mean")
    summary = summary.join(by_cond.add_prefix("iou_")).join(
        df.groupby("variant").apply(lambda d: int((d["iou"] < 0.1).sum()), include_groups=False).rename("so_anh_iou_duoi_0.1"))
    summary.to_csv(out / "ablation_summary.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
