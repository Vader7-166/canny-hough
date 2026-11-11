"""Ablation cho buoc hau xu ly 3b (pipeline v3): bat / tat tung thanh phan tren cung mat na GrabCut.

GrabCut chi chay mot lan moi anh; cac bien the chi khac nhau o hau xu ly nen so sanh cong bang.
Bao cao IoU tren ca 50 anh, nua "dev" (anh chi so chan, thu tu sap xep ten tep) dung de chon nguong,
va nua "val" (chi so le) chi dung de kiem tra.

Chay tu thu muc goc: python scripts/ablation_pipeline_v3.py
Ket qua: outputs/pipeline_v3/ablation.csv (tung anh) va ablation_summary.csv
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
from preprocessing import resize_rgb  # noqa: E402
from road_pipeline import (RoadConfig, _read_rgb, edge_map, normalize_contrast, refine_mask,  # noqa: E402
                           road_prior, segment_road)

BASE = RoadConfig()
OFF = replace(BASE, hood_curve=False, close_size=0, ray_cut=False)
VARIANTS = {
    "v2 (khong hau xu ly)": None,
    "chi cat capo": replace(OFF, hood_curve=True),
    "chi dong hinh thai": replace(OFF, close_size=BASE.close_size),
    "chi cat theo tia": replace(OFF, ray_cut=True),
    "capo + dong": replace(BASE, ray_cut=False),
    "v3 day du (capo + dong + tia)": BASE,
    "v3 bo capo": replace(BASE, hood_curve=False),
    "v3 bo dong": replace(BASE, close_size=0),
}


def main():
    cond = pd.read_csv(ROOT / "outputs/canny/conditions.csv").set_index("name")["condition"]
    paths = sorted((ROOT / "dataset/images").rglob("*.jpg"))
    rows = []
    for i, path in enumerate(paths):
        g, n = path.parent.name, path.stem
        gt = np.asarray(Image.open(ROOT / "dataset/ground_truth" / g / f"{n}.png").convert("L")
                        .resize((960, 540), Image.Resampling.NEAREST))
        rgb = resize_rgb(_read_rgb(path), 960, 540)
        enhanced, _ = normalize_contrast(rgb)
        vp, hood, _ = road_prior(rgb)
        road = segment_road(enhanced, vp, hood, BASE)
        edges = edge_map(enhanced, vp)
        for name, cfg in VARIANTS.items():
            mask = road if cfg is None else refine_mask(road, enhanced, vp, hood, edges, cfg)
            rows.append(dict(group=g, image=n, condition=cond[n], split="dev" if i % 2 == 0 else "val",
                             variant=name, **evaluate_road_mask(mask.astype(np.uint8), gt)))
        print(n, flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "outputs/pipeline_v3"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablation.csv", index=False)
    summary = df.groupby("variant", sort=False)[["iou", "precision", "recall", "f1"]].mean()
    summary = summary.join(df.pivot_table(index="variant", columns="split", values="iou").add_prefix("iou_"))
    summary = summary.join(df.pivot_table(index="variant", columns="group", values="iou").add_prefix("iou_"))
    base = df[df.variant == "v2 (khong hau xu ly)"].set_index("image")["iou"]
    summary["so_anh_giam_hon_0.02"] = [int((d.set_index("image")["iou"] < base - 0.02).sum())
                                       for _, d in df.groupby("variant", sort=False)]
    summary.to_csv(out / "ablation_summary.csv")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
