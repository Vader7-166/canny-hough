"""Evaluate Hough on the unchanged Module 3 handoff and export experiments.

Run from any directory: py -3.14 scripts/generate_hough_assets.py
Preprocessing and Canny inputs are read-only. The grid is exploratory, not held-out.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import cv2
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import evaluate_road_result, summarize_records
from hough import (
    HoughConfig, HoughResult, apply_roi, detect_hough_lines_p,
    detect_hough_lines_standard, detect_road_boundaries,
    draw_hough_visualization, get_trapezoid_vertices,
)

GROUPS = ('normal', 'curved', 'adverse')


@dataclass(frozen=True)
class Sample:
    group: str
    name: str
    edges: np.ndarray
    rgb: np.ndarray
    ground_truth: np.ndarray


def read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return image


def load_samples(root: Path = ROOT) -> list[Sample]:
    manifest_path = root / 'outputs/preprocessed_for_canny/preprocessing_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    samples = []
    for group in GROUPS:
        sources = sorted((root / 'dataset/images' / group).glob('*.jpg'))
        if not sources:
            raise FileNotFoundError(f'No source images for {group}')
        for path in sources:
            relative = Path(group) / f'{path.stem}.png'
            edges = read_image(root / 'outputs/canny/edges' / relative, cv2.IMREAD_GRAYSCALE)
            rgb = cv2.cvtColor(read_image(root / 'outputs/preprocessed_for_canny/resized' / relative, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            gt_raw = read_image(root / 'dataset/ground_truth' / relative, cv2.IMREAD_GRAYSCALE)
            h, w = edges.shape
            gt = cv2.resize(gt_raw, (w, h), interpolation=cv2.INTER_NEAREST)
            if rgb.shape[:2] != edges.shape:
                raise ValueError(f'Input geometry mismatch for {path.stem}')
            samples.append(Sample(group, path.stem, edges, rgb, gt))
    if len(samples) != manifest['processed_images']:
        raise ValueError('Source images and preprocessing manifest disagree')
    return samples


def evaluate_dataset(samples: list[Sample], config: HoughConfig) -> tuple[pd.DataFrame, list[HoughResult]]:
    rows, results = [], []
    for sample in samples:
        start = time.perf_counter()
        result = detect_road_boundaries(sample.edges, config)
        runtime_ms = (time.perf_counter() - start) * 1000
        rows.append(dict(
            group=sample.group, image=sample.name, raw_lines=len(result.raw_lines),
            left_segments=len(result.left_segments), right_segments=len(result.right_segments),
            runtime_ms=runtime_ms, **evaluate_road_result(result, sample.ground_truth),
        ))
        results.append(result)
    return pd.DataFrame(rows), results


def parameter_grid(samples: list[Sample]) -> pd.DataFrame:
    rows = []
    # Contains the frozen default (25, 25, 25). Never auto-select on the test set.
    for threshold, length, gap in product((15, 25, 35), (20, 25, 45), (15, 25, 35)):
        cfg = HoughConfig(threshold=threshold, min_line_length=length, max_line_gap=gap)
        frame, _ = evaluate_dataset(samples, cfg)
        summary = summarize_records(frame).iloc[-1].to_dict()
        rows.append(dict(threshold=threshold, min_line_length=length, max_line_gap=gap, **summary))
    return pd.DataFrame(rows)


def benchmark_hough(samples: list[Sample], config: HoughConfig) -> pd.DataFrame:
    rows = []
    for sample in samples:
        h, w = sample.edges.shape
        roi, _ = apply_roi(sample.edges, get_trapezoid_vertices(h, w, config))
        timings = {}
        for name, method in [('standard', detect_hough_lines_standard), ('probabilistic', detect_hough_lines_p)]:
            method(roi, config)  # Warm up each method on the same ROI.
            elapsed = []
            for _ in range(5):
                start = time.perf_counter()
                lines = method(roi, config)
                elapsed.append((time.perf_counter() - start) * 1000)
            timings[f'{name}_median_ms'] = float(np.median(elapsed))
            timings[f'{name}_line_count'] = len(lines)
        rows.append(dict(group=sample.group, image=sample.name, **timings))
    return pd.DataFrame(rows)


def save_figures(samples, results, grid, config, output):
    figdir = output / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    index = next(i for i, sample in enumerate(samples) if sample.name == 'normal_05')
    sample, result = samples[index], results[index]
    h, w = sample.edges.shape
    roi, _ = apply_roi(sample.edges, get_trapezoid_vertices(h, w, config))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    panels = [(sample.edges, 'Canny: unchanged Module 3 handoff'), (roi, 'Fixed trapezoid ROI'),
              (draw_hough_visualization(sample.rgb, result, True), 'Slope candidates + consensus fit'),
              (draw_hough_visualization(sample.rgb, result, False, False), 'Ordered road polygon (not semantic proof)')]
    for ax, (image, title) in zip(axes.flat, panels):
        ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
        ax.set_title(title, fontsize=10); ax.axis('off')
    fig.tight_layout(); fig.savefig(figdir / 'figure_1_hough_pipeline_stages.png', dpi=130); plt.close(fig)

    std_canvas, p_canvas = sample.rgb.copy(), sample.rgb.copy()
    for rho, theta in detect_hough_lines_standard(roi, config):
        a, b = np.cos(theta), np.sin(theta)
        x, y = a * rho, b * rho
        cv2.line(std_canvas, (int(x - 2000*b), int(y + 2000*a)), (int(x + 2000*b), int(y - 2000*a)), (255, 80, 80), 1)
    for x1, y1, x2, y2 in result.raw_lines:
        cv2.line(p_canvas, (x1, y1), (x2, y2), (255, 80, 80), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, image, title in zip(axes, [std_canvas, p_canvas], ['Standard Hough: infinite lines', 'P-Hough: finite segments']):
        ax.imshow(image); ax.set_title(title); ax.axis('off')
    fig.tight_layout(); fig.savefig(figdir / 'figure_2_standard_vs_probabilistic.png', dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, gap in zip(axes, (15, 25, 35)):
        sub = grid[grid.max_line_gap == gap]
        pivot = sub.pivot(index='threshold', columns='min_line_length', values='mean_iou_all')
        im = ax.imshow(pivot, vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_xlabel('minLineLength'); ax.set_ylabel('threshold'); ax.set_title(f'maxLineGap={gap}')
    fig.colorbar(im, ax=axes, label='Whole-dataset IoU', shrink=0.8)
    fig.savefig(figdir / 'figure_3_parameter_grid_sensitivity.png', dpi=130, bbox_inches='tight'); plt.close(fig)

    names = ['normal_03', 'normal_05', 'curved_04', 'curved_09', 'adverse_04', 'adverse_19']
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for ax, name in zip(axes.flat, names):
        i = next(i for i, s in enumerate(samples) if s.name == name)
        canvas = draw_hough_visualization(samples[i].rgb, results[i], draw_roi=False)
        contours, _ = cv2.findContours((samples[i].ground_truth > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 0), 1)
        metrics = evaluate_road_result(results[i], samples[i].ground_truth)
        ax.imshow(canvas); ax.set_title(f'{name}: IoU={metrics["iou"]:.3f}; GT=yellow', fontsize=10); ax.axis('off')
    fig.tight_layout(); fig.savefig(figdir / 'figure_4_group_examples.png', dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    cv2.setRNGSeed(0)
    samples = load_samples()
    config = HoughConfig()
    output = ROOT / 'outputs/hough'
    output.mkdir(parents=True, exist_ok=True)
    frame, results = evaluate_dataset(samples, config)
    frame.to_csv(output / 'hough_50_images_results.csv', index=False)
    summary = summarize_records(frame)
    summary.to_csv(output / 'hough_group_statistics.csv', index=False)
    grid = parameter_grid(samples)
    grid.to_csv(output / 'hough_parameters_grid.csv', index=False)
    benchmark_hough(samples, config).to_csv(output / 'hough_method_benchmark.csv', index=False)

    for sample, result in zip(samples, results):
        path = output / 'visualizations' / sample.group / f'{sample.name}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(draw_hough_visualization(sample.rgb, result), cv2.COLOR_RGB2BGR))
    save_figures(samples, results, grid, config, output)
    (output / 'run_metadata.json').write_text(json.dumps({
        'python': platform.python_version(), 'opencv': cv2.__version__, 'numpy': np.__version__,
        'platform': platform.platform(), 'hough_config': asdict(config),
        'input_scope': 'Read-only Module 3 handoff; preprocessing and Canny are not executed or modified.',
        'input_paths': {
            'edges': 'outputs/canny/edges',
            'rgb': 'outputs/preprocessed_for_canny/resized',
            'ground_truth': 'dataset/ground_truth',
        },
        'preprocessing_handoff': json.loads(
            (ROOT / 'outputs/preprocessed_for_canny/preprocessing_manifest.json').read_text(encoding='utf-8')
        )['config'],
        'images': [f'{s.group}/{s.name}' for s in samples], 'random_seed': 0,
        'grid_policy': 'Exploratory on all images; no held-out generalization claim or automatic default selection.',
        'metrics': 'IoU whole image, missing polygon=empty. MAE requires both lines at all 3 rows. SR: MAE<15px over all images.',
        'runtime_scope': 'ROI + P-Hough + consensus + fit + polygon only; excludes preprocessing, Canny and image IO.',
        'benchmark': 'Per image, one warmup + median of five calls per method on the same ROI; excludes postprocessing.',
    }, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
    print('Artifacts:', output)


if __name__ == '__main__':
    main()
