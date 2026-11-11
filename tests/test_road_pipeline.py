import unittest

import cv2
import numpy as np

from canny import canny_edges, canny_edges_auto, gradient_percentile_thresholds
from evaluation import evaluate_road_mask
from road_pipeline import RoadConfig, detect_hood_curve, normalize_contrast, refine_mask, segment_road


class AdaptiveCannyTests(unittest.TestCase):
    def test_thresholds_follow_percentile_and_ratio(self):
        mag = np.arange(100, dtype=float).reshape(10, 10)
        low, high = gradient_percentile_thresholds(mag, high_percentile=90, low_ratio=0.4)
        self.assertAlmostEqual(high, np.percentile(mag, 90))
        self.assertAlmostEqual(low, 0.4 * high)

    def test_thresholds_have_floor_on_flat_image(self):
        low, high = gradient_percentile_thresholds(np.zeros((5, 5)))
        self.assertEqual(high, 8.0)

    def test_low_contrast_step_found_only_by_auto_thresholds(self):
        img = np.full((40, 40), 20.0)
        img[:, 20:] = 32.0                       # buoc sang 12 muc xam: anh toi, tuong phan thap
        self.assertEqual(int(canny_edges(img, 50, 150, 1.4).max()), 0)
        edges, (low, high) = canny_edges_auto(img)
        self.assertGreater(int((edges[:, 18:23] > 0).sum()), 30)
        self.assertLess(high, 150)


class NormalizeContrastTests(unittest.TestCase):
    def test_dark_image_is_brightened(self):
        rng = np.random.default_rng(0)
        dark = rng.integers(5, 30, size=(60, 80, 3), dtype=np.uint8)
        out, applied = normalize_contrast(dark)
        self.assertTrue(applied)
        self.assertGreater(out.mean(), dark.mean() + 10)

    def test_bright_image_is_unchanged(self):
        bright = np.full((60, 80, 3), 180, np.uint8)
        out, applied = normalize_contrast(bright)
        self.assertFalse(applied)
        self.assertIs(out, bright)


class SegmentRoadTests(unittest.TestCase):
    def test_grey_road_on_green_background(self):
        h, w = 540, 960
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[:] = (40, 140, 40)                                  # co
        rgb[: int(0.45 * h)] = (120, 170, 230)                  # troi
        road = np.zeros((h, w), np.uint8)
        cv2.fillPoly(road, [np.array([[150, h - 1], [440, 250], [520, 250], [810, h - 1]], np.int32)], 1)
        rgb[road > 0] = (95, 95, 100)                           # mat duong xam
        rng = np.random.default_rng(1)
        rgb = np.clip(rgb.astype(int) + rng.integers(-6, 7, rgb.shape), 0, 255).astype(np.uint8)
        pred = segment_road(rgb, (480.0, 245.0), h - 1, RoadConfig(gc_bottom_ratio=0.0))
        # Phu du mat duong va khong vuot len tren duong chan troi. (Han che da biet: khoi tao
        # theo nem tu diem tu khong ngan duoc vung tran sang hai ben khi nen dong mau.)
        self.assertGreater(evaluate_road_mask(pred.astype(np.uint8), road)["recall"], 0.95)
        self.assertEqual(int(pred[:240].sum()), 0)

    def test_colour_gated_init_separates_grass(self):
        h, w = 540, 960
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[:] = (40, 140, 40)
        rgb[: int(0.45 * h)] = (120, 170, 230)
        road = np.zeros((h, w), np.uint8)
        cv2.fillPoly(road, [np.array([[150, h - 1], [440, 250], [520, 250], [810, h - 1]], np.int32)], 1)
        rgb[road > 0] = (95, 95, 100)
        pred = segment_road(rgb, (480.0, 245.0), h - 1, RoadConfig(gc_bottom_ratio=0.0, gc_color_k=5.0))
        self.assertGreater(evaluate_road_mask(pred.astype(np.uint8), road)["iou"], 0.9)


def _scene_with_hood(hood: bool):
    """Duong xam tren nen co; neu hood=True them nap capo toi mau, mep tren la cung parabol."""
    h, w = 540, 960
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (40, 140, 40)
    rgb[: int(0.45 * h)] = (120, 170, 230)
    road = np.zeros((h, w), np.uint8)
    cv2.fillPoly(road, [np.array([[60, h - 1], [440, 250], [520, 250], [900, h - 1]], np.int32)], 1)
    rgb[road > 0] = (95, 95, 100)
    curve = None
    if hood:
        u = (np.arange(w) - w / 2) / (w / 2)
        curve = 470 + 40 * u ** 2                               # cao nhat o giua (y = 470), thap dan ra hai mep
        below = np.mgrid[0:h, 0:w][0] >= curve[None]
        rgb[below] = (25, 30, 45)                               # capo xanh den
    rng = np.random.default_rng(2)
    rgb = np.clip(rgb.astype(int) + rng.integers(-5, 6, rgb.shape), 0, 255).astype(np.uint8)
    return rgb, curve


class HoodCurveTests(unittest.TestCase):
    def test_parabolic_hood_edge_is_found(self):
        rgb, curve = _scene_with_hood(True)
        found = detect_hood_curve(rgb, (480.0, 245.0), 539)
        self.assertIsNotNone(found)
        cols = slice(200, 760)                                  # phan giua: mep capo nam duoi mat duong
        self.assertLess(float(np.abs(found[cols] - curve[cols]).max()), 6.0)

    def test_no_hood_means_no_cut(self):
        rgb, _ = _scene_with_hood(False)
        self.assertIsNone(detect_hood_curve(rgb, (480.0, 245.0), 539))

    def test_refine_cuts_mask_below_hood(self):
        rgb, curve = _scene_with_hood(True)
        road = np.zeros(rgb.shape[:2], bool)
        road[300:, 100:860] = True                              # mat na GrabCut lan ca xuong capo
        cfg = RoadConfig(close_size=0, ray_cut=False)
        out = refine_mask(road, rgb, (480.0, 245.0), 539, None, cfg)
        self.assertFalse(out[480:, 470:490].any())              # duoi mep capo o giua: bi cat
        self.assertTrue(out[300:460, 470:490].all())            # tren mep capo: giu nguyen

    def test_refine_disabled_returns_same_mask(self):
        rgb, _ = _scene_with_hood(True)
        road = np.zeros(rgb.shape[:2], bool)
        road[300:, 100:860] = True
        cfg = RoadConfig(hood_curve=False, close_size=0, ray_cut=False)
        self.assertTrue(np.array_equal(refine_mask(road, rgb, (480.0, 245.0), 539, None, cfg), road))


class EvaluateRoadMaskTests(unittest.TestCase):
    def test_perfect_and_empty_prediction(self):
        gt = np.zeros((100, 100), np.uint8)
        gt[50:, 20:80] = 255
        perfect = evaluate_road_mask(gt, gt)
        self.assertEqual(perfect["iou"], 1.0)
        self.assertEqual(perfect["mae"], 0.0)
        empty = evaluate_road_mask(np.zeros_like(gt), gt)
        self.assertEqual(empty["iou"], 0.0)
        self.assertIsNone(empty["mae"])


if __name__ == "__main__":
    unittest.main()
