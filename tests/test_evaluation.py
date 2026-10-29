import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from evaluation import evaluate_road_result, summarize_records
from hough import HoughResult


class RoadMetricTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((100, 100), np.uint8)
        self.gt[50:99, 20:81] = 255
        empty = np.empty((0, 4), np.int32)
        self.result = HoughResult(
            left_line=((20, 98), (20, 50)), right_line=((80, 98), (80, 50)),
            polygon=np.array([[20, 98], [20, 50], [80, 50], [80, 98]], np.int32),
            raw_lines=empty, left_segments=empty, right_segments=empty,
            roi_mask=np.full((100, 100), 255, np.uint8),
        )

    def test_matching_area_and_boundaries_are_successful(self):
        row = evaluate_road_result(self.result, self.gt)
        self.assertEqual(row['iou'], 1.0)
        self.assertEqual(row['mae'], 0.0)
        self.assertEqual(row['success_mae15'], 1)

    def test_missing_boundary_is_not_zero_error_or_success(self):
        row = evaluate_road_result(replace(self.result, right_line=None, polygon=None), self.gt)
        self.assertEqual(row['iou'], 0.0)
        self.assertIsNone(row['mae'])
        self.assertEqual(row['success_mae15'], 0)

    def test_partial_vertical_support_does_not_extrapolate_mae(self):
        short = replace(self.result, left_line=((20, 98), (20, 70)))
        self.assertIsNone(evaluate_road_result(short, self.gt)['mae'])

    def test_self_intersecting_polygon_predicts_empty_mask(self):
        crossed = np.array([[20, 98], [80, 50], [20, 50], [80, 98]], np.int32)
        row = evaluate_road_result(replace(self.result, polygon=crossed), self.gt)
        self.assertEqual(row['polygon_available'], 0)
        self.assertEqual(row['iou'], 0.0)
        self.assertIsNone(row['mae'])

    def test_success_threshold_is_strict_and_all_samples_are_required(self):
        shifted = replace(
            self.result,
            left_line=((35, 98), (35, 50)), right_line=((95, 98), (95, 50)),
            polygon=self.result.polygon + np.array([15, 0]),
        )
        row = evaluate_road_result(shifted, self.gt)
        self.assertEqual(row['mae'], 15.0)
        self.assertEqual(row['success_mae15'], 0)
        self.gt[75] = 0
        self.assertIsNone(evaluate_road_result(self.result, self.gt)['mae'])

    def test_failures_remain_in_iou_coverage_and_success_denominators(self):
        good = evaluate_road_result(self.result, self.gt)
        missing = evaluate_road_result(replace(self.result, left_line=None, polygon=None), self.gt)
        records = pd.DataFrame([
            dict(group='normal', raw_lines=2, runtime_ms=1, **good),
            dict(group='normal', raw_lines=1, runtime_ms=1, **missing),
        ])
        summary = summarize_records(records).set_index('group').loc['all']
        self.assertEqual(summary['mean_iou_all'], 0.5)
        self.assertEqual(summary['mean_iou_with_polygon'], 1.0)
        self.assertEqual(summary['success_mae15_pct'], 50.0)
        self.assertEqual(summary['mae_evaluable_count'], 1)
        self.assertEqual(summary['mean_mae_evaluable'], 0.0)


if __name__ == '__main__':
    unittest.main()
