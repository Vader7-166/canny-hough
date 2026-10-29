"""Kiem thu tu dong cho module Hough (Phan 4)."""

import unittest

import cv2
import numpy as np

from hough import (
    HoughConfig,
    apply_roi,
    create_road_polygon,
    detect_road_boundaries,
    fit_boundary_line,
    get_trapezoid_vertices,
    split_left_right_segments,
)


class TrapezoidRoiTests(unittest.TestCase):
    def test_vertices_inside_image_bounds_and_symmetric(self):
        height, width = 540, 960
        cfg = HoughConfig()
        vertices = get_trapezoid_vertices(height, width, cfg)

        self.assertEqual(vertices.shape, (1, 4, 2))
        pts = vertices[0]

        # Kiem tra tat ca diem nam trong khung hinh
        self.assertTrue(np.all(pts[:, 0] >= 0))
        self.assertTrue(np.all(pts[:, 0] < width))
        self.assertTrue(np.all(pts[:, 1] >= 0))
        self.assertTrue(np.all(pts[:, 1] < height))

        # Kiem tra doi xung qua truc giua
        center_x = width / 2.0
        # pts[0] la bottom_left, pts[3] la bottom_right
        dist_bl = center_x - pts[0, 0]
        dist_br = pts[3, 0] - center_x
        self.assertAlmostEqual(dist_bl, dist_br, delta=1.0)

        # pts[1] la top_left, pts[2] la top_right
        dist_tl = center_x - pts[1, 0]
        dist_tr = pts[2, 0] - center_x
        self.assertAlmostEqual(dist_tl, dist_tr, delta=1.0)

    def test_apply_roi_zeros_outside(self):
        edges = np.full((100, 200), 255, dtype=np.uint8)
        # Tao hinh thang nho o giua
        vertices = np.array([[[40, 80], [70, 40], [130, 40], [160, 80]]], dtype=np.int32)
        masked, mask = apply_roi(edges, vertices)

        # Cac diem chac chan o ngoai ROI phai bang 0
        self.assertEqual(masked[10, 10], 0)
        self.assertEqual(masked[90, 100], 0)
        # Diem o giua tam ROI phai duoc giu nguyen 255
        self.assertEqual(masked[60, 100], 255)


class LineFilteringAndSplittingTests(unittest.TestCase):
    def test_horizontal_and_vertical_lines_filtered(self):
        cfg = HoughConfig(min_abs_slope=0.4, max_abs_slope=3.0)
        width = 1000

        # Doan 1: nam ngang (dy = 0 => slope = 0)
        horiz = np.array([100, 400, 300, 400], dtype=np.int32)
        # Doan 2: thang dung (dx = 0 => slope = inf)
        vert = np.array([200, 100, 200, 400], dtype=np.int32)

        lines = np.array([horiz, vert], dtype=np.int32)
        left, right = split_left_right_segments(lines, width, cfg)

        self.assertEqual(len(left), 0)
        self.assertEqual(len(right), 0)

    def test_valid_left_and_right_segments(self):
        cfg = HoughConfig(min_abs_slope=0.4, max_abs_slope=3.0)
        width = 1000

        # Doan bien trai: tu (100, 500) toi (350, 300)
        # dx = 250, dy = -200 => slope = -0.8 (am), x_mid = 225 < 650
        left_seg = np.array([100, 500, 350, 300], dtype=np.int32)

        # Doan bien phai: tu (900, 500) toi (650, 300)
        # dx = -250, dy = -200 => slope = 0.8 (duong), x_mid = 775 > 350
        right_seg = np.array([900, 500, 650, 300], dtype=np.int32)

        lines = np.array([left_seg, right_seg], dtype=np.int32)
        left, right = split_left_right_segments(lines, width, cfg)

        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 1)
        np.testing.assert_array_equal(left[0], left_seg)
        np.testing.assert_array_equal(right[0], right_seg)

    def test_wrong_spatial_position_rejected(self):
        cfg = HoughConfig(min_abs_slope=0.4, max_abs_slope=3.0)
        width = 1000

        # Doan co slope duong (giong bien phai) nhung lai nam hoan toan ben trai man hinh (x_mid = 150 < 350)
        seg = np.array([100, 300, 200, 400], dtype=np.int32)
        lines = np.array([seg], dtype=np.int32)
        left, right = split_left_right_segments(lines, width, cfg)

        self.assertEqual(len(left), 0)
        self.assertEqual(len(right), 0)


class BoundaryLineFittingTests(unittest.TestCase):
    def test_fit_collinear_segments(self):
        # Cac doan thang deu nam tren duong x = -0.5 * y + 500
        # y = 500 => x = 250
        # y = 300 => x = 350
        seg1 = np.array([250, 500, 290, 420], dtype=np.int32)
        seg2 = np.array([310, 380, 350, 300], dtype=np.int32)
        segments = np.array([seg1, seg2], dtype=np.int32)

        line = fit_boundary_line(
            segments, y_bottom=500, y_top=300,
            side="left", width=960, config=HoughConfig(),
        )
        self.assertIsNotNone(line)
        (x_bot, y_bot), (x_top, y_top) = line

        self.assertEqual(y_bot, 500)
        self.assertEqual(y_top, 300)
        self.assertAlmostEqual(x_bot, 250, delta=2)
        self.assertAlmostEqual(x_top, 350, delta=2)

    def test_fit_empty_segments_returns_none(self):
        segments = np.empty((0, 4), dtype=np.int32)
        line = fit_boundary_line(
            segments, y_bottom=500, y_top=300,
            side="left", width=960, config=HoughConfig(),
        )
        self.assertIsNone(line)

    def test_disjoint_negative_slopes_do_not_invert_fitted_left_boundary(self):
        # Pooled endpoints have positive covariance despite negative segment slopes.
        segments = np.array([[100, 300, 130, 260], [500, 500, 530, 460]], np.int32)
        line = fit_boundary_line(
            segments, 529, 259, side="left", width=960, config=HoughConfig(),
        )
        self.assertIsNotNone(line)
        self.assertLess(line[0][0], line[1][0])

    def test_consensus_excludes_spatially_separate_outlier(self):
        segments = np.array([
            [100, 500, 150, 400], [150, 400, 200, 300],
            [600, 500, 650, 400],
        ], np.int32)
        line = fit_boundary_line(
            segments, 500, 300, side="left", width=960, config=HoughConfig(),
        )
        self.assertEqual(line, ((100, 500), (200, 300)))


class RoadPolygonTests(unittest.TestCase):
    def test_crossing_above_road_is_trimmed_below_intersection(self):
        polygon = create_road_polygon(((41, 529), (516, 259)), ((700, 529), (363, 259)))
        self.assertIsNotNone(polygon)
        self.assertGreater(polygon[1, 1], 259)
        self.assertLess(polygon[1, 0], polygon[2, 0])
        self.assertTrue(cv2.isContourConvex(polygon))

    def test_reversed_bottom_and_disjoint_vertical_support_are_rejected(self):
        self.assertIsNone(create_road_polygon(((700, 529), (500, 259)), ((41, 529), (300, 259))))
        self.assertIsNone(create_road_polygon(((100, 500), (200, 400)), ((700, 300), (600, 200))))

    def test_different_line_extents_use_only_common_rows(self):
        polygon = create_road_polygon(((100, 500), (300, 300)), ((800, 450), (600, 250)))
        np.testing.assert_array_equal(polygon, [[150, 450], [300, 300], [650, 300], [800, 450]])


class EndToEndPipelineTests(unittest.TestCase):
    def test_synthetic_road_boundaries(self):
        # Tao anh canh nhan tao kich thuoc 540 x 960
        edges = np.zeros((540, 960), dtype=np.uint8)

        # Ve duong bien trai: tu (100, 530) toi (400, 320)
        cv2.line(edges, (100, 530), (400, 320), 255, thickness=2)
        # Ve duong bien phai: tu (860, 530) toi (560, 320)
        cv2.line(edges, (860, 530), (560, 320), 255, thickness=2)

        cfg = HoughConfig(
            threshold=15,
            min_line_length=20.0,
            max_line_gap=10.0,
        )
        res = detect_road_boundaries(edges, cfg)

        self.assertIsNotNone(res.left_line)
        self.assertIsNotNone(res.right_line)
        self.assertIsNotNone(res.polygon)
        self.assertEqual(res.polygon.shape, (4, 2))

        # Kiem tra da giac khong bi suy bien
        area = cv2.contourArea(res.polygon)
        self.assertGreater(area, 10000.0)

    def test_crossing_road_outputs_ordered_in_frame_boundaries(self):
        edges = np.zeros((540, 960), dtype=np.uint8)
        cv2.line(edges, (0, 539), (500, 259), 255, 2)
        cv2.line(edges, (959, 539), (460, 259), 255, 2)
        res = detect_road_boundaries(edges, HoughConfig(roi_bottom_y_ratio=1.0))
        self.assertIsNotNone(res.polygon)
        self.assertTrue(cv2.isContourConvex(res.polygon))
        for bottom, top in (res.left_line, res.right_line):
            self.assertTrue(0 <= bottom[0] < 960 and 0 <= top[0] < 960)
            self.assertTrue(0 <= top[1] < bottom[1] < 540)
        self.assertLess(res.left_line[0][0], res.right_line[0][0])
        self.assertLess(res.left_line[1][0], res.right_line[1][0])

    def test_empty_edges_have_no_road_prediction(self):
        res = detect_road_boundaries(np.zeros((540, 960), np.uint8))
        self.assertIsNone(res.left_line)
        self.assertIsNone(res.right_line)
        self.assertIsNone(res.polygon)



if __name__ == "__main__":
    unittest.main()
