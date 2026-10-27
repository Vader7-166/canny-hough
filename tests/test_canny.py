import unittest

import numpy as np

from canny import canny_edges, gaussian_kernel, non_max_suppression, sobel_gradient


class CannyTests(unittest.TestCase):
    def test_gaussian_kernel_sums_to_one(self):
        k = gaussian_kernel(5, 1.4)
        self.assertAlmostEqual(k.sum(), 1.0)
        self.assertEqual(int(np.argmax(k)), 2)

    def test_sobel_vertical_step(self):
        img = np.zeros((7, 7))
        img[:, 4:] = 100
        mag, theta = sobel_gradient(img)
        self.assertEqual(mag[3, 3], 400)
        self.assertEqual(mag[3, 0], 0)
        self.assertAlmostEqual(theta[3, 3], 0)

    def test_nms_makes_edge_one_pixel_wide(self):
        img = np.zeros((9, 9))
        img[:, 5:] = 100
        mag, theta = sobel_gradient(img)
        nms = non_max_suppression(mag, theta)
        self.assertEqual(int((nms[4] > 0).sum()), 1)

    def test_hysteresis_keeps_weak_connected_to_strong(self):
        img = np.zeros((20, 20))
        img[:, 10:] = 200
        edges = canny_edges(img, 50, 150, 1.0)
        self.assertGreater(edges[10].sum(), 0)
        self.assertEqual(edges[:, :6].sum(), 0)

    def test_flat_image_has_no_edges(self):
        self.assertEqual(canny_edges(np.full((20, 20), 90.0), 50, 150, 1.0).sum(), 0)


if __name__ == "__main__":
    unittest.main()
