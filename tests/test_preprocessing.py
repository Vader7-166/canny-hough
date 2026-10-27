import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing import (
    PreprocessConfig,
    lane_marking_mask,
    preprocess_rgb,
    process_dataset,
    rgb_to_hsv_opencv,
)


class ColorConversionTests(unittest.TestCase):
    def test_rgb_to_hsv_uses_opencv_ranges(self):
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)

        hsv = rgb_to_hsv_opencv(rgb)

        np.testing.assert_array_equal(hsv[0, :, 0], np.array([0, 60, 120], dtype=np.uint8))
        np.testing.assert_array_equal(hsv[0, :, 1:], np.full((3, 2), 255, dtype=np.uint8))

    def test_lane_marking_mask_keeps_white_and_yellow(self):
        rgb = np.array(
            [[
                [230, 230, 230],  # white lane marking
                [220, 180, 30],   # yellow lane marking
                [125, 122, 118],  # asphalt
                [30, 150, 45],    # vegetation
                [35, 35, 35],     # too dark
            ]],
            dtype=np.uint8,
        )
        hsv = rgb_to_hsv_opencv(rgb)

        mask = lane_marking_mask(hsv)

        np.testing.assert_array_equal(mask, np.array([[255, 255, 0, 0, 0]], dtype=np.uint8))


class PipelineTests(unittest.TestCase):
    def test_pipeline_stops_at_canny_ready_grayscale_image(self):
        rgb = np.zeros((60, 100, 3), dtype=np.uint8)
        rgb[25:, :] = [130, 128, 125]
        config = PreprocessConfig(target_width=80, target_height=40, gaussian_radius=1.0)

        result = preprocess_rgb(rgb, config)

        self.assertEqual(result.resized_rgb.shape, (40, 80, 3))
        self.assertEqual(result.hsv.shape, (40, 80, 3))
        self.assertEqual(result.color_mask.shape, (40, 80))
        self.assertEqual(result.canny_input.shape, (40, 80))
        self.assertEqual(result.canny_input.dtype, np.uint8)
        self.assertFalse(hasattr(result, "roi_mask"))
        self.assertFalse(hasattr(result, "edges"))
        self.assertFalse(hasattr(result, "hough_lines"))

    def test_dataset_processing_preserves_group_structure_and_writes_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "images"
            output_dir = root / "processed"
            (input_dir / "normal").mkdir(parents=True)
            image = np.full((30, 50, 3), [120, 118, 115], dtype=np.uint8)
            Image.fromarray(image, mode="RGB").save(input_dir / "normal" / "sample.jpg")

            summary = process_dataset(
                input_dir,
                output_dir,
                PreprocessConfig(target_width=40, target_height=20),
                save_intermediate=True,
            )

            self.assertEqual(summary["processed_images"], 1)
            self.assertTrue((output_dir / "canny_input" / "normal" / "sample.png").is_file())
            self.assertTrue((output_dir / "color_filtered" / "normal" / "sample.png").is_file())
            manifest = json.loads((output_dir / "preprocessing_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["processed_images"], 1)
            self.assertEqual(manifest["config"]["target_width"], 40)
            manifest_output = output_dir / manifest["files"][0]["output"]
            self.assertTrue(manifest_output.is_file())
            self.assertEqual(
                manifest["files"][0]["output"],
                "canny_input/normal/sample.png",
            )


if __name__ == "__main__":
    unittest.main()
