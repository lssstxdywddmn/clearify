import unittest

import cv2
import numpy as np

from clarity import assess


class ClarityTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((128, 128))
        self.sharp = (((x // 8 + y // 8) % 2) * 255).astype(np.uint8)

    def test_controlled_blur_reduces_score(self):
        scores = [assess(self.sharp)["laplacian_variance"]]
        for sigma in (1, 3, 8):
            blurred = cv2.GaussianBlur(self.sharp, (0, 0), sigma)
            scores.append(assess(blurred)["laplacian_variance"])
        self.assertTrue(all(a > b for a, b in zip(scores, scores[1:])))

    def test_flat_image_needs_review(self):
        result = assess(np.full((64, 64), 128, dtype=np.uint8))
        self.assertEqual(result["laplacian_variance"], 0)
        self.assertTrue(result["requires_review"])
        self.assertFalse(result["semantic_visibility_assessed"])

    def test_roi_measures_subject_separately(self):
        blurred = cv2.GaussianBlur(self.sharp, (0, 0), 8)
        image = np.concatenate([self.sharp, blurred], axis=1)
        left = assess(image, roi=(0, 0, 128, 128))
        right = assess(image, roi=(128, 0, 256, 128))
        self.assertGreater(left["laplacian_variance"], right["laplacian_variance"])
        self.assertEqual(left["scope"], "roi")

    def test_invalid_roi_and_thresholds_fail(self):
        with self.assertRaises(ValueError):
            assess(self.sharp, roi=(-1, 0, 20, 20))
        with self.assertRaises(ValueError):
            assess(self.sharp, low=100, high=30)


if __name__ == "__main__":
    unittest.main()
