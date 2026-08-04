import unittest

import cv2
import numpy as np

from SciCam.blue_mask_detector import BlueMaskDetector


class TestBlueMaskDetector(unittest.TestCase):
    PERSON = (100, 50, 500, 650)

    @staticmethod
    def _frame() -> np.ndarray:
        return np.zeros((720, 640, 3), dtype=np.uint8)

    def test_blue_mask_in_face_region_is_detected(self):
        frame = self._frame()
        cv2.rectangle(frame, (240, 180), (360, 270), (255, 170, 90), -1)

        result = BlueMaskDetector().detect(frame, self.PERSON)

        self.assertTrue(result.detected)
        self.assertIsNotNone(result.bounds)

    def test_blue_shirt_below_face_region_is_ignored(self):
        frame = self._frame()
        cv2.rectangle(frame, (180, 410), (420, 640), (255, 170, 90), -1)

        result = BlueMaskDetector().detect(frame, self.PERSON)

        self.assertFalse(result.detected)

    def test_blue_collar_below_face_cutoff_is_ignored(self):
        frame = self._frame()
        cv2.rectangle(frame, (190, 360), (410, 470), (255, 170, 90), -1)

        result = BlueMaskDetector().detect(frame, self.PERSON)

        self.assertFalse(result.detected)

    def test_side_profile_blue_mask_is_detected(self):
        frame = self._frame()
        cv2.rectangle(frame, (400, 190), (485, 280), (255, 170, 90), -1)

        result = BlueMaskDetector().detect(frame, self.PERSON)

        self.assertTrue(result.detected)

    def test_non_blue_face_covering_is_ignored(self):
        frame = self._frame()
        cv2.rectangle(frame, (240, 180), (360, 270), (0, 0, 255), -1)

        result = BlueMaskDetector().detect(frame, self.PERSON)

        self.assertFalse(result.detected)


if __name__ == "__main__":
    unittest.main()
