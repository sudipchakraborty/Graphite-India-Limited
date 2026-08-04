import unittest

import cv2
import numpy as np

from SciCam.blue_glove_detector import BlueGloveDetector


class TestBlueGloveDetector(unittest.TestCase):
    PERSON = (100, 50, 500, 650)

    @staticmethod
    def _frame() -> np.ndarray:
        return np.zeros((720, 640, 3), dtype=np.uint8)

    def test_detects_blue_regions_on_both_hands(self):
        frame = self._frame()
        cv2.rectangle(frame, (130, 220), (190, 330), (255, 80, 0), -1)
        cv2.rectangle(frame, (410, 220), (470, 330), (255, 80, 0), -1)

        result = BlueGloveDetector().detect(frame, self.PERSON)

        self.assertTrue(result.both_hands_detected)
        self.assertIsNotNone(result.left_bounds)
        self.assertIsNotNone(result.right_bounds)

    def test_one_blue_hand_is_not_compliant(self):
        frame = self._frame()
        cv2.rectangle(frame, (130, 220), (190, 330), (255, 80, 0), -1)

        result = BlueGloveDetector().detect(frame, self.PERSON)

        self.assertTrue(result.left_detected)
        self.assertFalse(result.right_detected)
        self.assertFalse(result.both_hands_detected)

    def test_central_blue_shirt_is_ignored(self):
        frame = self._frame()
        cv2.rectangle(frame, (275, 200), (325, 550), (255, 80, 0), -1)

        result = BlueGloveDetector().detect(frame, self.PERSON)

        self.assertFalse(result.left_detected)
        self.assertFalse(result.right_detected)

    def test_non_blue_regions_are_ignored(self):
        frame = self._frame()
        cv2.rectangle(frame, (130, 220), (190, 330), (0, 0, 255), -1)
        cv2.rectangle(frame, (410, 220), (470, 330), (0, 255, 0), -1)

        result = BlueGloveDetector().detect(frame, self.PERSON)

        self.assertFalse(result.both_hands_detected)


if __name__ == "__main__":
    unittest.main()
