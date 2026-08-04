import unittest

from SciCam.ppe_detector import PPEDetector


class TestPPEDetectorCompliance(unittest.TestCase):
    PERSON = (100, 50, 500, 700)

    def test_person_with_gloves_and_mask_has_no_alert(self):
        detections = [
            ("mask", (220, 80, 300, 170)),
            ("gloves", (150, 300, 230, 400)),
        ]

        self.assertEqual(
            PPEDetector._missing_items_for_people([self.PERSON], detections),
            (),
        )

    def test_explicit_no_gloves_alerts(self):
        detections = [
            ("mask", (220, 80, 300, 170)),
            ("no_gloves", (150, 300, 230, 400)),
        ]

        self.assertEqual(
            PPEDetector._missing_items_for_people([self.PERSON], detections),
            ("No Gloves",),
        )

    def test_explicit_no_mask_alerts(self):
        detections = [
            ("gloves", (150, 300, 230, 400)),
            ("no_mask", (220, 80, 300, 170)),
        ]

        self.assertEqual(
            PPEDetector._missing_items_for_people([self.PERSON], detections),
            ("No Mask",),
        )

    def test_explicit_negative_detection_alerts(self):
        detections = [
            ("mask", (220, 80, 300, 170)),
            ("gloves", (150, 300, 230, 400)),
            ("no_gloves", (350, 300, 430, 400)),
        ]

        self.assertEqual(
            PPEDetector._missing_items_for_people([self.PERSON], detections),
            ("No Gloves",),
        )

    def test_undetected_gloves_are_not_ppe_ok(self):
        detections = [("mask", (220, 80, 300, 170))]

        self.assertEqual(
            PPEDetector._missing_items_for_people([self.PERSON], detections),
            ("No Gloves",),
        )

    def test_no_ppe_detections_alert_for_both_items(self):
        self.assertEqual(
            PPEDetector._missing_items_for_people([self.PERSON], []),
            ("No Gloves", "No Mask"),
        )

    def test_large_mask_box_outside_face_region_is_rejected(self):
        detections = [
            ("mask", (0, 100, 620, 650)),
            ("gloves", (150, 300, 230, 400)),
        ]

        self.assertEqual(
            PPEDetector._missing_items_for_people([self.PERSON], detections),
            ("No Mask",),
        )

    def test_compact_mask_box_in_face_region_is_accepted(self):
        detections = [
            ("mask", (220, 90, 330, 210)),
            ("gloves", (150, 300, 230, 400)),
        ]

        self.assertEqual(
            PPEDetector._missing_items_for_people([self.PERSON], detections),
            (),
        )

    def test_any_noncompliant_person_triggers_alert(self):
        people = [(0, 0, 300, 700), (400, 0, 700, 700)]
        detections = [
            ("mask", (100, 50, 180, 140)),
            ("gloves", (80, 300, 150, 390)),
            ("mask", (500, 50, 580, 140)),
            ("no_gloves", (480, 300, 550, 390)),
        ]

        self.assertEqual(
            PPEDetector._missing_items_for_people(people, detections),
            ("No Gloves",),
        )


if __name__ == "__main__":
    unittest.main()
