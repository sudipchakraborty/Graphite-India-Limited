import unittest

from SciCam.ppe_detector import PPEHysteresis


class TestPPEHysteresis(unittest.TestCase):
    def test_requires_ten_positive_frames_for_ok(self):
        hysteresis = PPEHysteresis(10)

        for count in range(1, 10):
            state, _, progress = hysteresis.update(True, ())
            self.assertEqual(state, "checking")
            self.assertEqual(progress, count)

        state, missing, progress = hysteresis.update(True, ())
        self.assertEqual((state, missing, progress), ("ok", (), 10))

    def test_requires_ten_negative_frames_for_alert(self):
        hysteresis = PPEHysteresis(10)

        for _ in range(9):
            state, _, _ = hysteresis.update(True, ("No Mask",))
            self.assertEqual(state, "checking")

        state, missing, progress = hysteresis.update(True, ("No Mask",))
        self.assertEqual(state, "alert")
        self.assertEqual(missing, ("No Mask",))
        self.assertEqual(progress, 10)

    def test_ok_is_retained_until_ten_consecutive_negative_frames(self):
        hysteresis = PPEHysteresis(10)
        for _ in range(10):
            hysteresis.update(True, ())

        for _ in range(9):
            state, missing, _ = hysteresis.update(True, ("No Gloves",))
            self.assertEqual((state, missing), ("ok", ()))

        state, missing, _ = hysteresis.update(True, ("No Gloves",))
        self.assertEqual((state, missing), ("alert", ("No Gloves",)))

    def test_alert_is_retained_until_ten_consecutive_positive_frames(self):
        hysteresis = PPEHysteresis(10)
        for _ in range(10):
            hysteresis.update(True, ("No Mask",))

        for _ in range(9):
            state, missing, _ = hysteresis.update(True, ())
            self.assertEqual((state, missing), ("alert", ("No Mask",)))

        state, missing, _ = hysteresis.update(True, ())
        self.assertEqual((state, missing), ("ok", ()))

    def test_opposite_frame_resets_transition_streak(self):
        hysteresis = PPEHysteresis(10)
        for _ in range(6):
            hysteresis.update(True, ())
        hysteresis.update(True, ("No Gloves",))

        state, _, progress = hysteresis.update(True, ())
        self.assertEqual(state, "checking")
        self.assertEqual(progress, 1)

    def test_no_person_resets_state(self):
        hysteresis = PPEHysteresis(10)
        for _ in range(10):
            hysteresis.update(True, ())

        self.assertEqual(hysteresis.update(False, ()), ("checking", (), 0))


if __name__ == "__main__":
    unittest.main()
