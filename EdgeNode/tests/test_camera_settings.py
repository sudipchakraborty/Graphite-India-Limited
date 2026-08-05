import unittest

from SciCam.camera_settings import CameraSettings


class CameraSettingsTests(unittest.TestCase):
    def test_camera_settings_supports_both_rtsp_stream_templates(self):
        path = "./tmp_camera_settings_test.json"

        settings = CameraSettings(path)
        self.assertEqual(
            settings.rtsp_url,
            "rtsp://admin:DPDYWJ@192.168.0.201:554/Streaming/Channels/101",
        )

        settings.camera_type = "rpi"
        settings.rtsp_ip = "192.168.1.97"
        self.assertEqual(settings.rtsp_url, "rtsp://192.168.1.97:8554/gibnew")

        settings.save("192.168.1.97")
        loaded = CameraSettings(path)
        self.assertEqual(loaded.camera_type, "rpi")
        self.assertEqual(loaded.rtsp_url, "rtsp://192.168.1.97:8554/gibnew")


if __name__ == "__main__":
    unittest.main()
