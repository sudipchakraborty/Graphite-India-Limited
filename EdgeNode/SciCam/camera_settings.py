from __future__ import annotations

import ipaddress
import json
from pathlib import Path

from .app_paths import application_root


DEFAULT_CAMERA_CONFIG = application_root() / "config" / "camera.json"


class CameraSettings:
    """Persistent RTSP camera address configuration."""

    EZVIZ_URL_TEMPLATE = (
        "rtsp://admin:DPDYWJ@{ip}:554/Streaming/Channels/101"
    )
    RPI_URL_TEMPLATE = "rtsp://{ip}:8554/gibnew"

    def __init__(self, path: Path = DEFAULT_CAMERA_CONFIG) -> None:
        self.path = Path(path)
        self.rtsp_ip = "192.168.0.201"
        self.camera_type = "ezviz"
        self.load()

    @property
    def rtsp_url(self) -> str:
        if self.camera_type == "rpi":
            return self.RPI_URL_TEMPLATE.format(ip=self.rtsp_ip)
        return self.EZVIZ_URL_TEMPLATE.format(ip=self.rtsp_ip)

    @staticmethod
    def validate_ip(value: str) -> str:
        address = ipaddress.ip_address(value.strip())
        if address.version != 4:
            raise ValueError("Please enter an IPv4 address.")
        return str(address)

    def load(self) -> None:
        if not self.path.exists():
            self.save(self.rtsp_ip)
            return
        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        self.rtsp_ip = self.validate_ip(data.get("rtsp_ip", self.rtsp_ip))
        self.camera_type = str(data.get("camera_type", self.camera_type)).lower()
        if self.camera_type not in {"ezviz", "rpi"}:
            self.camera_type = "ezviz"

    def save(self, value: str, camera_type: str | None = None) -> None:
        self.rtsp_ip = self.validate_ip(value)
        if camera_type is not None:
            self.camera_type = str(camera_type).lower()
            if self.camera_type not in {"ezviz", "rpi"}:
                self.camera_type = "ezviz"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "camera_type": self.camera_type,
                    "rtsp_ip": self.rtsp_ip,
                },
                file,
                indent=2,
            )
            file.write("\n")
