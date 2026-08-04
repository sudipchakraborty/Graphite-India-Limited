from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class BlueGloveResult:
    left_detected: bool
    right_detected: bool
    left_bounds: tuple[int, int, int, int] | None = None
    right_bounds: tuple[int, int, int, int] | None = None

    @property
    def both_hands_detected(self) -> bool:
        return self.left_detected and self.right_detected


class BlueGloveDetector:
    """Detect saturated blue glove-sized regions on both sides of a person."""

    def __init__(
        self,
        lower_hsv: tuple[int, int, int] = (90, 135, 55),
        upper_hsv: tuple[int, int, int] = (135, 255, 255),
        minimum_person_area_ratio: float = 0.0025,
        maximum_person_area_ratio: float = 0.12,
    ) -> None:
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.minimum_person_area_ratio = minimum_person_area_ratio
        self.maximum_person_area_ratio = maximum_person_area_ratio

    @staticmethod
    def _clip_bounds(
        bounds: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = bounds
        return (
            max(0, min(frame_width, x1)),
            max(0, min(frame_height, y1)),
            max(0, min(frame_width, x2)),
            max(0, min(frame_height, y2)),
        )

    def _largest_blue_component(
        self,
        frame: np.ndarray,
        zone: tuple[int, int, int, int],
        person_area: int,
    ) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = zone
        if x2 <= x1 or y2 <= y1:
            return None

        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        minimum_area = max(40, int(person_area * self.minimum_person_area_ratio))
        maximum_area = max(minimum_area, int(person_area * self.maximum_person_area_ratio))
        candidates: list[tuple[int, tuple[int, int, int, int]]] = []
        for index in range(1, component_count):
            component_x, component_y, width, height, area = (
                int(value) for value in stats[index]
            )
            if minimum_area <= area <= maximum_area:
                candidates.append(
                    (
                        area,
                        (
                            x1 + component_x,
                            y1 + component_y,
                            x1 + component_x + width,
                            y1 + component_y + height,
                        ),
                    )
                )
        return max(candidates, default=(0, None), key=lambda item: item[0])[1]

    def detect(
        self,
        frame: np.ndarray,
        person_bounds: tuple[int, int, int, int],
    ) -> BlueGloveResult:
        x1, y1, x2, y2 = self._clip_bounds(person_bounds, frame.shape)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return BlueGloveResult(False, False)

        zone_top = y1 + int(height * 0.08)
        zone_bottom = y1 + int(height * 0.82)
        left_zone = (x1, zone_top, x1 + int(width * 0.43), zone_bottom)
        right_zone = (x1 + int(width * 0.57), zone_top, x2, zone_bottom)
        person_area = width * height

        left_bounds = self._largest_blue_component(
            frame,
            left_zone,
            person_area,
        )
        right_bounds = self._largest_blue_component(
            frame,
            right_zone,
            person_area,
        )
        return BlueGloveResult(
            left_detected=left_bounds is not None,
            right_detected=right_bounds is not None,
            left_bounds=left_bounds,
            right_bounds=right_bounds,
        )
