from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class BlueMaskResult:
    detected: bool
    bounds: tuple[int, int, int, int] | None = None


class BlueMaskDetector:
    """Detect a light-blue surgical mask inside a person's face region."""

    def __init__(
        self,
        lower_hsv: tuple[int, int, int] = (82, 35, 75),
        upper_hsv: tuple[int, int, int] = (118, 255, 255),
        minimum_person_area_ratio: float = 0.003,
        maximum_person_area_ratio: float = 0.10,
    ) -> None:
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.minimum_person_area_ratio = minimum_person_area_ratio
        self.maximum_person_area_ratio = maximum_person_area_ratio

    def detect(
        self,
        frame: np.ndarray,
        person_bounds: tuple[int, int, int, int],
    ) -> BlueMaskResult:
        frame_height, frame_width = frame.shape[:2]
        px1, py1, px2, py2 = person_bounds
        px1 = max(0, min(frame_width, px1))
        px2 = max(0, min(frame_width, px2))
        py1 = max(0, min(frame_height, py1))
        py2 = max(0, min(frame_height, py2))
        width = px2 - px1
        height = py2 - py1
        if width <= 0 or height <= 0:
            return BlueMaskResult(False)

        # Restrict color fallback to the head/face area. The lower torso and
        # blue shirt are intentionally excluded.
        x1 = px1 + int(width * 0.05)
        x2 = px1 + int(width * 0.98)
        y1 = py1 + int(height * 0.05)
        y2 = py1 + int(height * 0.50)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return BlueMaskResult(False)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        person_area = width * height
        minimum_area = max(35, int(person_area * self.minimum_person_area_ratio))
        maximum_area = max(minimum_area, int(person_area * self.maximum_person_area_ratio))
        candidates: list[tuple[int, tuple[int, int, int, int]]] = []
        for index in range(1, count):
            component_x, component_y, component_width, component_height, area = (
                int(value) for value in stats[index]
            )
            if minimum_area <= area <= maximum_area:
                candidates.append(
                    (
                        area,
                        (
                            x1 + component_x,
                            y1 + component_y,
                            x1 + component_x + component_width,
                            y1 + component_y + component_height,
                        ),
                    )
                )

        bounds = max(candidates, default=(0, None), key=lambda item: item[0])[1]
        return BlueMaskResult(bounds is not None, bounds)
