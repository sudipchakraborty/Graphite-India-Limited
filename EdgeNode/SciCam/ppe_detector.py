from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .app_paths import application_root
from .blue_glove_detector import BlueGloveDetector
from .blue_mask_detector import BlueMaskDetector


@dataclass(frozen=True, slots=True)
class PPEBox:
    bounds: tuple[int, int, int, int]
    label: str
    confidence: float
    color: tuple[int, int, int]


@dataclass(slots=True)
class PPEResult:
    person_present: bool
    missing_items: tuple[str, ...]
    confirmed: bool
    boxes: list[PPEBox] = field(default_factory=list)
    state: str = "checking"
    observed_missing_items: tuple[str, ...] = ()
    transition_count: int = 0
    transition_required: int = 10

    @property
    def event_name(self) -> str:
        if not self.confirmed or not self.missing_items:
            return ""
        return f"Danger Zone: {' and '.join(self.missing_items)}"


class PPEHysteresis:
    """Require consecutive observations before changing stable PPE state."""

    def __init__(self, required_frames: int = 10) -> None:
        self.required_frames = max(1, int(required_frames))
        self.state = "checking"
        self.stable_missing_items: tuple[str, ...] = ()
        self.positive_streak = 0
        self.negative_streak = 0

    def update(
        self,
        person_present: bool,
        missing_items: tuple[str, ...],
    ) -> tuple[str, tuple[str, ...], int]:
        if not person_present:
            self.state = "checking"
            self.stable_missing_items = ()
            self.positive_streak = 0
            self.negative_streak = 0
            return self.state, self.stable_missing_items, 0

        if missing_items:
            self.negative_streak += 1
            self.positive_streak = 0
            if self.negative_streak >= self.required_frames:
                self.state = "alert"
                self.stable_missing_items = missing_items
                self.negative_streak = self.required_frames
            return self.state, self.stable_missing_items, self.negative_streak

        self.positive_streak += 1
        self.negative_streak = 0
        if self.positive_streak >= self.required_frames:
            self.state = "ok"
            self.stable_missing_items = ()
            self.positive_streak = self.required_frames
        return self.state, self.stable_missing_items, self.positive_streak


class PPEDetector:
    """Demo full-frame person, mask, and glove compliance detector."""

    RELEVANT_LABELS = {
        "person",
        "mask",
        "no_mask",
        "gloves",
        "no_gloves",
    }

    def __init__(
        self,
        model_path: Path | None = None,
        confidence: float = 0.20,
        person_confidence: float = 0.60,
        confirmation_frames: int = 10,
    ) -> None:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "PPE detection requires torch and ultralytics."
            ) from error

        self.model_path = model_path or (
            application_root() / "models" / "ppe_demo.pt"
        )
        self.person_model_path = application_root() / "models" / "yolov8n.pt"
        self.mask_model_path = application_root() / "models" / "person_mask.pt"
        if not self.model_path.is_file():
            raise RuntimeError(f"PPE model not found: {self.model_path}")
        if not self.person_model_path.is_file():
            raise RuntimeError(f"Person model not found: {self.person_model_path}")
        if not self.mask_model_path.is_file():
            raise RuntimeError(f"Mask model not found: {self.mask_model_path}")

        original_torch_load = torch.load

        def load_trusted_model(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = load_trusted_model
        try:
            self.model = YOLO(str(self.model_path))
            self.person_model = YOLO(str(self.person_model_path))
            self.mask_model = YOLO(str(self.mask_model_path))
        finally:
            torch.load = original_torch_load

        self.confidence = confidence
        self.person_confidence = person_confidence
        self.blue_glove_detector = BlueGloveDetector()
        self.blue_mask_detector = BlueMaskDetector()
        self.confirmation_frames = max(1, int(confirmation_frames))
        self.hysteresis = PPEHysteresis(self.confirmation_frames)

    @staticmethod
    def _normalise(label: str) -> str:
        return label.strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _belongs_to_person(
        item_bounds: tuple[int, int, int, int],
        person_bounds: tuple[int, int, int, int],
    ) -> bool:
        """Return whether the PPE item's centre lies inside a person box."""
        item_x1, item_y1, item_x2, item_y2 = item_bounds
        person_x1, person_y1, person_x2, person_y2 = person_bounds
        item_center_x = (item_x1 + item_x2) / 2
        item_center_y = (item_y1 + item_y2) / 2
        return (
            person_x1 <= item_center_x <= person_x2
            and person_y1 <= item_center_y <= person_y2
        )

    @classmethod
    def _ppe_belongs_to_person(
        cls,
        label: str,
        item_bounds: tuple[int, int, int, int],
        person_bounds: tuple[int, int, int, int],
    ) -> bool:
        if not cls._belongs_to_person(item_bounds, person_bounds):
            return False
        if label not in {"mask", "no_mask"}:
            return True

        item_x1, item_y1, item_x2, item_y2 = item_bounds
        person_x1, person_y1, person_x2, person_y2 = person_bounds
        item_width = item_x2 - item_x1
        item_height = item_y2 - item_y1
        person_width = person_x2 - person_x1
        person_height = person_y2 - person_y1
        item_center_y = (item_y1 + item_y2) / 2

        return (
            item_center_y <= person_y1 + person_height * 0.42
            and item_width <= person_width * 0.60
            and item_height <= person_height * 0.50
        )

    @classmethod
    def _missing_items_for_people(
        cls,
        person_boxes: list[tuple[int, int, int, int]],
        ppe_detections: list[
            tuple[str, tuple[int, int, int, int]]
        ],
    ) -> tuple[str, ...]:
        """Require positive mask and glove detections for every person."""
        missing_gloves = False
        missing_mask = False

        for person_bounds in person_boxes:
            labels = {
                label
                for label, item_bounds in ppe_detections
                if cls._ppe_belongs_to_person(
                    label,
                    item_bounds,
                    person_bounds,
                )
            }
            if "no_gloves" in labels or "gloves" not in labels:
                missing_gloves = True
            if "no_mask" in labels or "mask" not in labels:
                missing_mask = True

        missing_items: list[str] = []
        if missing_gloves:
            missing_items.append("No Gloves")
        if missing_mask:
            missing_items.append("No Mask")
        return tuple(missing_items)

    def process(self, frame: np.ndarray) -> PPEResult:
        person_prediction = self.person_model.predict(
            source=frame,
            conf=self.person_confidence,
            classes=[0],
            imgsz=640,
            device=0,
            verbose=False,
        )[0]
        mask_prediction = self.mask_model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=640,
            device=0,
            verbose=False,
        )[0]
        ppe_prediction = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=640,
            device=0,
            verbose=False,
        )[0]

        boxes: list[PPEBox] = []
        person_boxes: list[tuple[int, int, int, int]] = []
        ppe_detections: list[
            tuple[str, tuple[int, int, int, int]]
        ] = []
        colors = {
            "person": (255, 180, 0),
            "mask": (0, 200, 0),
            "gloves": (0, 200, 0),
            "no_mask": (0, 0, 255),
            "no_gloves": (0, 0, 255),
        }

        predictions = (
            (person_prediction, self.person_model, True),
            (mask_prediction, self.mask_model, False),
            (ppe_prediction, self.model, False),
        )
        for prediction, model, is_person_source in predictions:
            if prediction.boxes is None:
                continue
            for box in prediction.boxes:
                class_id = int(box.cls.item())
                label = self._normalise(str(model.names[class_id]))
                if label not in self.RELEVANT_LABELS:
                    continue
                bounds = tuple(
                    int(value) for value in box.xyxy[0].tolist()
                )
                if is_person_source and label == "person":
                    person_boxes.append(bounds)
                elif label != "person":
                    ppe_detections.append((label, bounds))
                boxes.append(
                    PPEBox(
                        bounds=bounds,
                        label=label.replace("_", " ").title(),
                        confidence=float(box.conf.item()),
                        color=colors[label],
                    )
                )

        for person_bounds in person_boxes:
            blue_mask = self.blue_mask_detector.detect(frame, person_bounds)
            if (
                blue_mask.detected
                and blue_mask.bounds is not None
                and self._ppe_belongs_to_person(
                    "mask",
                    blue_mask.bounds,
                    person_bounds,
                )
            ):
                ppe_detections.append(("mask", blue_mask.bounds))
                boxes.append(
                    PPEBox(
                        bounds=blue_mask.bounds,
                        label="Blue Mask",
                        confidence=1.0,
                        color=(0, 200, 0),
                    )
                )

            blue_gloves = self.blue_glove_detector.detect(frame, person_bounds)
            blue_bounds = (
                blue_gloves.left_bounds,
                blue_gloves.right_bounds,
            )
            for side, bounds in zip(("L", "R"), blue_bounds):
                if bounds is not None:
                    boxes.append(
                        PPEBox(
                            bounds=bounds,
                            label=f"Blue Glove {side}",
                            confidence=1.0,
                            color=(0, 200, 0),
                        )
                    )
            if blue_gloves.both_hands_detected:
                for bounds in blue_bounds:
                    if bounds is not None:
                        ppe_detections.append(("gloves", bounds))

        boxes = [
            item
            for item in boxes
            if self._normalise(item.label) not in {"mask", "no_mask"}
            or any(
                self._ppe_belongs_to_person(
                    self._normalise(item.label),
                    item.bounds,
                    person_bounds,
                )
                for person_bounds in person_boxes
            )
        ]

        person_present = bool(person_boxes)
        missing_items = self._missing_items_for_people(
            person_boxes,
            ppe_detections,
        )

        state, stable_missing_items, transition_count = self.hysteresis.update(
            person_present,
            missing_items,
        )

        return PPEResult(
            person_present=person_present,
            missing_items=stable_missing_items,
            confirmed=state == "alert",
            boxes=boxes,
            state=state,
            observed_missing_items=missing_items,
            transition_count=transition_count,
            transition_required=self.confirmation_frames,
        )

    @staticmethod
    def draw(frame: np.ndarray, result: PPEResult | None) -> np.ndarray:
        output = frame.copy()
        height, width = output.shape[:2]

        if result is None:
            border_color = (0, 200, 255)
            status = "DANGER ZONE - PPE CHECK STARTING"
        elif not result.person_present:
            border_color = (0, 200, 255)
            status = "DANGER ZONE"
        elif result.state == "alert":
            border_color = (0, 0, 255)
            status = f"ALERT: {' + '.join(result.missing_items)}"
        elif result.state == "ok":
            border_color = (0, 200, 0)
            status = "DANGER ZONE - PPE OK"
        else:
            border_color = (0, 165, 255)
            observation = (
                " + ".join(result.observed_missing_items)
                if result.observed_missing_items
                else "PPE OK"
            )
            status = (
                f"VERIFYING {result.transition_count}/"
                f"{result.transition_required}: {observation}"
            )

        cv2.rectangle(output, (3, 3), (width - 4, height - 4), border_color, 6)
        cv2.rectangle(output, (8, 8), (min(width - 8, 650), 58), (0, 0, 0), -1)
        cv2.putText(
            output,
            status,
            (18, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            border_color,
            2,
            cv2.LINE_AA,
        )

        if result is not None:
            for item in result.boxes:
                x1, y1, x2, y2 = item.bounds
                cv2.rectangle(output, (x1, y1), (x2, y2), item.color, 2)
                cv2.putText(
                    output,
                    f"{item.label} {item.confidence:.0%}",
                    (x1, max(28, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    item.color,
                    2,
                    cv2.LINE_AA,
                )
        return output
