"""Label Studio ML backend for the passport-field YOLOv8 model.

The backend receives tasks from Label Studio, runs Ultralytics YOLO inference and
returns regions in Label Studio's ``rectanglelabels`` format.

Environment variables
---------------------
YOLO_MODEL_PATH   Path to .pt weights. If empty, the backend looks for
                  ``labelstudio_backend/models/best.pt`` and then for the newest
                  ``runs/detect/*/weights/best.pt``.
YOLO_DEVICE       ``0`` for CUDA, ``cpu`` for CPU (default: auto).
YOLO_CONF         Default confidence threshold (default: 0.25).
YOLO_IOU          NMS IoU threshold (default: 0.45).
YOLO_IMGSZ        Inference image size (default: 640).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from ultralytics import YOLO

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Label Studio creates a new model instance per request, so weights must be cached
# at module level. Key: resolved weight path; value: (mtime, model).
_MODEL_CACHE: Dict[str, tuple[int, YOLO]] = {}
_MODEL_CACHE_LOCK = threading.Lock()
# Ultralytics keeps a predictor on the model instance, so serialize concurrent calls.
_INFERENCE_LOCK = threading.Lock()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class YOLOBackend(LabelStudioMLBase):
    """Ultralytics YOLOv8 object detection backend for Label Studio."""

    def setup(self) -> None:
        self._model: Optional[YOLO] = None
        self._model_path: Optional[Path] = None
        self._names: Dict[int, str] = {}
        self._conf = _env_float("YOLO_CONF", 0.25)
        self._iou = _env_float("YOLO_IOU", 0.45)
        self._imgsz = _env_int("YOLO_IMGSZ", 640)
        env_device = os.getenv("YOLO_DEVICE", "").strip()
        self._device = env_device or ("0" if torch.cuda.is_available() else "cpu")
        self.set("model_version", "not-loaded")

    def _resolve_model_path(self) -> Path:
        env_path = os.getenv("YOLO_MODEL_PATH", "").strip()
        candidates: list[Path] = []
        if env_path:
            candidates.append(Path(env_path).expanduser())

        candidates.append(BACKEND_DIR / "models" / "best.pt")

        # Local layout: <project>/runs next to <project>/labelstudio_backend.
        # Docker layout: the backend is copied to /app, so runs is mounted at
        # /app/runs (BACKEND_DIR), not at PROJECT_ROOT/runs.
        seen: set[Path] = set()
        for runs_dir in (PROJECT_ROOT / "runs" / "detect", BACKEND_DIR / "runs" / "detect"):
            if not runs_dir.is_dir():
                continue
            for path in sorted(
                runs_dir.glob("*/weights/best.pt"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            ):
                if path not in seen:
                    seen.add(path)
                    candidates.append(path)

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        checked = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            "YOLO weights were not found. Set YOLO_MODEL_PATH or put best.pt into "
            f"labelstudio_backend/models/. Checked:\n{checked}"
        )

    def _get_model(self) -> YOLO:
        if self._model is not None:
            return self._model

        path = self._resolve_model_path()
        cache_key = str(path)
        mtime = int(path.stat().st_mtime)

        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None and cached[0] == mtime:
                logger.debug("Reusing cached YOLO model: %s", path)
                self._model = cached[1]
            else:
                logger.info("Loading YOLO model: %s (device=%s)", path, self._device)
                model = YOLO(str(path))
                _MODEL_CACHE[cache_key] = (mtime, model)
                self._model = model
                logger.info(
                    "Model labels: %s",
                    [str(value) for value in model.names.values()],
                )

        self._names = {int(key): str(value) for key, value in self._model.names.items()}
        self._model_path = path
        self.set("model_version", f"{path.name}:{mtime}")
        return self._model

    def _detect_controls(self) -> list:
        controls = [
            control
            for control in self.label_interface.controls
            if control.tag == "RectangleLabels" and control.to_name
        ]
        if not controls:
            raise ValueError(
                "No <RectangleLabels> control connected to an <Image> was found in "
                "the Label Studio labeling config."
            )
        return controls

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> ModelResponse:
        model = self._get_model()
        controls = self._detect_controls()
        model_version = self.get("model_version") or "unknown"
        predictions = []

        for task in tasks:
            regions = []
            for control in controls:
                from_name = control.name
                to_name = control.to_name[0]
                image_key = control.objects[0].value_name
                image_ref = task.get("data", {}).get(image_key)
                if not image_ref:
                    logger.warning("Task %s has no image in '%s'", task.get("id"), image_key)
                    continue

                path = (
                    image_ref
                    if os.path.exists(image_ref)
                    else self.get_local_path(image_ref, task_id=task.get("id"))
                )

                conf = float(
                    control.attr.get("model_score_threshold")
                    or control.attr.get("score_threshold")
                    or self._conf
                )
                label_map = self.build_label_map(from_name, list(self._names.values()))

                with _INFERENCE_LOCK:
                    results = model.predict(
                        source=path,
                        conf=conf,
                        iou=self._iou,
                        imgsz=self._imgsz,
                        device=self._device,
                        verbose=False,
                    )
                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    continue

                xywhn = boxes.xywhn.cpu().tolist()
                scores = boxes.conf.cpu().tolist()
                classes = boxes.cls.cpu().tolist()

                for (x, y, w, h), score, cls in zip(xywhn, scores, classes):
                    model_label = self._names.get(int(cls))
                    if model_label not in label_map:
                        continue
                    regions.append(
                        {
                            "from_name": from_name,
                            "to_name": to_name,
                            "type": "rectanglelabels",
                            "value": {
                                "rectanglelabels": [label_map[model_label]],
                                "x": (x - w / 2) * 100,
                                "y": (y - h / 2) * 100,
                                "width": w * 100,
                                "height": h * 100,
                                "rotation": 0,
                            },
                            "score": float(score),
                        }
                    )

            all_scores = [region["score"] for region in regions if "score" in region]
            predictions.append(
                {
                    "result": regions,
                    "score": sum(all_scores) / max(len(all_scores), 1),
                    "model_version": model_version,
                }
            )

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """Retraining is done outside Label Studio (see notebooks/train_yolov8.ipynb)."""
        logger.warning(
            "fit() is not implemented: retrain YOLO manually and restart the backend "
            "to load the new best.pt"
        )
        return False
