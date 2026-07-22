"""Shared preprocessing/postprocessing helpers.

These intentionally mirror `preprocess_for_yolo`, `remap_from_yolo`,
`to_input_value`, `from_output_value` and the popup template-match thresholds
in `backend/src/detect.rs`, and the fixed solving-region math in
`backend/src/player/solve_shape.rs`. If you change anything here, check
whether the corresponding Rust code needs to change too (and vice versa) —
the exported ONNX model must keep the exact same input/output contract the
Rust inference code assumes.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import cv2
import numpy as np


def _round(x: float) -> int:
    """Round-half-away-from-zero, matching Rust's `f32::round()` (Python's
    builtin `round()` uses round-half-to-even, which can disagree on exact
    .5 ties and silently shift the letterbox padding by a pixel).
    """
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))

# --- Model input/output contract (backend/src/detect.rs) -------------------

LETTERBOX_SIZE = 640
LETTERBOX_PAD_VALUE = 114  # cv::Scalar::all(114.0) in preprocess_for_yolo
CLASS_NAME = "shape"
NUM_CLASSES = 1

# --- Solving region (backend/src/player/solve_shape.rs) --------------------

REGION_OFFSET = (0, 20)  # tl = title.tl() + (0, 20)
REGION_SIZE = (755, 505)  # br = tl + (755, 505)

# --- Popup detection thresholds (backend/src/detect.rs) --------------------

POPUP_MATCH_THRESHOLD = 0.6  # detect_lie_detector_shape uses TM_CCOEFF_NORMED, threshold 0.6


@dataclasses.dataclass
class LetterboxResult:
    image: np.ndarray  # HxWx3 uint8, RGB, letterboxed to LETTERBOX_SIZE^2
    ratio: float  # uniform scale applied to the original image
    left: int
    top: int
    orig_size: tuple[int, int]  # (width, height)


def letterbox(bgr: np.ndarray) -> LetterboxResult:
    """Replicates `preprocess_for_yolo` in backend/src/detect.rs exactly:
    BGR->RGB, uniform-scale resize (min of w/h ratio) with INTER_LINEAR,
    then constant-pad with 114 gray to 640x640.
    """
    h, w = bgr.shape[:2]
    w_ratio = LETTERBOX_SIZE / w
    h_ratio = LETTERBOX_SIZE / h
    ratio = min(w_ratio, h_ratio)

    new_w = _round(w * ratio)
    new_h = _round(h * ratio)

    pad_w = (LETTERBOX_SIZE - new_w) / 2
    pad_h = (LETTERBOX_SIZE - new_h) / 2

    top = _round(pad_h - 0.1)
    bottom = _round(pad_h + 0.1)
    left = _round(pad_w - 0.1)
    right = _round(pad_w + 0.1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE),
    )

    return LetterboxResult(image=padded, ratio=ratio, left=left, top=top, orig_size=(w, h))


def to_model_input(letterboxed: np.ndarray) -> np.ndarray:
    """RGB uint8 HWC [0,255] -> float32 NCHW [0,1], matching `to_input_value`."""
    normalized = letterboxed.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)  # [1, 3, 640, 640]


def remap_box_from_model_space(
    x1: float, y1: float, x2: float, y2: float,
    orig_size: tuple[int, int], ratio: float, left: int, top: int,
) -> tuple[int, int, int, int]:
    """Replicates `remap_from_yolo`: undo letterbox padding/scale, clamp to image bounds."""
    w, h = orig_size
    rx1 = min(max((x1 - left) / ratio, 0.0), w)
    ry1 = min(max((y1 - top) / ratio, 0.0), h)
    rx2 = min(max((x2 - left) / ratio, 0.0), w)
    ry2 = min(max((y2 - top) / ratio, 0.0), h)
    return int(rx1), int(ry1), int(rx2), int(ry2)


def run_onnx_detect(
    session, bgr: np.ndarray, conf_threshold: float = 0.25,
) -> list[tuple[int, int, int, int, float]]:
    """Runs a `transparent_shape_nms.onnx`-contract model on a single BGR frame.

    Returns boxes in original-image pixel coordinates as (x1, y1, x2, y2, conf).
    Assumes an NMS-baked export (Ultralytics `nms=True`) whose output rows are
    `[x1, y1, x2, y2, conf, ...]` in letterboxed model-pixel space — matching
    what `detect_transparent_shapes` in backend/src/detect.rs reads (only
    columns 0..4 are used there, so extra columns like class id are ignored).
    """
    lb = letterbox(bgr)
    model_input = to_model_input(lb.image)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: model_input})
    preds = outputs[0]
    preds = np.squeeze(preds, axis=0) if preds.ndim == 3 else preds

    boxes = []
    for row in preds:
        conf = float(row[4])
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = remap_box_from_model_space(
            float(row[0]), float(row[1]), float(row[2]), float(row[3]),
            lb.orig_size, lb.ratio, lb.left, lb.top,
        )
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2, conf))
    return boxes


def find_popup_region(
    frame_bgr: np.ndarray, templates: list[np.ndarray], threshold: float = POPUP_MATCH_THRESHOLD,
) -> tuple[int, int, int, int] | None:
    """Locates the Lie Detector popup title via TM_CCOEFF_NORMED template match,
    matching `detect_lie_detector_shape`. Returns the *solving region*
    (REGION_OFFSET/REGION_SIZE applied), or None if no template matches.
    """
    best_score = -1.0
    best_box = None
    for template in templates:
        th, tw = template.shape[:2]
        result = cv2.matchTemplate(frame_bgr, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_box = (max_loc[0], max_loc[1], tw, th)

    if best_box is None or best_score < threshold:
        return None

    tx, ty, _, _ = best_box
    rx = tx + REGION_OFFSET[0]
    ry = ty + REGION_OFFSET[1]
    return rx, ry, REGION_SIZE[0], REGION_SIZE[1]


def load_templates(resources_dir: Path) -> list[np.ndarray]:
    names = ["lie_detector_new_ideal_ratio.png", "lie_detector_old_ideal_ratio.png"]
    templates = []
    for name in names:
        path = resources_dir / name
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                templates.append(img)
    if not templates:
        raise FileNotFoundError(
            f"No lie-detector popup templates found in {resources_dir}. "
            f"Expected one of {names}."
        )
    return templates


def crop_region(frame_bgr: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray | None:
    x, y, w, h = region
    fh, fw = frame_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 - x0 < w or y1 - y0 < h:
        return None  # region falls partly outside the frame, skip it
    return frame_bgr[y0:y1, x0:x1]
