"""
2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) Detector
Detects RF signals from spectrogram power data using separable 2D moving-average box filtering,
adaptive local noise floor estimation, morphological closing, and contour bounding box extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class CFARConfig:
    threshold_factor: float = 1.10
    guard_rows: int = 12
    guard_cols: int = 25
    train_rows: int = 15
    train_cols: int = 15
    morph_kernel: int = 5
    min_area: int = 20
    max_boxes: int = 32
    target_category_id: int = 1


def ca_cfar_2d(
    spectrogram: np.ndarray,
    guard_cells: Tuple[int, int] = (12, 25),
    train_cells: Tuple[int, int] = (15, 15),
    threshold_factor: float = 1.10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized 2D Cell-Averaging CFAR using separable O(1) box filtering.
    Computes:
        local_noise_floor = (BoxSum_outer - BoxSum_inner) / num_training_cells
    
    Returns:
        (detections_bool_mask, local_noise_floor_matrix)
    """
    gr_r, gr_c = int(guard_cells[0]), int(guard_cells[1])
    tr_r, tr_c = int(train_cells[0]), int(train_cells[1])

    kr_r = 1 + (2 * gr_r) + (2 * tr_r)
    kr_c = 1 + (2 * gr_c) + (2 * tr_c)
    gr_inner_r = 1 + (2 * gr_r)
    gr_inner_c = 1 + (2 * gr_c)

    num_train_cells = float(kr_r * kr_c - gr_inner_r * gr_inner_c)
    if num_train_cells <= 0:
        num_train_cells = 1.0

    # Ensure float32 for high precision convolution
    if spectrogram.dtype != np.float32:
        spectrogram = spectrogram.astype(np.float32)

    # Box filters are O(1) per pixel with separable moving average and symmetric reflection
    outer_sum = cv2.boxFilter(
        spectrogram, cv2.CV_32F, (kr_c, kr_r),
        normalize=False, borderType=cv2.BORDER_REFLECT
    )
    inner_sum = cv2.boxFilter(
        spectrogram, cv2.CV_32F, (gr_inner_c, gr_inner_r),
        normalize=False, borderType=cv2.BORDER_REFLECT
    )

    local_noise_floor = (outer_sum - inner_sum) / num_train_cells
    detections = spectrogram > (local_noise_floor * threshold_factor)

    return detections, local_noise_floor


class CFARDetector:
    """
    High-Performance 2D CA-CFAR Detector for Spectrogram Images.
    """

    def __init__(self, config: Optional[CFARConfig] = None):
        self.config = config or CFARConfig()

    @staticmethod
    def normalize_channel(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.dtype != np.float32:
            gray = gray.astype(np.float32)
        return gray

    def detect_boxes(self, image_or_power: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run 2D CA-CFAR detection and return proposal annotations:
        [{ id, category_id, x, y, width, height, confidence, source: "cfar", isProposal: True }]
        """
        cfg = self.config
        spectrogram = self.normalize_channel(image_or_power)
        height, width = spectrogram.shape[:2]

        guard_cells = (int(cfg.guard_rows), int(cfg.guard_cols))
        train_cells = (int(cfg.train_rows), int(cfg.train_cols))

        detections, noise_floor = ca_cfar_2d(
            spectrogram=spectrogram,
            guard_cells=guard_cells,
            train_cells=train_cells,
            threshold_factor=float(cfg.threshold_factor)
        )

        # Binary detection mask
        mask = (detections * 255).astype(np.uint8)

        # Morphology closing to bridge sub-carrier and time-slice gaps
        if cfg.morph_kernel > 1:
            k_size = int(cfg.morph_kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            mask_closed = mask

        # Find external contours (connected components)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_candidates = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area >= cfg.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Ignore full-frame bounding box background artifacts
                if (w >= width * 0.95 and h >= height * 0.95) or area >= (width * height * 0.80):
                    continue

                # Ensure within image boundaries
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                w = min(width - x1, int(w))
                h = min(height - y1, int(h))

                if w >= 2 and h >= 2:
                    # Estimate confidence based on ratio of peak power over local noise floor in ROI
                    roi_spec = spectrogram[y1:y1 + h, x1:x1 + w]
                    roi_noise = noise_floor[y1:y1 + h, x1:x1 + w]
                    if roi_noise.size > 0 and roi_noise.mean() > 0:
                        ratio = float(roi_spec.mean() / (roi_noise.mean() + 1e-6))
                        conf = min(0.99, max(0.50, (ratio - 1.0) * 0.5 + 0.65))
                    else:
                        conf = 0.75

                    raw_candidates.append({
                        "area": area,
                        "x": x1,
                        "y": y1,
                        "width": w,
                        "height": h,
                        "confidence": round(conf, 2)
                    })

        # Sort candidate bounding boxes by area descending
        raw_candidates.sort(key=lambda item: item["area"], reverse=True)
        selected_candidates = raw_candidates[: cfg.max_boxes]

        proposals = []
        for idx, item in enumerate(selected_candidates):
            proposal_id = f"cfar_prop_{idx}_{np.random.randint(1000, 9999)}"
            proposals.append({
                "id": proposal_id,
                "category_id": cfg.target_category_id,
                "x": item["x"],
                "y": item["y"],
                "width": item["width"],
                "height": item["height"],
                "confidence": item["confidence"],
                "source": "cfar",
                "isProposal": True
            })

        return proposals
