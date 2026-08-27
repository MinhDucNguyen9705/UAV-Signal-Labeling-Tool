"""
ONNX AI Model Inference Engine for Spectrogram Object Detection
Supports YOLOv5, YOLOv8, YOLOv10, YOLOv11, YOLO26, and generic ONNX object detection architectures
with letterbox preprocessing, NMS on letterboxed space, and exact Ultralytics coordinate scaling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort


def letterbox(im: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
    """
    Resizes and pads image to new_shape maintaining aspect ratio.
    """
    shape = im.shape[:2]  # [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert [cx, cy, w, h] bounding box format to [x1, y1, x2, y2].
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_boxes(img1_shape: Tuple[int, int], boxes: np.ndarray, img0_shape: Tuple[int, int]) -> np.ndarray:
    """
    Rescale boxes (xyxy) from letterbox model space (e.g. 640x640) to original image size.
    Matches Ultralytics coordinate transform logic.
    """
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain        # scale back

    # Clip boxes to original image boundaries
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1] - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0] - 1)
    return boxes


@dataclass
class ONNXModelInfo:
    model_name: str
    filepath: str
    input_name: str
    input_shape: List[Any]
    output_names: List[str]
    num_classes: int
    class_names: List[str]
    input_width: int = 640
    input_height: int = 640
    is_end2end: bool = False


class ONNXDetector:
    """
    Manages ONNX runtime inference sessions for object detection on spectrograms.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.session: Optional[ort.InferenceSession] = None
        self.model_info: Optional[ONNXModelInfo] = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str) -> ONNXModelInfo:
        """
        Load an ONNX model file and parse metadata and input/output tensor configurations.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        # Use CPU execution provider by default for maximum portability
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)

        # Inspect Inputs
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError("ONNX model has no input tensors.")
        input_tensor = inputs[0]
        input_name = input_tensor.name
        input_shape = input_tensor.shape

        # Resolve input resolution (default 640x640 if dynamic)
        input_height = 640
        input_width = 640
        if len(input_shape) >= 4:
            h = input_shape[2]
            w = input_shape[3]
            if isinstance(h, int) and h > 0:
                input_height = h
            if isinstance(w, int) and w > 0:
                input_width = w

        # Inspect Outputs
        outputs = self.session.get_outputs()
        output_names = [o.name for o in outputs]

        # Extract custom metadata if present in ONNX
        meta = self.session.get_modelmeta().custom_metadata_map
        class_names = []
        if "names" in meta:
            try:
                import ast
                parsed_names = ast.literal_eval(meta["names"])
                if isinstance(parsed_names, dict):
                    class_names = [parsed_names[k] for k in sorted(parsed_names.keys(), key=lambda x: int(x))]
                elif isinstance(parsed_names, list):
                    class_names = parsed_names
            except Exception:
                try:
                    import json
                    parsed_names = json.loads(meta["names"].replace("'", '"'))
                    if isinstance(parsed_names, dict):
                        class_names = [parsed_names[k] for k in sorted(parsed_names.keys(), key=lambda x: int(x))]
                    elif isinstance(parsed_names, list):
                        class_names = parsed_names
                except Exception:
                    pass

        # Check for End-to-End detection format (e.g. YOLOv10 / YOLO11 / YOLO26 / RT-DETR)
        is_end2end = meta.get("end2end", "").lower() in ["true", "1"]
        if outputs:
            out_shape = outputs[0].shape
            if len(out_shape) == 3 and out_shape[-1] == 6 and out_shape[-2] == 300:
                is_end2end = True
            elif len(out_shape) == 2 and out_shape[-1] == 6 and out_shape[-2] == 300:
                is_end2end = True

        num_classes = len(class_names) if class_names else 0

        # If class_names was not in metadata, infer num_classes from shape
        if num_classes == 0:
            if is_end2end:
                num_classes = 1
            elif outputs:
                out_shape = outputs[0].shape
                if len(out_shape) == 3:
                    d1, d2 = out_shape[1], out_shape[2]
                    if isinstance(d1, int) and isinstance(d2, int):
                        channel_dim = min(d1, d2)
                        num_anchors = max(d1, d2)
                        if channel_dim == 6 and num_anchors == 300:
                            is_end2end = True
                            num_classes = 1
                        elif channel_dim >= 5:
                            num_classes = channel_dim - 4
                    elif isinstance(d1, int) and d1 >= 5:
                        num_classes = d1 - 4
                    elif isinstance(d2, int) and d2 >= 5:
                        num_classes = d2 - 4
                elif len(out_shape) == 2 and isinstance(out_shape[1], int) and out_shape[1] >= 5:
                    num_classes = out_shape[1] - 4

        if num_classes <= 0:
            num_classes = 1

        if not class_names:
            class_names = [f"Class {i}" for i in range(num_classes)]
        elif len(class_names) < num_classes:
            while len(class_names) < num_classes:
                class_names.append(f"Class {len(class_names)}")
        elif len(class_names) > num_classes:
            num_classes = len(class_names)

        self.model_info = ONNXModelInfo(
            model_name=os.path.basename(model_path),
            filepath=model_path,
            input_name=input_name,
            input_shape=list(input_shape),
            output_names=output_names,
            num_classes=num_classes,
            class_names=class_names,
            input_width=input_width,
            input_height=input_height,
            is_end2end=is_end2end
        )

        return self.model_info

    def preprocess(self, image: np.ndarray, target_w: int = 640, target_h: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Convenience preprocessing helper returning (blob, scale, (pad_x, pad_y)).
        """
        orig_img = image
        if len(orig_img.shape) == 2:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
        lb = letterbox(orig_img, (target_h, target_w))
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.expand_dims(np.transpose(rgb, (2, 0, 1)), axis=0)
        gain = min(target_h / orig_img.shape[0], target_w / orig_img.shape[1])
        pad_x = (target_w - orig_img.shape[1] * gain) / 2.0
        pad_y = (target_h - orig_img.shape[0] * gain) / 2.0
        return blob, gain, (int(round(pad_x)), int(round(pad_y)))

    def detect(
        self,
        image: np.ndarray,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        default_category_id: int = 1,
        class_mapping: Optional[Dict[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run inference on image and return list of proposal bounding boxes.
        Matches test.py PyTorch/Ultralytics parity.
        """
        if self.session is None or self.model_info is None:
            raise RuntimeError("No ONNX model loaded. Please load a model first.")

        orig_img = image
        if len(orig_img.shape) == 2:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

        orig_shape = orig_img.shape[:2]  # [height, width]
        target_shape = (self.model_info.input_height, self.model_info.input_width)

        # 1. Letterbox Preprocessing
        img_letterbox = letterbox(orig_img, new_shape=target_shape)
        img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(np.transpose(img_normalized, (2, 0, 1)), axis=0)

        # 2. Forward inference pass
        outputs = self.session.run(None, {self.model_info.input_name: input_tensor})
        predictions = outputs[0]

        # Squeeze batch dimension
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]

        # Transpose so rows are predictions (e.g. [300, 6] or [8400, 84])
        if predictions.ndim == 2:
            if predictions.shape[0] < predictions.shape[1]:
                predictions = predictions.T

        if len(predictions) == 0:
            return []

        # 3. Extract Boxes, Scores, and Class IDs
        # Check if model is End-to-End format: [x1, y1, x2, y2, score, class]
        if self.model_info.is_end2end or predictions.shape[-1] == 6:
            boxes_xyxy = predictions[:, :4].copy()
            scores = predictions[:, 4].copy()
            class_ids = predictions[:, 5].astype(int)
        elif predictions.shape[-1] == 5 and self.model_info.num_classes == 1:
            # 1-class raw YOLOv8 format: [cx, cy, w, h, score]
            boxes_xywh = predictions[:, :4]
            scores = predictions[:, 4].copy()
            class_ids = np.zeros(len(predictions), dtype=int)
            boxes_xyxy = xywh2xyxy(boxes_xywh)
        else:
            # Standard raw YOLO format: [cx, cy, w, h, score_0, score_1, ..., score_N]
            boxes_xywh = predictions[:, :4]
            class_scores = predictions[:, 4:]
            if class_scores.shape[1] > 0:
                scores = np.max(class_scores, axis=1)
                class_ids = np.argmax(class_scores, axis=1)
            else:
                scores = np.ones(len(predictions))
                class_ids = np.zeros(len(predictions), dtype=int)
            boxes_xyxy = xywh2xyxy(boxes_xywh)

        # 4. Filter by confidence threshold
        valid_indices = scores > conf_thresh
        boxes_xyxy = boxes_xyxy[valid_indices]
        scores = scores[valid_indices]
        class_ids = class_ids[valid_indices]

        if len(scores) == 0:
            return []

        # 5. Non-Maximum Suppression (NMS) ON THE LETTERBOX MODEL SPACE
        boxes_for_nms = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            boxes_for_nms.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

        nms_indices = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            scores.tolist(),
            score_threshold=float(conf_thresh),
            nms_threshold=float(iou_thresh)
        )

        if len(nms_indices) == 0:
            return []

        final_indices = nms_indices.flatten()
        final_boxes_xyxy = boxes_xyxy[final_indices]
        final_scores = scores[final_indices]
        final_classes = class_ids[final_indices]

        # 6. Rescale boxes back to original image dimensions AFTER NMS
        final_boxes_scaled = scale_boxes(target_shape, final_boxes_xyxy, orig_shape)

        proposals = []
        for idx, (box, score, raw_cls) in enumerate(zip(final_boxes_scaled, final_scores, final_classes)):
            x1, y1, x2, y2 = box
            w = max(2.0, float(x2 - x1))
            h = max(2.0, float(y2 - y1))
            x = max(0.0, min(float(orig_shape[1] - 1), float(x1)))
            y = max(0.0, min(float(orig_shape[0] - 1), float(y1)))

            # Clip width and height within image boundaries
            w = min(float(orig_shape[1] - x), w)
            h = min(float(orig_shape[0] - y), h)

            raw_cls_int = int(raw_cls)
            if class_mapping is not None:
                if raw_cls_int in class_mapping:
                    cat_id = int(class_mapping[raw_cls_int])
                elif str(raw_cls_int) in class_mapping:
                    cat_id = int(class_mapping[str(raw_cls_int)])
                else:
                    cat_id = default_category_id
            else:
                cat_id = default_category_id

            prop_id = f"onnx_prop_{idx}_{np.random.randint(1000, 9999)}"
            proposals.append({
                "id": prop_id,
                "category_id": cat_id,
                "x": round(x, 1),
                "y": round(y, 1),
                "width": round(w, 1),
                "height": round(h, 1),
                "confidence": round(float(score), 3),
                "source": "onnx",
                "isProposal": True
            })

        return proposals
