import os
import shutil
import io
import json
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rf_processor import RFProcessor
from backend.sample_generator import RFSampleGenerator
from backend.coco_exporter import COCOBDWExporter, COCOBDWImporter
from backend.yolo_exporter import YOLOExporter
from backend.cfar_detector import CFARDetector, CFARConfig
from backend.onnx_detector import ONNXDetector, ONNXModelInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import uuid
import threading
import cv2
import numpy as np
from backend.logger import logger, get_recent_logs, clear_recent_logs, LOG_FILE_PATH

app = FastAPI(title="RF Spectrogram Bounding Box Tool with BDW Support")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests_and_no_cache(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000.0

    path = request.url.path
    if path.startswith("/js/") or path.startswith("/css/") or path == "/" or path.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    # Structured request logging
    if path.startswith("/api/"):
        if path.startswith("/api/chunks/") and path.endswith("/spectrogram"):
            logger.debug(f"HTTP {request.method} {path} -> {response.status_code} ({duration_ms:.1f}ms)")
        else:
            logger.info(f"HTTP {request.method} {path} -> {response.status_code} ({duration_ms:.1f}ms)")

    return response

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "samples")
MODELS_DIR = os.path.join(UPLOAD_DIR, "models")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Global State
rf_proc = RFProcessor(fs=61.44e6, center_freq=2400.0e6, chunk_duration_ms=1.0, colormap="turbo", colormap_engine="opencv", db_min=-90.0, db_max=0.0)
onnx_detector = ONNXDetector()

# Default RF Signal Classes
classes_state: List[Dict[str, Any]] = [
    {"id": 1, "name": "WiFi_OFDM", "color": "#00e5ff", "type_of_signal": "OFDM", "protocol": "802.11ax"},
    {"id": 2, "name": "FMCW_Radar", "color": "#ff007f", "type_of_signal": "FMCW", "protocol": "Automotive Radar"},
    {"id": 3, "name": "Bluetooth_FHSS", "color": "#00ff66", "type_of_signal": "GFSK_FHSS", "protocol": "Bluetooth 5.0"},
    {"id": 4, "name": "5G_NR_SSB", "color": "#ffaa00", "type_of_signal": "OFDM", "protocol": "5G NR"},
    {"id": 5, "name": "Pulsed_CW", "color": "#bd00ff", "type_of_signal": "CW", "protocol": "Doppler Radar"},
    {"id": 6, "name": "Noise_Jammer", "color": "#ff3333", "type_of_signal": "AWGN_Jammer", "protocol": "EW"}
]

# Annotations dictionary: { "chunk_id": [ { "id": "...", "category_id": 1, "x": 10, "y": 20, "width": 50, "height": 30, "bdw": {...} } ] }
annotations_state: Dict[str, List[Dict[str, Any]]] = {}

# Pydantic Schemas
class STFTConfigRequest(BaseModel):
    fs_hz: Optional[float] = None
    center_freq_hz: Optional[float] = None
    drone_name: Optional[str] = None
    render_width: Optional[int] = None
    render_height: Optional[int] = None
    chunk_duration_ms: Optional[float] = None
    overlap_duration_ms: Optional[float] = None
    default_snr_db: Optional[float] = None
    nfft: Optional[int] = None
    hop_length: Optional[int] = None
    window: Optional[str] = None
    colormap: Optional[str] = None
    colormap_engine: Optional[str] = None
    db_min: Optional[float] = None
    db_max: Optional[float] = None
    eager_threshold: Optional[int] = None
    batch_size: Optional[int] = None

class ClassItem(BaseModel):
    id: int
    name: str
    color: str
    type_of_signal: Optional[str] = "Unknown"
    protocol: Optional[str] = "Generic"

class ClassesListRequest(BaseModel):
    classes: List[ClassItem]

class SaveAnnotationsRequest(BaseModel):
    chunk_id: int
    annotations: List[Dict[str, Any]]

class BDWCalculationRequest(BaseModel):
    chunk_id: int
    bbox: List[float]  # [x, y, w, h]
    img_width: int
    img_height: int
    category_id: Optional[int] = None
    signal_type: Optional[str] = None
    protocol: Optional[str] = None

class AWGNRequest(BaseModel):
    snr_db: float

class CFARAutoLabelRequest(BaseModel):
    scope: str = "all"  # "all" or "current"
    chunk_id: Optional[int] = 0
    target_category_id: int = 1
    threshold_factor: float = 1.10
    guard_rows: int = 12
    guard_cols: int = 25
    train_rows: int = 15
    train_cols: int = 15
    morph_kernel: int = 5
    min_area: int = 20
    max_boxes: int = 32
    stream: bool = False

class ONNXAutoLabelRequest(BaseModel):
    scope: str = "all"  # "all" or "current"
    chunk_id: Optional[int] = 0
    conf_thresh: float = 0.25
    iou_thresh: float = 0.45
    default_category_id: int = 1
    class_mapping: Optional[Dict[int, int]] = None
    stream: bool = False

# Initial sample generation on startup
def init_sample():
    sample_file = os.path.join(UPLOAD_DIR, "demo_capture_61_44mhz.h5")
    if not os.path.exists(sample_file):
        RFSampleGenerator.generate_synthetic_rf(
            fs=61.44e6,
            center_freq=2400.0e6,
            duration_ms=100.0,
            iq_format="float32",
            output_format="h5",
            output_path=sample_file
        )
    rf_proc.load_h5_file(sample_file, fs=61.44e6, center_freq=2400.0e6)

try:
    init_sample()
    logger.info("Demo RF dataset initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing demo sample: {e}")

# API Endpoints
@app.get("/api/session")
def get_session():
    return {
        "summary": rf_proc.get_summary(),
        "chunks": rf_proc.chunks_meta,
        "classes": classes_state,
        "total_annotations": sum(len(v) for v in annotations_state.values())
    }

@app.post("/api/session/config")
def update_config(cfg: STFTConfigRequest):
    global annotations_state
    old_w = rf_proc.render_width
    old_h = rf_proc.render_height

    fs_hz = cfg.fs_hz
    if fs_hz is not None and 0 < fs_hz < 10000.0:
        fs_hz = fs_hz * 1e6
    center_freq_hz = cfg.center_freq_hz
    if center_freq_hz is not None and 0 < center_freq_hz < 100000.0:
        center_freq_hz = center_freq_hz * 1e6
    rf_proc.set_config(
        fs=fs_hz,
        center_freq=center_freq_hz,
        drone_name=cfg.drone_name,
        render_width=cfg.render_width,
        render_height=cfg.render_height,
        chunk_duration_ms=cfg.chunk_duration_ms,
        overlap_duration_ms=cfg.overlap_duration_ms,
        default_snr_db=cfg.default_snr_db,
        nfft=cfg.nfft,
        hop_length=cfg.hop_length,
        window=cfg.window,
        colormap=cfg.colormap,
        colormap_engine=cfg.colormap_engine,
        db_min=cfg.db_min,
        db_max=cfg.db_max
    )
    new_w = rf_proc.render_width
    new_h = rf_proc.render_height

    # Adaptively rescale existing annotation boxes to new resolution
    if (old_w, old_h) != (new_w, new_h) and old_w > 0 and old_h > 0:
        scale_x = new_w / float(old_w)
        scale_y = new_h / float(old_h)
        for chunk_id, boxes in annotations_state.items():
            for box in boxes:
                box["x"] = max(0, round(float(box.get("x", 0)) * scale_x))
                box["y"] = max(0, round(float(box.get("y", 0)) * scale_y))
                box["width"] = min(new_w - box["x"], max(2, round(float(box.get("width", 10)) * scale_x)))
                box["height"] = min(new_h - box["y"], max(2, round(float(box.get("height", 10)) * scale_y)))
                box["img_width"] = new_w
                box["img_height"] = new_h

    logger.info(f"Updated session/STFT configuration: Drone='{rf_proc.drone_name}', Fs={rf_proc.fs/1e6:.2f}MHz, Res={rf_proc.render_width}x{rf_proc.render_height}, SNR={rf_proc.default_snr_db:.1f}dB, NFFT={rf_proc.nfft}.")
    return {
        "status": "success",
        "summary": rf_proc.get_summary(),
        "chunks": rf_proc.chunks_meta,
        "annotations": annotations_state
    }

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    fs: float = Form(61.44e6),
    center_freq: float = Form(2400.0e6),
    drone_name: str = Form("DJI-MAVIC-PRO-3"),
    render_width: int = Form(1024),
    render_height: int = Form(512),
    iq_format: str = Form("float32"),
    default_snr_db: float = Form(18.0),
    apply_awgn: bool = Form(False),
    target_snr_db: Optional[float] = Form(None),
    chunk_duration_ms: float = Form(30.0),
    overlap_duration_ms: float = Form(10.0),
    nfft: int = Form(1024),
    colormap: str = Form("turbo"),
    colormap_engine: str = Form("opencv"),
    window: str = Form("hann")
):
    global annotations_state
    filename = file.filename or "uploaded_signal.iq"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if 0 < center_freq < 100000.0:
        center_freq = center_freq * 1e6

    if 0 < fs < 10000.0:
        fs = fs * 1e6

    awgn_snr = float(target_snr_db) if apply_awgn and target_snr_db is not None else None
    effective_default_snr = awgn_snr if awgn_snr is not None else default_snr_db

    rf_proc.set_config(
        fs=fs,
        center_freq=center_freq,
        drone_name=drone_name,
        render_width=render_width,
        render_height=render_height,
        chunk_duration_ms=chunk_duration_ms,
        overlap_duration_ms=overlap_duration_ms,
        default_snr_db=effective_default_snr,
        nfft=nfft,
        colormap=colormap,
        colormap_engine=colormap_engine,
        window=window
    )

    if filename.endswith(".h5") or filename.endswith(".hdf5"):
        summary = rf_proc.load_h5_file(
            file_path,
            fs=fs,
            center_freq=center_freq,
            drone_name=drone_name,
            render_width=render_width,
            render_height=render_height,
            default_snr_db=effective_default_snr,
            apply_awgn_snr_db=awgn_snr
        )
    else:
        summary = rf_proc.load_iq_file(
            file_path,
            iq_format=iq_format,
            fs=fs,
            center_freq=center_freq,
            drone_name=drone_name,
            render_width=render_width,
            render_height=render_height,
            default_snr_db=effective_default_snr,
            apply_awgn_snr_db=awgn_snr
        )

    # Reset annotations for new file
    annotations_state = {}
    awgn_msg = f", AWGN applied: {awgn_snr}dB" if awgn_snr is not None else ""
    logger.info(f"File '{filename}' processed successfully. Total chunks: {len(rf_proc.chunks_meta)}, duration: {summary['total_duration_ms']:.2f}ms, default SNR: {effective_default_snr}dB{awgn_msg}.")

    return {
        "status": "success",
        "message": f"Successfully loaded {filename}",
        "summary": summary,
        "chunks": rf_proc.chunks_meta
    }

@app.post("/api/generate_sample")
def generate_sample(
    fs: float = Form(61.44e6),
    center_freq: float = Form(2400.0e6),
    drone_name: str = Form("DJI-MAVIC-PRO-3"),
    render_width: int = Form(1024),
    render_height: int = Form(512),
    iq_format: str = Form("float32"),
    duration_ms: float = Form(100.0),
    default_snr_db: float = Form(18.0),
    apply_awgn: bool = Form(False),
    target_snr_db: Optional[float] = Form(None),
    output_format: str = Form("h5"),
    chunk_duration_ms: float = Form(30.0),
    overlap_duration_ms: float = Form(10.0)
):
    global annotations_state
    if 0 < center_freq < 100000.0:
        center_freq = center_freq * 1e6

    if 0 < fs < 10000.0:
        fs = fs * 1e6

    filename = f"sample_{int(fs/1e6)}mhz_{int(duration_ms)}ms.{output_format}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    output_path, gt_boxes = RFSampleGenerator.generate_synthetic_rf(
        fs=fs,
        center_freq=center_freq,
        duration_ms=duration_ms,
        iq_format=iq_format,
        output_format=output_format,
        output_path=file_path
    )

    awgn_snr = float(target_snr_db) if apply_awgn and target_snr_db is not None else None
    effective_default_snr = awgn_snr if awgn_snr is not None else default_snr_db

    rf_proc.set_config(
        fs=fs,
        center_freq=center_freq,
        drone_name=drone_name,
        render_width=render_width,
        render_height=render_height,
        chunk_duration_ms=chunk_duration_ms,
        overlap_duration_ms=overlap_duration_ms,
        default_snr_db=effective_default_snr
    )
    if output_format == "h5":
        summary = rf_proc.load_h5_file(
            output_path,
            fs=fs,
            center_freq=center_freq,
            drone_name=drone_name,
            render_width=render_width,
            render_height=render_height,
            default_snr_db=effective_default_snr,
            apply_awgn_snr_db=awgn_snr
        )
    else:
        summary = rf_proc.load_iq_file(
            output_path,
            iq_format=iq_format,
            fs=fs,
            center_freq=center_freq,
            drone_name=drone_name,
            render_width=render_width,
            render_height=render_height,
            default_snr_db=effective_default_snr,
            apply_awgn_snr_db=awgn_snr
        )

    annotations_state = {}
    logger.info(f"Generated synthetic RF dataset '{filename}' ({duration_ms}ms, Fs: {fs/1e6:.2f}MHz, default SNR: {effective_default_snr}dB, {len(gt_boxes)} synthetic bursts).")

    return {
        "status": "success",
        "message": f"Generated and loaded synthetic dataset ({fs / 1e6} MHz)",
        "summary": summary,
        "chunks": rf_proc.chunks_meta
    }

@app.get("/api/chunks/{chunk_id}")
def get_chunk_info(chunk_id: int):
    if chunk_id < 0 or chunk_id >= len(rf_proc.chunks_meta):
        raise HTTPException(status_code=404, detail="Chunk not found")
    meta = rf_proc.chunks_meta[chunk_id]
    boxes = annotations_state.get(str(chunk_id), [])
    return {
        "chunk": meta,
        "annotations": boxes
    }

@app.get("/api/chunks/{chunk_id}/spectrogram")
def get_spectrogram_image(
    chunk_id: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
    engine: Optional[str] = None,
    image_format: str = "png",
    quality: int = 95
):
    try:
        clean_fmt = image_format.lower().lstrip('.')
        if clean_fmt not in ["png", "jpg", "jpeg"]:
            clean_fmt = "png"
        img_bytes, meta = rf_proc.render_spectrogram_image(
            chunk_id=chunk_id,
            width=width,
            height=height,
            engine=engine,
            image_format=clean_fmt,
            quality=quality
        )
        media_type = "image/jpeg" if clean_fmt in ["jpg", "jpeg"] else "image/png"
        return Response(content=img_bytes, media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chunks/batch/{chunk_id}")
def ensure_batch_cached_endpoint(chunk_id: int):
    if chunk_id < 0 or chunk_id >= len(rf_proc.chunks_meta):
        raise HTTPException(status_code=404, detail="Chunk not found")
    rf_proc.ensure_batch_cached(chunk_id)
    return {
        "status": "success",
        "active_batch_range": list(rf_proc.active_batch_range),
        "render_mode": rf_proc.render_mode
    }

@app.get("/api/classes")
def get_classes():
    return {"classes": classes_state}

@app.post("/api/classes")
def update_classes(req: ClassesListRequest):
    global classes_state
    classes_state = [c.model_dump() if hasattr(c, "model_dump") else c.dict() for c in req.classes]
    return {"status": "success", "classes": classes_state}

@app.get("/api/annotations")
def get_annotations():
    return {"annotations": annotations_state}

@app.post("/api/annotations")
def save_annotations(req: SaveAnnotationsRequest):
    annotations_state[str(req.chunk_id)] = req.annotations
    logger.info(f"Saved {len(req.annotations)} annotations for chunk {req.chunk_id}.")
    return {"status": "success", "chunk_id": req.chunk_id, "count": len(req.annotations)}

@app.get("/api/logs")
def get_logs_endpoint(limit: int = 100, level: Optional[str] = None):
    """Retrieve structured logs from memory ring buffer."""
    logs = get_recent_logs(limit=limit, min_level=level)
    return {
        "status": "success",
        "log_file": LOG_FILE_PATH,
        "count": len(logs),
        "logs": logs
    }

@app.get("/api/logs/download")
def download_logs():
    """Download the full persistent backend.log file."""
    if not os.path.exists(LOG_FILE_PATH):
        raise HTTPException(status_code=404, detail="Log file not found")
    return FileResponse(path=LOG_FILE_PATH, media_type="text/plain", filename="backend.log")

@app.delete("/api/logs")
def clear_logs_endpoint():
    """Clear memory logs and truncate log file."""
    clear_recent_logs()
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "w") as f:
            f.truncate(0)
    logger.info("Logs cleared by user request.")
    return {"status": "success", "message": "Logs cleared"}

@app.post("/api/annotations/calculate_bdw")
def calculate_bdw(req: BDWCalculationRequest):
    cat_info = next((c for c in classes_state if c["id"] == req.category_id), None)
    
    if req.signal_type and req.signal_type != "Unknown":
        sig_type = req.signal_type
    elif cat_info and cat_info.get("type_of_signal") and cat_info["type_of_signal"] != "Unknown":
        sig_type = cat_info["type_of_signal"]
    else:
        sig_type = req.signal_type or (cat_info["type_of_signal"] if cat_info else "Unknown")

    if req.protocol and req.protocol != "Generic":
        proto = req.protocol
    elif cat_info and cat_info.get("protocol") and cat_info["protocol"] != "Generic":
        proto = cat_info["protocol"]
    else:
        proto = req.protocol or (cat_info["protocol"] if cat_info else "Generic")

    bdw = rf_proc.calculate_bdw_parameters(
        chunk_id=req.chunk_id,
        bbox=req.bbox,
        img_width=req.img_width,
        img_height=req.img_height,
        signal_type=sig_type,
        protocol=proto
    )
    return {"status": "success", "bdw": bdw}

@app.post("/api/session/snr")
def apply_session_awgn(req: AWGNRequest):
    """
    Dynamically injects AWGN noise into the active dataset to achieve target SNR.
    """
    try:
        summary = rf_proc.apply_awgn(req.snr_db)
        return {
            "status": "success",
            "message": f"Applied AWGN noise (Target SNR: {req.snr_db:.1f} dB)",
            "summary": summary,
            "chunks": rf_proc.chunks_meta
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/export/manifest")
def get_export_manifest(
    format_type: str = "yolo",
    drone_name: Optional[str] = None,
    include_images: bool = True,
    include_labels: bool = True,
    include_csv: bool = True,
    include_iq: bool = False,
    include_video: bool = False,
    include_metadata: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
    img_format: str = "jpg",
    export_snr_db: Optional[float] = None
):
    target_drone = drone_name or rf_proc.drone_name or "DJI-MAVIC-PRO-3"
    w = width or rf_proc.render_width
    h = height or rf_proc.render_height
    clean_ext = img_format.lower().lstrip('.')
    if clean_ext not in ["jpg", "jpeg", "png"]:
        clean_ext = "jpg" if format_type == "yolo" else "png"

    clean_base = rf_proc.get_formatted_filename(drone_name=target_drone, extension="")
    num_chunks = len(rf_proc.chunks_meta)
    files_list = []

    if format_type == "yolo":
        zip_filename = f"yolo_dataset_{clean_base}.zip"
        if include_labels:
            files_list.append("data.yaml")
        if include_metadata:
            files_list.append("metadata.json")
        if include_csv:
            files_list.append("signal_parameters_bdw.csv")
        if include_images:
            for c in rf_proc.chunks_meta:
                files_list.append(f"images/{rf_proc.get_formatted_filename(chunk_id=c['id'], extension=clean_ext, drone_name=target_drone)}")
        if include_labels:
            for c in rf_proc.chunks_meta:
                files_list.append(f"labels/{rf_proc.get_formatted_filename(chunk_id=c['id'], extension='txt', drone_name=target_drone)}")
        if include_iq:
            for c in rf_proc.chunks_meta:
                files_list.append(f"iq/{rf_proc.get_formatted_filename(chunk_id=c['id'], extension='iq', drone_name=target_drone)}")
        if include_video:
            files_list.append(f"video/waterfall_{target_drone}.mp4")
    else:
        zip_filename = f"coco_bdw_dataset_{clean_base}.zip"
        if include_labels:
            files_list.append("annotations_coco_bdw.json")
        if include_metadata:
            files_list.append("metadata.json")
        if include_csv:
            files_list.append("signal_parameters_bdw.csv")
        if include_images:
            for c in rf_proc.chunks_meta:
                files_list.append(f"spectrograms/{rf_proc.get_formatted_filename(chunk_id=c['id'], extension=clean_ext, drone_name=target_drone)}")
        if include_iq:
            for c in rf_proc.chunks_meta:
                files_list.append(f"iq/{rf_proc.get_formatted_filename(chunk_id=c['id'], extension='iq', drone_name=target_drone)}")
        if include_video:
            files_list.append(f"video/waterfall_{target_drone}.mp4")

    est_img_size_kb = 120 if clean_ext in ["jpg", "jpeg"] else 350
    est_iq_size_kb = int((rf_proc.chunks_meta[0]["num_samples"] * 8) / 1024) if num_chunks > 0 else 0
    total_est_kb = 0
    if include_images:
        total_est_kb += num_chunks * est_img_size_kb
    if include_labels:
        total_est_kb += num_chunks * 2
    if include_iq:
        total_est_kb += num_chunks * est_iq_size_kb
    if include_video:
        total_est_kb += 5000
    if include_csv:
        total_est_kb += 20
    if include_metadata:
        total_est_kb += 5

    return {
        "format_type": format_type,
        "zip_filename": zip_filename,
        "drone_name": target_drone,
        "resolution": f"{w}x{h}",
        "img_format": clean_ext,
        "export_snr_db": export_snr_db,
        "num_chunks": num_chunks,
        "total_files": len(files_list),
        "estimated_size_mb": round(total_est_kb / 1024, 2),
        "files_sample": files_list[:20],
        "total_files_count": len(files_list),
        "summary": {
            "images": num_chunks if include_images else 0,
            "labels": num_chunks if include_labels else 0,
            "iq_chunks": num_chunks if include_iq else 0,
            "video": 1 if include_video else 0,
            "csv": 1 if include_csv else 0,
            "metadata": 1 if include_metadata else 0
        }
    }

@app.get("/api/export/coco")
def export_coco(
    drone_name: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    img_format: str = "png",
    export_snr_db: Optional[float] = None
):
    target_drone = drone_name or rf_proc.drone_name or "DJI-MAVIC-PRO-3"
    w = width or rf_proc.render_width
    h = height or rf_proc.render_height
    coco_data = COCOBDWExporter.build_coco_json(
        rf_processor=rf_proc,
        classes=classes_state,
        annotations_by_chunk=annotations_state,
        drone_name=target_drone,
        img_width=w,
        img_height=h,
        img_format=img_format,
        export_snr_db=export_snr_db
    )
    json_filename = f"annotations_coco_bdw_{rf_proc.get_formatted_filename(extension='json', drone_name=target_drone)}"
    return JSONResponse(
        content=coco_data,
        headers={"Content-Disposition": f"attachment; filename={json_filename}"}
    )

@app.get("/api/export/zip")
def export_zip(
    drone_name: Optional[str] = None,
    include_images: bool = True,
    include_labels: bool = True,
    include_coco_json: bool = True,
    include_csv: bool = True,
    include_iq: bool = False,
    include_video: bool = False,
    include_metadata: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
    img_format: str = "png",
    export_snr_db: Optional[float] = None
):
    """
    Export full dataset ZIP bundle in COCO format with selectable components.
    """
    target_drone = drone_name or rf_proc.drone_name or "DJI-MAVIC-PRO-3"
    w = width or rf_proc.render_width
    h = height or rf_proc.render_height
    zip_bytes = COCOBDWExporter.generate_zip_bundle(
        rf_processor=rf_proc,
        classes=classes_state,
        annotations_by_chunk=annotations_state,
        drone_name=target_drone,
        img_width=w,
        img_height=h,
        img_format=img_format,
        include_images=include_images,
        include_coco_json=(include_coco_json and include_labels),
        include_csv=include_csv,
        include_iq=include_iq,
        include_video=include_video,
        include_metadata=include_metadata,
        export_snr_db=export_snr_db
    )
    suffix = "_with_iq" if include_iq else ""
    if export_snr_db is not None:
        suffix += f"_{export_snr_db:.0f}db"
    clean_base = rf_proc.get_formatted_filename(drone_name=target_drone, extension="")
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=rf_spectrogram_bundle_{clean_base}{suffix}.zip"}
    )

@app.get("/api/export/yolo")
def export_yolo(
    drone_name: Optional[str] = None,
    include_images: bool = True,
    include_labels: bool = True,
    include_csv: bool = True,
    include_iq: bool = False,
    include_video: bool = False,
    include_metadata: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
    img_format: str = "jpg",
    export_snr_db: Optional[float] = None
):
    """
    Export dataset in standard YOLO training format with selectable components.
    """
    target_drone = drone_name or rf_proc.drone_name or "DJI-MAVIC-PRO-3"
    w = width or rf_proc.render_width
    h = height or rf_proc.render_height
    zip_bytes = YOLOExporter.generate_yolo_zip_bundle(
        rf_processor=rf_proc,
        classes=classes_state,
        annotations_by_chunk=annotations_state,
        drone_name=target_drone,
        img_width=w,
        img_height=h,
        img_format=img_format,
        include_images=include_images,
        include_labels=include_labels,
        include_csv=include_csv,
        include_iq=include_iq,
        include_video=include_video,
        include_metadata=include_metadata,
        export_snr_db=export_snr_db
    )
    suffix = "_with_iq" if include_iq else ""
    if export_snr_db is not None:
        suffix += f"_{export_snr_db:.0f}db"
    clean_base = rf_proc.get_formatted_filename(drone_name=target_drone, extension="")
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=yolo_dataset_{clean_base}{suffix}.zip"}
    )

# ---------------------------------------------------------
# Real-Time Export Progress & Job Manager
# ---------------------------------------------------------
class ExportJobRequest(BaseModel):
    format_type: str = "yolo"
    drone_name: Optional[str] = None
    include_images: bool = True
    include_labels: bool = True
    include_csv: bool = True
    include_iq: bool = False
    include_video: bool = False
    include_metadata: bool = True
    width: Optional[int] = None
    height: Optional[int] = None
    img_format: str = "jpg"
    export_snr_db: Optional[float] = None

export_jobs_lock = threading.Lock()
export_jobs: Dict[str, Dict[str, Any]] = {}

def _run_export_task(job_id: str, req: ExportJobRequest):
    target_drone = req.drone_name or rf_proc.drone_name or "DJI-MAVIC-PRO-3"
    w = req.width or rf_proc.render_width
    h = req.height or rf_proc.render_height
    clean_ext = req.img_format.lower().lstrip('.')
    if clean_ext not in ["jpg", "jpeg", "png"]:
        clean_ext = "jpg" if req.format_type == "yolo" else "png"

    clean_base = rf_proc.get_formatted_filename(drone_name=target_drone, extension="")
    suffix = "_with_iq" if req.include_iq else ""
    if req.export_snr_db is not None:
        suffix += f"_{req.export_snr_db:.0f}db"

    if req.format_type == "yolo":
        zip_filename = f"yolo_dataset_{clean_base}{suffix}.zip"
    else:
        zip_filename = f"rf_spectrogram_bundle_{clean_base}{suffix}.zip"

    def progress_cb(pct: int, stage: str, detail: str, stats: str):
        with export_jobs_lock:
            if job_id in export_jobs:
                export_jobs[job_id].update({
                    "progress": min(99, max(0, pct)),
                    "stage": stage,
                    "detail": detail,
                    "stats": stats
                })

    try:
        if req.format_type == "yolo":
            zip_bytes = YOLOExporter.generate_yolo_zip_bundle(
                rf_processor=rf_proc,
                classes=classes_state,
                annotations_by_chunk=annotations_state,
                drone_name=target_drone,
                img_width=w,
                img_height=h,
                img_format=clean_ext,
                include_images=req.include_images,
                include_labels=req.include_labels,
                include_csv=req.include_csv,
                include_iq=req.include_iq,
                include_video=req.include_video,
                include_metadata=req.include_metadata,
                export_snr_db=req.export_snr_db,
                progress_callback=progress_cb
            )
        else:
            zip_bytes = COCOBDWExporter.generate_zip_bundle(
                rf_processor=rf_proc,
                classes=classes_state,
                annotations_by_chunk=annotations_state,
                drone_name=target_drone,
                img_width=w,
                img_height=h,
                img_format=clean_ext,
                include_images=req.include_images,
                include_coco_json=req.include_labels,
                include_csv=req.include_csv,
                include_iq=req.include_iq,
                include_video=req.include_video,
                include_metadata=req.include_metadata,
                export_snr_db=req.export_snr_db,
                progress_callback=progress_cb
            )

        with export_jobs_lock:
            export_jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "stage": "completed",
                "detail": "Export Complete! Downloading package...",
                "stats": f"{len(zip_bytes)/(1024*1024):.2f} MB generated",
                "zip_bytes": zip_bytes,
                "zip_filename": zip_filename,
                "completed_at": time.time()
            })
    except Exception as e:
        logger.error(f"Export job {job_id} failed: {e}", exc_info=True)
        with export_jobs_lock:
            export_jobs[job_id].update({
                "status": "error",
                "progress": 0,
                "stage": "error",
                "detail": f"Export failed: {str(e)}",
                "error": str(e)
            })

@app.post("/api/export/start")
def start_export_job(req: ExportJobRequest):
    # Cleanup jobs older than 15 minutes
    now = time.time()
    with export_jobs_lock:
        to_delete = [jid for jid, info in export_jobs.items() if now - info.get("created_at", now) > 900]
        for jid in to_delete:
            del export_jobs[jid]

    job_id = f"exp_{int(now*1000)}_{uuid.uuid4().hex[:6]}"
    with export_jobs_lock:
        export_jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "progress": 0,
            "stage": "initializing",
            "detail": "Preparing export package...",
            "stats": f"Format: {req.format_type.upper()}",
            "created_at": now
        }
    t = threading.Thread(target=_run_export_task, args=(job_id, req), daemon=True)
    t.start()
    return {"status": "success", "job_id": job_id}

@app.get("/api/export/status/{job_id}")
def get_export_job_status(job_id: str):
    with export_jobs_lock:
        job = export_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Export job not found")
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job.get("progress", 0),
            "stage": job.get("stage", "running"),
            "detail": job.get("detail", ""),
            "stats": job.get("stats", ""),
            "zip_filename": job.get("zip_filename", "dataset.zip"),
            "error": job.get("error")
        }

@app.get("/api/export/download/{job_id}")
def download_export_job(job_id: str):
    with export_jobs_lock:
        job = export_jobs.get(job_id)
        if not job or job.get("status") != "completed" or not job.get("zip_bytes"):
            raise HTTPException(status_code=404, detail="Export job output not available or expired")
        zip_bytes = job["zip_bytes"]
        zip_filename = job.get("zip_filename", "dataset.zip")

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )

@app.get("/api/export/waterfall_video")
def export_waterfall_video(fps: float = 10.0, width: int = 1024, height: int = 512, download: bool = True):
    try:
        clean_name = os.path.splitext(rf_proc.source_filename)[0] or "capture"
        video_filename = f"waterfall_{clean_name}.mp4"
        video_path = os.path.join(UPLOAD_DIR, video_filename)
        res = rf_proc.render_waterfall_video(output_filepath=video_path, fps=fps, width=width, height=height)

        if download:
            return FileResponse(
                path=video_path,
                media_type="video/mp4",
                filename=video_filename
            )
        return {
            "status": "success",
            "metadata": res,
            "stream_url": f"/api/export/waterfall_stream?t={int(os.path.getmtime(video_path))}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/waterfall_stream")
def stream_waterfall_video():
    clean_name = os.path.splitext(rf_proc.source_filename)[0] or "capture"
    video_filename = f"waterfall_{clean_name}.mp4"
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    if not os.path.exists(video_path):
        rf_proc.render_waterfall_video(output_filepath=video_path, fps=10.0, width=1024, height=512)
    return FileResponse(path=video_path, media_type="video/mp4")

@app.post("/api/models/upload")
async def upload_onnx_model(file: UploadFile = File(...)):
    """
    Upload an ONNX model file (.onnx) and initialize inference engine.
    """
    if not file.filename.lower().endswith(".onnx"):
        raise HTTPException(status_code=400, detail="Only .onnx model files are supported.")

    model_path = os.path.join(MODELS_DIR, file.filename)
    with open(model_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        model_info = onnx_detector.load_model(model_path)
        return {
            "status": "success",
            "message": f"Successfully loaded ONNX model '{file.filename}'",
            "model_info": {
                "model_name": model_info.model_name,
                "input_shape": model_info.input_shape,
                "input_width": model_info.input_width,
                "input_height": model_info.input_height,
                "num_classes": model_info.num_classes,
                "class_names": model_info.class_names
            }
        }
    except Exception as e:
        if os.path.exists(model_path):
            os.remove(model_path)
        raise HTTPException(status_code=400, detail=f"Failed to load ONNX model: {str(e)}")

@app.get("/api/models/info")
def get_model_info():
    """
    Return metadata of currently active ONNX model.
    """
    if onnx_detector.model_info is None:
        return {"status": "none", "loaded": False, "model_info": None}
    info = onnx_detector.model_info
    return {
        "status": "success",
        "loaded": True,
        "model_info": {
            "model_name": info.model_name,
            "input_shape": info.input_shape,
            "input_width": info.input_width,
            "input_height": info.input_height,
            "num_classes": info.num_classes,
            "class_names": info.class_names
        }
    }

@app.post("/api/autolabel/cfar")
def autolabel_cfar(req: CFARAutoLabelRequest):
    """
    Run 2D CA-CFAR RF signal detection on spectrogram chunks to generate candidate proposals.
    Supports multi-threaded parallel execution across all CPU cores and NDJSON streaming.
    """
    try:
        cfg = CFARConfig(
            threshold_factor=req.threshold_factor,
            guard_rows=req.guard_rows,
            guard_cols=req.guard_cols,
            train_rows=req.train_rows,
            train_cols=req.train_cols,
            morph_kernel=req.morph_kernel,
            min_area=req.min_area,
            max_boxes=req.max_boxes,
            target_category_id=req.target_category_id
        )
        detector = CFARDetector(cfg)

        target_chunks = []
        if req.scope == "current":
            target_chunks = [req.chunk_id if req.chunk_id is not None else 0]
        else:
            target_chunks = list(range(len(rf_proc.chunks_meta)))

        target_cat = next((c for c in classes_state if c["id"] == req.target_category_id), None)
        sig_type = target_cat["type_of_signal"] if target_cat else "Unknown"
        protocol = target_cat["protocol"] if target_cat else "Generic"

        logger.info(f"Starting CA-CFAR auto-labeling on {len(target_chunks)} chunks (scope={req.scope}, threshold_factor={req.threshold_factor}, target_cat={req.target_category_id}, res={rf_proc.render_width}x{rf_proc.render_height})...")

        img_width = rf_proc.render_width
        img_height = rf_proc.render_height

        def process_single_chunk(c_id: int) -> Tuple[int, List[Dict[str, Any]]]:
            if c_id < 0 or c_id >= len(rf_proc.chunks_meta):
                return c_id, []
            img_bgr, _ = rf_proc.render_spectrogram_image_array(c_id, width=img_width, height=img_height)
            chunk_props = detector.detect_boxes(img_bgr)
            for prop in chunk_props:
                prop["img_width"] = img_width
                prop["img_height"] = img_height
                bdw = rf_proc.calculate_bdw_parameters(
                    chunk_id=c_id,
                    bbox=[prop["x"], prop["y"], prop["width"], prop["height"]],
                    img_width=img_width,
                    img_height=img_height,
                    signal_type=sig_type,
                    protocol=protocol
                )
                prop["bdw"] = bdw
            return c_id, chunk_props

        if req.stream:
            def cfar_stream_generator():
                total_chunks = len(target_chunks)
                proposals_by_chunk: Dict[str, List[Dict[str, Any]]] = {}
                total_proposals = 0
                completed_count = 0

                max_workers = min(8, max(1, os.cpu_count() or 4))
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    future_to_chunk = {pool.submit(process_single_chunk, c_id): c_id for c_id in target_chunks}
                    for future in as_completed(future_to_chunk):
                        c_id, chunk_props = future.result()
                        completed_count += 1
                        proposals_by_chunk[str(c_id)] = chunk_props
                        total_proposals += len(chunk_props)

                        progress_evt = {
                            "type": "progress",
                            "current": completed_count,
                            "total": total_chunks,
                            "chunk_id": c_id,
                            "chunk_proposals": len(chunk_props),
                            "total_proposals": total_proposals
                        }
                        yield json.dumps(progress_evt) + "\n"

                complete_evt = {
                    "type": "complete",
                    "status": "success",
                    "detector": "cfar",
                    "scope": req.scope,
                    "total_proposals": total_proposals,
                    "proposals_by_chunk": proposals_by_chunk
                }
                yield json.dumps(complete_evt) + "\n"

            return StreamingResponse(cfar_stream_generator(), media_type="application/x-ndjson")

        # Synchronous parallel execution for non-streaming requests
        proposals_by_chunk: Dict[str, List[Dict[str, Any]]] = {}
        total_proposals = 0
        max_workers = min(8, max(1, os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(process_single_chunk, target_chunks))
            for c_id, chunk_props in results:
                proposals_by_chunk[str(c_id)] = chunk_props
                total_proposals += len(chunk_props)

        logger.info(f"Completed CA-CFAR auto-labeling: {total_proposals} proposals detected across {len(target_chunks)} chunks.")
        return {
            "status": "success",
            "detector": "cfar",
            "scope": req.scope,
            "total_proposals": total_proposals,
            "proposals_by_chunk": proposals_by_chunk
        }
    except Exception as e:
        logger.error(f"CFAR Auto-Label error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"CFAR Auto-Label error: {str(e)}")

@app.post("/api/autolabel/onnx")
def autolabel_onnx(req: ONNXAutoLabelRequest):
    """
    Run ONNX AI Model inference on spectrogram chunks to generate candidate proposals.
    Supports multi-threaded parallel execution and NDJSON streaming.
    """
    if onnx_detector.session is None:
        raise HTTPException(status_code=400, detail="No ONNX model loaded. Please upload a .onnx model first.")

    try:
        target_chunks = []
        if req.scope == "current":
            target_chunks = [req.chunk_id if req.chunk_id is not None else 0]
        else:
            target_chunks = list(range(len(rf_proc.chunks_meta)))

        img_width = rf_proc.render_width
        img_height = rf_proc.render_height

        def process_single_onnx_chunk(c_id: int) -> Tuple[int, List[Dict[str, Any]]]:
            if c_id < 0 or c_id >= len(rf_proc.chunks_meta):
                return c_id, []
            img_bgr, _ = rf_proc.render_spectrogram_image_array(c_id, width=img_width, height=img_height)

            chunk_props = onnx_detector.detect(
                image=img_bgr,
                conf_thresh=req.conf_thresh,
                iou_thresh=req.iou_thresh,
                default_category_id=req.default_category_id,
                class_mapping=req.class_mapping
            )

            for prop in chunk_props:
                prop["img_width"] = img_width
                prop["img_height"] = img_height
                cat = next((c for c in classes_state if c["id"] == prop["category_id"]), None)
                sig_type = cat["type_of_signal"] if cat else "Unknown"
                protocol = cat["protocol"] if cat else "Generic"

                bdw = rf_proc.calculate_bdw_parameters(
                    chunk_id=c_id,
                    bbox=[prop["x"], prop["y"], prop["width"], prop["height"]],
                    img_width=img_width,
                    img_height=img_height,
                    signal_type=sig_type,
                    protocol=protocol
                )
                prop["bdw"] = bdw
            return c_id, chunk_props

        if req.stream:
            def onnx_stream_generator():
                total_chunks = len(target_chunks)
                proposals_by_chunk: Dict[str, List[Dict[str, Any]]] = {}
                total_proposals = 0
                completed_count = 0

                max_workers = min(8, max(1, os.cpu_count() or 4))
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    future_to_chunk = {pool.submit(process_single_onnx_chunk, c_id): c_id for c_id in target_chunks}
                    for future in as_completed(future_to_chunk):
                        c_id, chunk_props = future.result()
                        completed_count += 1
                        proposals_by_chunk[str(c_id)] = chunk_props
                        total_proposals += len(chunk_props)

                        progress_evt = {
                            "type": "progress",
                            "current": completed_count,
                            "total": total_chunks,
                            "chunk_id": c_id,
                            "chunk_proposals": len(chunk_props),
                            "total_proposals": total_proposals
                        }
                        yield json.dumps(progress_evt) + "\n"

                complete_evt = {
                    "type": "complete",
                    "status": "success",
                    "detector": "onnx",
                    "model_name": onnx_detector.model_info.model_name if onnx_detector.model_info else "model.onnx",
                    "scope": req.scope,
                    "total_proposals": total_proposals,
                    "proposals_by_chunk": proposals_by_chunk
                }
                yield json.dumps(complete_evt) + "\n"

            return StreamingResponse(onnx_stream_generator(), media_type="application/x-ndjson")

        proposals_by_chunk: Dict[str, List[Dict[str, Any]]] = {}
        total_proposals = 0
        max_workers = min(8, max(1, os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(process_single_onnx_chunk, target_chunks))
            for c_id, chunk_props in results:
                proposals_by_chunk[str(c_id)] = chunk_props
                total_proposals += len(chunk_props)

        logger.info(f"Completed ONNX AI inference: {total_proposals} proposals detected across {len(target_chunks)} chunks.")
        return {
            "status": "success",
            "detector": "onnx",
            "model_name": onnx_detector.model_info.model_name if onnx_detector.model_info else "model.onnx",
            "scope": req.scope,
            "total_proposals": total_proposals,
            "proposals_by_chunk": proposals_by_chunk
        }
    except Exception as e:
        logger.error(f"ONNX Auto-Label error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ONNX Auto-Label error: {str(e)}")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# Static files for frontend
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
