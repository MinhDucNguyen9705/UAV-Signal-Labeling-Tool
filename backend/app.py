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
from backend.cfar_detector import CFARDetector, CFARConfig
from backend.onnx_detector import ONNXDetector, ONNXModelInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

app = FastAPI(title="RF Spectrogram Bounding Box Tool with BDW Support")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/js/") or request.url.path.startswith("/css/") or request.url.path == "/" or request.url.path.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

UPLOAD_DIR = "/home/dev/labelling_tool/samples"
MODELS_DIR = os.path.join(UPLOAD_DIR, "models")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Global State
rf_proc = RFProcessor(fs=61.44e6, center_freq=2400.0e6, chunk_duration_ms=1.0, colormap="turbo", db_min=-90.0, db_max=0.0)
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
    chunk_duration_ms: Optional[float] = None
    overlap_duration_ms: Optional[float] = None
    nfft: Optional[int] = None
    hop_length: Optional[int] = None
    window: Optional[str] = None
    colormap: Optional[str] = None
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
except Exception as e:
    print(f"Error initializing demo sample: {e}")

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
    rf_proc.set_config(
        fs=cfg.fs_hz,
        center_freq=cfg.center_freq_hz,
        chunk_duration_ms=cfg.chunk_duration_ms,
        overlap_duration_ms=cfg.overlap_duration_ms,
        nfft=cfg.nfft,
        hop_length=cfg.hop_length,
        window=cfg.window,
        colormap=cfg.colormap,
        db_min=cfg.db_min,
        db_max=cfg.db_max,
        eager_threshold=cfg.eager_threshold,
        batch_size=cfg.batch_size
    )
    return {
        "status": "success",
        "summary": rf_proc.get_summary(),
        "chunks": rf_proc.chunks_meta
    }

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    fs: float = Form(61.44e6),
    center_freq: float = Form(2400.0e6),
    iq_format: str = Form("float32"),
    chunk_duration_ms: float = Form(30.0),
    overlap_duration_ms: float = Form(10.0),
    nfft: int = Form(1024),
    colormap: str = Form("turbo"),
    window: str = Form("hann")
):
    global annotations_state
    filename = file.filename or "uploaded_signal.iq"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if 0 < center_freq < 100000.0:
        center_freq = center_freq * 1e6

    rf_proc.set_config(
        fs=fs,
        center_freq=center_freq,
        chunk_duration_ms=chunk_duration_ms,
        overlap_duration_ms=overlap_duration_ms,
        nfft=nfft,
        colormap=colormap,
        window=window
    )

    if filename.endswith(".h5") or filename.endswith(".hdf5"):
        summary = rf_proc.load_h5_file(file_path, fs=fs, center_freq=center_freq)
    else:
        summary = rf_proc.load_iq_file(file_path, iq_format=iq_format, fs=fs, center_freq=center_freq)

    # Reset annotations for new file
    annotations_state = {}

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
    iq_format: str = Form("float32"),
    duration_ms: float = Form(100.0),
    output_format: str = Form("h5"),
    chunk_duration_ms: float = Form(30.0),
    overlap_duration_ms: float = Form(10.0)
):
    global annotations_state
    if 0 < center_freq < 100000.0:
        center_freq = center_freq * 1e6

    fs_str = f"{fs / 1e6:.2f}".replace('.', '_')
    filename = f"synthetic_rf_{fs_str}mhz_{iq_format}.{output_format}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    output_path, gt_boxes = RFSampleGenerator.generate_synthetic_rf(
        fs=fs,
        center_freq=center_freq,
        duration_ms=duration_ms,
        iq_format=iq_format,
        output_format=output_format,
        output_path=file_path
    )

    rf_proc.set_config(
        fs=fs,
        center_freq=center_freq,
        chunk_duration_ms=chunk_duration_ms,
        overlap_duration_ms=overlap_duration_ms
    )
    if output_format == "h5":
        summary = rf_proc.load_h5_file(output_path, fs=fs, center_freq=center_freq)
    else:
        summary = rf_proc.load_iq_file(output_path, iq_format=iq_format, fs=fs, center_freq=center_freq)

    annotations_state = {}

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
def get_spectrogram_image(chunk_id: int, width: Optional[int] = 1024, height: Optional[int] = 512):
    try:
        img_bytes, meta = rf_proc.render_spectrogram_image(chunk_id=chunk_id, width=width, height=height)
        return Response(content=img_bytes, media_type="image/png")
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
    return {"status": "success", "chunk_id": req.chunk_id, "count": len(req.annotations)}

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

@app.get("/api/export/coco")
def export_coco(width: int = 1024, height: int = 512):
    coco_data = COCOBDWExporter.build_coco_json(
        rf_processor=rf_proc,
        classes=classes_state,
        annotations_by_chunk=annotations_state,
        img_width=width,
        img_height=height
    )
    return JSONResponse(
        content=coco_data,
        headers={"Content-Disposition": f"attachment; filename=annotations_coco_bdw_{rf_proc.source_filename}.json"}
    )

@app.post("/api/import/coco")
async def import_coco_endpoint(request: Request):
    global annotations_state, classes_state
    content_type = request.headers.get("content-type", "")
    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file")
            if file and hasattr(file, "read"):
                content = await file.read()
                data = json.loads(content.decode("utf-8"))
            elif "coco_json" in form:
                data = json.loads(form["coco_json"])
            else:
                raise ValueError("No file or JSON data provided in form.")
        else:
            data = await request.json()

        ann_by_chunk, updated_classes, stats = COCOBDWImporter.parse_coco_json(
            coco_data=data,
            rf_processor=rf_proc,
            existing_classes=classes_state,
            img_width=1024,
            img_height=512
        )

        classes_state = updated_classes
        annotations_state = ann_by_chunk

        return {
            "status": "success",
            "message": f"Successfully imported {stats['total_imported']} bounding boxes across {stats['chunks_updated']} chunks",
            "stats": stats,
            "classes": classes_state,
            "annotations": annotations_state
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to import COCO JSON: {str(e)}")

@app.get("/api/export/zip")
def export_zip(include_iq: bool = False, width: int = 1024, height: int = 512):
    """
    Export full dataset ZIP bundle.
    If include_iq=True, includes chunked raw .iq binary files alongside spectrogram PNG images and annotations.
    """
    clean_name = os.path.splitext(rf_proc.source_filename)[0] or "rf_dataset"
    zip_bytes = COCOBDWExporter.generate_zip_bundle(
        rf_processor=rf_proc,
        classes=classes_state,
        annotations_by_chunk=annotations_state,
        img_width=width,
        img_height=height,
        include_iq=include_iq
    )
    suffix = "_with_iq" if include_iq else ""
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=rf_spectrogram_bundle_{clean_name}{suffix}.zip"}
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

        img_width = 1024
        img_height = 512

        def process_single_chunk(c_id: int) -> Tuple[int, List[Dict[str, Any]]]:
            if c_id < 0 or c_id >= len(rf_proc.chunks_meta):
                return c_id, []
            img_bytes, _ = rf_proc.render_spectrogram_image(c_id, width=img_width, height=img_height)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            chunk_props = detector.detect_boxes(img)
            for prop in chunk_props:
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

        return {
            "status": "success",
            "detector": "cfar",
            "scope": req.scope,
            "total_proposals": total_proposals,
            "proposals_by_chunk": proposals_by_chunk
        }
    except Exception as e:
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

        img_width = 1024
        img_height = 512

        def process_single_onnx_chunk(c_id: int) -> Tuple[int, List[Dict[str, Any]]]:
            if c_id < 0 or c_id >= len(rf_proc.chunks_meta):
                return c_id, []
            img_bytes, _ = rf_proc.render_spectrogram_image(c_id, width=img_width, height=img_height)
            img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            chunk_props = onnx_detector.detect(
                image=img_bgr,
                conf_thresh=req.conf_thresh,
                iou_thresh=req.iou_thresh,
                default_category_id=req.default_category_id,
                class_mapping=req.class_mapping
            )

            for prop in chunk_props:
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

        return {
            "status": "success",
            "detector": "onnx",
            "model_name": onnx_detector.model_info.model_name if onnx_detector.model_info else "model.onnx",
            "scope": req.scope,
            "total_proposals": total_proposals,
            "proposals_by_chunk": proposals_by_chunk
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX Auto-Label error: {str(e)}")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# Static files for frontend
FRONTEND_DIR = "/home/dev/labelling_tool/frontend"
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
