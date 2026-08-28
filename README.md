# 🛰️ UAV & Drone RF Signal Spectrogram Labeling Tool

An interactive, high-performance web-based annotation platform for **RF time-domain IQ signals**, **STFT spectrograms**, and **drone communication / radar signal labeling** with automatic **BDW (Basic Data Word)** physical parameter extraction, automated CFAR / ONNX detection, AWGN noise injection, and unified **YOLO / COCO** dataset exporters.

---

## 🌟 Key Features

### 1. 📡 Advanced RF Signal Ingestion & Processing
* **Multi-Format Support**: Ingests raw `.iq` binary files (interleaved `int16`, `float32` complex) and `.h5` / `.hdf5` multi-session recordings.
* **Dual-Mode Rendering Engine**:
  * **Eager Mode ($\le 20$ chunks)**: Pre-renders and caches all chunks in RAM for zero-latency interactive scrubbing.
  * **Batched Sliding-Window Mode ($> 20$ chunks)**: Dynamically pre-computes sliding batches (10 chunks at a time) with automatic cache eviction to process multi-gigabyte files with strictly bounded RAM usage.
* **Synthetic RF Sample Generator**: Generate synthetic Drone Control, Video Link, and Radar signals on demand with customizable SNR and duration.

### 2. 🎯 Interactive Spectrogram Annotation Canvas
* **HTML5 Canvas Workspace**: Drag, resize, lock, hide, and label signal bounding boxes with sub-pixel precision.
* **Filmstrip Timeline Navigation**: Visual thumbnail strip showing chunk positions, time offsets, and candidate annotation badges.
* **Class & Category Manager**: Define custom signal classes, assign colors, protocols, and signal types (`FHSS`, `DSSS`, `OFDM`, `Chirp/FMCW`, etc.).
* **Adaptive Resolution Scaling**: Automatically rescales existing bounding boxes when changing canvas resolutions ($640\times 640$, $1024\times 512$, $1280\times 720$, $1920\times 1080$, or custom).

### 3. 📊 Automated BDW (Basic Data Word) Physical Parameter Extraction
Converts every bounding box into real-world physical RF metrics in $O(1)$ instantaneous time:
* **TOA (Time of Arrival)** & **TOD (Time of Departure)** in $\mu\text{s}$
* **PW (Pulse Width)** in $\mu\text{s}$
* **FC (Center Frequency)** & **BW (Bandwidth)** in $\text{MHz}$
* **Freq Low / High Bounds** in $\text{MHz}$
* **Signal-to-Noise Ratio (SNR)** in $\text{dB}$

### 4. ⚡ Automated Signal Detection
* **2D 2-Pass CA-CFAR Detector**: Automatic threshold detection with customizable guard/reference cells, PFA rate, and bounding box grouping.
* **ONNX AI Model Detector**: Run custom YOLOv8 / YOLOv11 object detection models directly in the browser via `onnxruntime`.

### 5. 🎛️ AWGN Noise Modification (Target SNR)
* Injects complex Additive White Gaussian Noise (AWGN) during file upload or dataset export to simulate realistic noise environments across specific target SNR levels (e.g. $5\text{ dB}$, $10\text{ dB}$, $15\text{ dB}$).

### 6. 📦 Unified Dataset Exporter & Package Configurator
Export complete training packages with selective components:
* **Formats Supported**:
  * **YOLO Format**: `images/` (JPG/PNG), `labels/*.txt` (normalized $[0, 1]$ coordinates), `data.yaml`, `signal_parameters_bdw.csv`, `metadata.json`.
  * **COCO + BDW Format**: `spectrograms/` (PNG/JPG), `annotations_coco_bdw.json`, `signal_parameters_bdw.csv`, `metadata.json`.
* **Optional Package Components**:
  * 🔲 **Raw IQ Binary Chunks** (`iq/*.iq` raw time-domain complex64 slices with optional AWGN).
  * 🔲 **Continuous Waterfall MP4 Video** (`video/waterfall_<drone_name>.mp4`).
* **Standardized Filename Format**:
  $$\text{<drone\_name>\_<fs>MHz\_<fc>MHz\_<dtype>\_0000.ext}$$
  *Example:* `DJI-MAVIC-PRO-3_100MHz_2450MHz_float_0000.jpg`
* **Live Manifest File Tree Preview**: Real-time folder hierarchy and estimated archive size calculation before downloading.
* **High-Speed Parallel Export**: Multi-threaded spectrogram generation ($8\times$ concurrency) and uncompressed `ZIP_STORED` raw IQ serialization.

---

## 🚀 Quickstart Guide

### Method 1: Local Python Environment

#### Prerequisites
* Python 3.10+
* `ffmpeg` (required for waterfall MP4 video rendering)

```bash
# 1. Clone repository
git clone https://github.com/MinhDucNguyen9705/UAV-Signal-Labeling-Tool.git
cd UAV-Signal-Labeling-Tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application server
bash run.sh
```
Open your browser and navigate to: **`http://localhost:8000`**

---

### Method 2: Docker & Docker Compose (Recommended)

#### Using Docker Compose
```bash
docker compose up --build -d
```

#### Using Standalone Docker CLI
```bash
# Build image
docker build -t rf-labeling-tool .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/samples:/app/samples \
  -v $(pwd)/logs:/app/logs \
  --name rf-labeling-tool \
  rf-labeling-tool
```

---

## 📁 Repository Structure

```
├── backend/
│   ├── app.py                # FastAPI server, REST API & background task managers
│   ├── rf_processor.py       # STFT STFT computation, caching, BDW physics & chunking
│   ├── yolo_exporter.py      # YOLOv5/v8/v11 training dataset bundler & data.yaml generator
│   ├── coco_exporter.py      # COCO + BDW JSON exporter & importer
│   ├── cfar_detector.py      # 2D 2-Pass CA-CFAR automated detector
│   ├── onnx_detector.py      # ONNX Runtime model inference engine
│   ├── sample_generator.py   # Synthetic RF signal & waveform generator
│   └── logger.py             # Rotating structured file and console logger
├── frontend/
│   ├── index.html            # Main single-page application interface
│   ├── css/
│   │   └── style.css         # Modern dark-mode styling & responsive design
│   └── js/
│       ├── app.js            # Main application coordinator & modal manager
│       ├── canvas.js         # Interactive HTML5 annotation canvas
│       ├── navigation.js     # Filmstrip timeline & chunk switcher
│       ├── classes.js        # Signal class & category manager
│       ├── export.js         # Unified export modal & live manifest preview
│       ├── autolabel.js      # CFAR & ONNX automated labeling controller
│       └── hotkeys.js        # Keyboard shortcuts manager
├── samples/                  # Sample recordings, synthetic RF files, and ONNX models
├── Dockerfile                # Production-grade container definition
├── docker-compose.yml        # Docker Compose configuration
├── requirements.txt          # Python dependencies
└── run.sh                    # Automated startup script
```

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `R` | Select Box Annotation Tool |
| `V` / `S` | Select Mode (Move / Resize) |
| `D` | Pan Viewport Tool |
| `Z` | Zoom Tool |
| `A` / `←` | Previous Chunk |
| `D` / `→` | Next Chunk |
| `Space` | Play / Pause Spectrogram Timeline |
| `Delete` / `Backspace` | Delete Selected Bounding Box |
| `L` | Lock / Unlock Selected Box |
| `H` | Hide / Show Selected Box |
| `Ctrl + S` | Save Current Annotations |
| `E` | Open Export Modal |
| `U` | Open Upload Dataset Modal |
| `?` | Show Hotkeys Guide |

---

## 🛠️ API Endpoints Summary

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/session` | `GET` | Retrieve active dataset metadata, chunks, and class definitions |
| `/api/upload` | `POST` | Upload and process raw `.iq` / `.h5` files with STFT and AWGN options |
| `/api/generate_sample` | `POST` | Generate synthetic multi-signal RF capture |
| `/api/session/config` | `POST` | Reconfigure STFT parameters (NFFT, colormap, resolution) |
| `/api/chunks/{id}/spectrogram` | `GET` | Render spectrogram image for a specific chunk |
| `/api/autolabel/cfar` | `POST` | Run 2D CA-CFAR signal detection |
| `/api/autolabel/onnx` | `POST` | Run ONNX object detection model inference |
| `/api/export/manifest` | `GET` | Compute live archive manifest file tree and size |
| `/api/export/start` | `POST` | Start asynchronous export job with live progress tracking |
| `/api/export/status/{job_id}` | `GET` | Poll export job stage, detail, and percentage |
| `/api/export/download/{job_id}` | `GET` | Stream completed dataset ZIP bundle |
| `/api/import/coco` | `POST` | Import annotations from an existing COCO JSON file |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.