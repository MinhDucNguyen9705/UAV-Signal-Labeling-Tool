import io
import os
import math
import concurrent.futures
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import cv2
from typing import Dict, Any, Tuple, List, Optional

import threading

class RFProcessor:
    """
    Processes RF IQ signals from .iq and .h5 files, performs time-chunking,
    computes STFT spectrograms, and calculates RF BDW parameters.
    Supports Dual-Mode Rendering (Eager for small datasets, Batched Sliding-Window with auto-eviction for large datasets).
    """

    SUPPORTED_FS = [245.76e6, 61.44e6, 30.72e6]
    COLORMAPS = ['turbo', 'hot', 'viridis', 'plasma', 'magma', 'inferno', 'jet', 'gray']
    OPENCV_COLORMAPS = {
        'hot': cv2.COLORMAP_HOT,
        'turbo': cv2.COLORMAP_TURBO,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'magma': cv2.COLORMAP_MAGMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'jet': cv2.COLORMAP_JET,
    }
    SUPPORTED_NFFT = [256, 512, 1024, 2048, 4096, 8192, 16384]
    EAGER_CHUNK_THRESHOLD = 20
    DEFAULT_BATCH_SIZE = 10

    def __init__(self,
                 fs: float = 61.44e6,
                 center_freq: float = 2400.0e6,
                 chunk_duration_ms: float = 30.0,
                 overlap_duration_ms: float = 10.0,
                 nfft: int = 1024,
                 hop_length: Optional[int] = None,
                 window: str = 'hann',
                 colormap: str = 'turbo',
                 db_min: float = -90.0,
                 db_max: float = 0.0,
                 eager_threshold: int = 20,
                 batch_size: int = 10):
        self.fs = float(fs)
        self.center_freq = float(center_freq)
        self.chunk_duration_ms = float(chunk_duration_ms)
        self.overlap_duration_ms = float(overlap_duration_ms)
        self.nfft = int(nfft)
        self.hop_length = int(hop_length) if hop_length else int(self.nfft // 4)
        self.window = str(window)
        self.colormap = str(colormap) if colormap in self.COLORMAPS else 'turbo'
        self.db_min = float(db_min)
        self.db_max = float(db_max)
        self.eager_threshold = int(eager_threshold)
        self.batch_size = int(batch_size)
        self.active_batch_range: Tuple[int, int] = (0, 0)
        self._lock = threading.Lock()

        self.iq_data: Optional[np.ndarray] = None  # 1D complex64
        self.source_filename: str = ""
        self.source_format: str = "float32"
        self.total_samples: int = 0
        self.chunks_meta: List[Dict[str, Any]] = []
        self._image_cache: Dict[str, Tuple[bytes, Dict[str, Any]]] = {}
        self._array_cache: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
        self._noise_floor_cache: Dict[int, float] = {}

    @property
    def render_mode(self) -> str:
        """Returns 'eager' if chunk count <= eager_threshold, else 'batched'."""
        if len(self.chunks_meta) <= self.eager_threshold:
            return "eager"
        return "batched"

    def load_iq_file(self, filepath: str, iq_format: str = "float32", fs: Optional[float] = None, center_freq: Optional[float] = None) -> Dict[str, Any]:
        """
        Load raw binary IQ file in int16 or float32 format with low-memory parsing.
        """
        self.iq_data = None
        if fs is not None:
            self.fs = float(fs)
        if center_freq is not None:
            self.center_freq = float(center_freq)

        self.source_filename = os.path.basename(filepath)
        self.source_format = iq_format

        if iq_format == "int16":
            raw = np.fromfile(filepath, dtype='<i2')
            if len(raw) % 2 != 0:
                raw = raw[:len(raw) - 1]
            i_samples = raw[0::2].astype(np.float32) / 32768.0
            q_samples = raw[1::2].astype(np.float32) / 32768.0
            self.iq_data = (i_samples + 1j * q_samples).astype(np.complex64)
            del raw, i_samples, q_samples
        elif iq_format == "float32":
            raw = np.fromfile(filepath, dtype='<f4')
            if len(raw) % 2 != 0:
                raw = raw[:len(raw) - 1]

            # Sample only the first 10,000 values to detect if integer file was uploaded as float
            sample_check = raw[:min(len(raw), 10000)]
            max_val = np.nanmax(np.abs(sample_check)) if len(sample_check) > 0 else 0
            if max_val > 1000.0:
                del raw
                # Fallback to int16
                raw_i16 = np.fromfile(filepath, dtype='<i2')
                if len(raw_i16) % 2 != 0:
                    raw_i16 = raw_i16[:len(raw_i16) - 1]
                i_samples = raw_i16[0::2].astype(np.float32) / 32768.0
                q_samples = raw_i16[1::2].astype(np.float32) / 32768.0
                self.iq_data = (i_samples + 1j * q_samples).astype(np.complex64)
                del raw_i16, i_samples, q_samples
                self.source_format = "int16 (auto-detected)"
            else:
                try:
                    self.iq_data = raw.view(np.complex64).copy()
                except Exception:
                    i_samples = raw[0::2].astype(np.float32)
                    q_samples = raw[1::2].astype(np.float32)
                    self.iq_data = (i_samples + 1j * q_samples).astype(np.complex64)
                    del i_samples, q_samples
                del raw
        else:
            raise ValueError(f"Unsupported IQ format: {iq_format}. Must be 'int16' or 'float32'.")

        self.total_samples = len(self.iq_data)
        self._calculate_chunks()
        self._initialize_cache_for_mode()

        return self.get_summary()

    def load_h5_file(self, filepath: str, dataset_name: Optional[str] = None, fs: Optional[float] = None, center_freq: Optional[float] = None) -> Dict[str, Any]:
        """
        Load HDF5 file containing RF IQ data.
        Supports:
        1. Multi-session format with 40-byte header: h5['session'][session_name]['raw']
        2. Standard datasets: 'iq', 'data', 'samples', 'signal', 'rx_data', etc.
        """
        self.iq_data = None
        self.source_filename = os.path.basename(filepath)
        HEADER_SIZE = 40
        SCALE = 32768.0

        with h5py.File(filepath, 'r') as h5f:
            # Check attributes for metadata
            if 'fs' in h5f.attrs:
                self.fs = float(h5f.attrs['fs'])
            elif 'sample_rate' in h5f.attrs:
                self.fs = float(h5f.attrs['sample_rate'])
            elif fs is not None:
                self.fs = float(fs)

            if center_freq is not None:
                self.center_freq = float(center_freq)
            elif 'center_freq' in h5f.attrs:
                self.center_freq = float(h5f.attrs['center_freq'])
            elif 'fc' in h5f.attrs:
                self.center_freq = float(h5f.attrs['fc'])

            # Case 1: Session group structure (h5['session'][session_name]['raw'])
            if 'session' in h5f and isinstance(h5f['session'], h5py.Group):
                session = h5f['session']
                session_names = sorted(session.keys())
                iq_parts = []

                for session_name in session_names:
                    sess_item = session[session_name]
                    if isinstance(sess_item, h5py.Group) and 'raw' in sess_item:
                        raw = sess_item['raw'][()]
                    elif isinstance(sess_item, h5py.Dataset):
                        raw = sess_item[()]
                    else:
                        continue

                    # Process raw byte/int16 payload
                    if raw.dtype in [np.uint8, np.int8]:
                        payload = raw[HEADER_SIZE:]
                        if len(payload) % 2 != 0:
                            payload = payload[:len(payload) - 1]
                        samples = payload.view('<i2')
                    elif raw.dtype == np.int16:
                        # 40 bytes = 20 int16 samples
                        samples = raw[HEADER_SIZE // 2:]
                    else:
                        raw_bytes = raw.tobytes() if hasattr(raw, 'tobytes') else bytes(raw)
                        payload = raw_bytes[HEADER_SIZE:]
                        samples = np.frombuffer(payload, dtype='<i2')

                    if len(samples) % 2 != 0:
                        samples = samples[:len(samples) - 1]

                    if len(samples) >= 2:
                        i_samples = samples[0::2].astype(np.float32) / SCALE
                        q_samples = samples[1::2].astype(np.float32) / SCALE
                        iq_complex = (i_samples + 1j * q_samples).astype(np.complex64)
                        iq_parts.append(iq_complex)

                if iq_parts:
                    self.iq_data = np.concatenate(iq_parts)
                    self.source_format = "h5_session_int16"

            # Case 2: Standard single or multiple datasets
            if self.iq_data is None:
                if dataset_name and dataset_name in h5f:
                    ds = h5f[dataset_name]
                else:
                    candidate_keys = ['iq', 'data', 'samples', 'signal', 'rx_data', 'raw_iq', 'rf_data', 'raw']
                    found_key = None
                    for k in candidate_keys:
                        if k in h5f and isinstance(h5f[k], h5py.Dataset):
                            found_key = k
                            break
                    if not found_key:
                        # Pick first dataset in h5 file
                        keys = []
                        h5f.visititems(lambda n, obj: keys.append(n) if isinstance(obj, h5py.Dataset) else None)
                        if keys:
                            found_key = keys[0]
                        else:
                            raise ValueError("No valid dataset found in HDF5 file.")
                    ds = h5f[found_key]

                # Read data and convert to complex64
                data = ds[()]
                if np.iscomplexobj(data):
                    self.iq_data = data.astype(np.complex64).flatten()
                    self.source_format = "complex64"
                elif data.ndim == 2:
                    if data.shape[0] == 2:
                        # [2, N]
                        self.iq_data = (data[0, :].astype(np.float32) + 1j * data[1, :].astype(np.float32)).astype(np.complex64)
                    elif data.shape[1] == 2:
                        # [N, 2]
                        self.iq_data = (data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)).astype(np.complex64)
                    else:
                        self.iq_data = (data.flatten().astype(np.float32) + 0j).astype(np.complex64)
                    self.source_format = "float32"
                else:
                    # 1D array - check if raw byte/int16 format with header
                    if ds.name.endswith('raw') and len(data) > HEADER_SIZE:
                        if data.dtype in [np.uint8, np.int8]:
                            samples = data[HEADER_SIZE:].view('<i2')
                            if len(samples) % 2 != 0:
                                samples = samples[:len(samples) - 1]
                            i_samples = samples[0::2].astype(np.float32) / SCALE
                            q_samples = samples[1::2].astype(np.float32) / SCALE
                            self.iq_data = (i_samples + 1j * q_samples).astype(np.complex64)
                            self.source_format = "int16"
                        elif data.dtype == np.int16:
                            samples = data[HEADER_SIZE // 2:]
                            if len(samples) % 2 != 0:
                                samples = samples[:len(samples) - 1]
                            i_samples = samples[0::2].astype(np.float32) / SCALE
                            q_samples = samples[1::2].astype(np.float32) / SCALE
                            self.iq_data = (i_samples + 1j * q_samples).astype(np.complex64)
                            self.source_format = "int16"
                        else:
                            flat = data.flatten()
                            self.iq_data = (flat[0::2].astype(np.float32) + 1j * flat[1::2].astype(np.float32)).astype(np.complex64)
                            self.source_format = "float32"
                    else:
                        flat = data.flatten()
                        if flat.dtype == np.int16:
                            if len(flat) % 2 != 0:
                                flat = flat[:len(flat) - 1]
                            i_samples = flat[0::2].astype(np.float32) / SCALE
                            q_samples = flat[1::2].astype(np.float32) / SCALE
                            self.iq_data = (i_samples + 1j * q_samples).astype(np.complex64)
                            self.source_format = "int16"
                        elif len(flat) % 2 == 0:
                            self.iq_data = (flat[0::2].astype(np.float32) + 1j * flat[1::2].astype(np.float32)).astype(np.complex64)
                            self.source_format = "float32"
                        else:
                            self.iq_data = (flat.astype(np.float32) + 0j).astype(np.complex64)
                            self.source_format = "float32"

        if fs is not None:
            self.fs = float(fs)

        self.total_samples = len(self.iq_data)
        self._calculate_chunks()
        self._initialize_cache_for_mode()
        return self.get_summary()

    def set_direct_data(self, iq_data: np.ndarray, filename: str = "generated_signal.iq", fs: Optional[float] = None, source_format: str = "float32"):
        """Directly set IQ numpy array."""
        self.iq_data = iq_data.astype(np.complex64)
        self.source_filename = filename
        self.source_format = source_format
        if fs is not None:
            self.fs = float(fs)
        self.total_samples = len(self.iq_data)
        self._calculate_chunks()
        self._initialize_cache_for_mode()

    def set_config(self,
                   fs: Optional[float] = None,
                   center_freq: Optional[float] = None,
                   chunk_duration_ms: Optional[float] = None,
                   overlap_duration_ms: Optional[float] = None,
                   nfft: Optional[int] = None,
                   hop_length: Optional[int] = None,
                   window: Optional[str] = None,
                   colormap: Optional[str] = None,
                   db_min: Optional[float] = None,
                   db_max: Optional[float] = None,
                   eager_threshold: Optional[int] = None,
                   batch_size: Optional[int] = None):
        """Update STFT / chunking configuration."""
        if fs is not None:
            self.fs = float(fs)
        if center_freq is not None:
            self.center_freq = float(center_freq)
        if chunk_duration_ms is not None:
            self.chunk_duration_ms = float(chunk_duration_ms)
        if overlap_duration_ms is not None:
            self.overlap_duration_ms = float(overlap_duration_ms)
        if nfft is not None:
            self.nfft = int(nfft)
        if hop_length is not None:
            self.hop_length = int(hop_length)
        elif nfft is not None:
            self.hop_length = int(self.nfft // 4)
        if window is not None:
            self.window = str(window)
        if colormap is not None and colormap in self.COLORMAPS:
            self.colormap = str(colormap)
        if db_min is not None:
            self.db_min = float(db_min)
        if db_max is not None:
            self.db_max = float(db_max)
        if eager_threshold is not None:
            self.eager_threshold = int(eager_threshold)
        if batch_size is not None:
            self.batch_size = int(batch_size)

        self._image_cache.clear()
        self._array_cache.clear()
        self._noise_floor_cache.clear()
        self.active_batch_range = (0, 0)
        if self.iq_data is not None:
            self._calculate_chunks()
            self._initialize_cache_for_mode()

    def _calculate_chunks(self):
        """Divides the IQ data into overlapping time chunks."""
        self._image_cache.clear()
        self._array_cache.clear()
        self._noise_floor_cache.clear()
        self.active_batch_range = (0, 0)
        if self.iq_data is None or self.total_samples == 0:
            self.chunks_meta = []
            return

        samples_per_chunk = max(int(round(self.chunk_duration_ms * 1e-3 * self.fs)), self.nfft * 2)
        if self.overlap_duration_ms > 0:
            effective_overlap_ms = min(self.overlap_duration_ms, self.chunk_duration_ms * 0.9)
            overlap_samples = int(round(effective_overlap_ms * 1e-3 * self.fs))
            step_samples = max(samples_per_chunk - overlap_samples, 1)
        else:
            step_samples = samples_per_chunk

        self.chunks_meta = []
        chunk_id = 0
        start_idx = 0

        while start_idx < self.total_samples:
            end_idx = min(start_idx + samples_per_chunk, self.total_samples)
            duration_s = (end_idx - start_idx) / self.fs

            start_time_us = (start_idx / self.fs) * 1e6
            end_time_us = (end_idx / self.fs) * 1e6

            self.chunks_meta.append({
                "id": chunk_id,
                "file_name": f"chunk_{chunk_id:04d}.png",
                "start_idx": start_idx,
                "end_idx": end_idx,
                "num_samples": end_idx - start_idx,
                "start_time_us": start_time_us,
                "end_time_us": end_time_us,
                "duration_ms": duration_s * 1e3,
                "freq_min_mhz": (self.center_freq - self.fs / 2.0) / 1e6,
                "freq_max_mhz": (self.center_freq + self.fs / 2.0) / 1e6,
                "fs_mhz": self.fs / 1e6,
                "center_freq_mhz": self.center_freq / 1e6
            })

            chunk_id += 1
            if end_idx >= self.total_samples:
                break
            start_idx += step_samples

    def _cache_key(self, chunk_id: int, width: Optional[int], height: Optional[int]) -> str:
        return f"{chunk_id}_{self.nfft}_{self.hop_length}_{self.colormap}_{self.window}_{self.db_min}_{self.db_max}_{width}_{height}"

    def _initialize_cache_for_mode(self):
        """Initializes cache based on current render mode."""
        self._image_cache.clear()
        self._array_cache.clear()
        self._noise_floor_cache.clear()
        self.active_batch_range = (0, 0)

        if not self.chunks_meta or self.iq_data is None:
            return

        if self.render_mode == "eager":
            self.precompute_all_spectrograms()
        else:
            self.ensure_batch_cached(0)

    def _evict_inactive_chunks(self, keep_chunk_ids: set):
        """
        Evicts cached image buffers and array matrices for chunks not in keep_chunk_ids.
        Keeps RAM usage strictly bounded.
        """
        keys_to_del = [k for k in list(self._image_cache.keys()) if int(k.split('_')[0]) not in keep_chunk_ids]
        for k in keys_to_del:
            self._image_cache.pop(k, None)

        arr_keys_to_del = [k for k in list(self._array_cache.keys()) if int(k.split('_')[0]) not in keep_chunk_ids]
        for k in arr_keys_to_del:
            self._array_cache.pop(k, None)

        noise_keys_to_del = [cid for cid in list(self._noise_floor_cache.keys()) if cid not in keep_chunk_ids]
        for cid in noise_keys_to_del:
            self._noise_floor_cache.pop(cid, None)

    def ensure_batch_cached(self, target_chunk_id: int, width: int = 1024, height: int = 512, max_workers: int = 4):
        """
        Ensures the batch containing target_chunk_id is precomputed and in cache.
        If in batched mode and target_chunk_id is outside the currently active batch window,
        evicts the previous batch from memory and pre-renders the new batch.
        """
        if not self.chunks_meta or self.iq_data is None:
            return

        total_chunks = len(self.chunks_meta)
        if total_chunks <= self.eager_threshold:
            return

        # Calculate batch window
        batch_start = (target_chunk_id // self.batch_size) * self.batch_size
        batch_end = min(batch_start + self.batch_size, total_chunks)

        with self._lock:
            # Check if active batch range already covers this
            if self.active_batch_range == (batch_start, batch_end):
                target_key = self._cache_key(target_chunk_id, width, height)
                if target_key in self._image_cache:
                    return

            keep_ids = set(range(batch_start, batch_end))
            self._evict_inactive_chunks(keep_ids)
            self.active_batch_range = (batch_start, batch_end)

            def _render_chunk(c_id):
                try:
                    self.render_spectrogram_image(c_id, width=width, height=height, _skip_batch_check=True)
                    self.render_spectrogram_image(c_id, width=160, height=90, _skip_batch_check=True)
                    if c_id not in self._noise_floor_cache:
                        power_db, _, _, _ = self.compute_spectrogram(c_id)
                        linear_all_power = 10.0 ** (power_db / 10.0)
                        self._noise_floor_cache[c_id] = max(float(np.percentile(linear_all_power, 15)), 1e-12)
                except Exception:
                    pass

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(_render_chunk, range(batch_start, batch_end)))

    def precompute_all_spectrograms(self, width: int = 1024, height: int = 512, max_workers: int = 8):
        """
        Precomputes and caches all spectrogram chunk images in parallel (for Eager mode)
        so user switching between chunks is instantaneous (0ms delay).
        """
        if not self.chunks_meta or self.iq_data is None:
            return

        def _render_task(c_id):
            try:
                # Pre-render main canvas image (1024x512)
                self.render_spectrogram_image(c_id, width=width, height=height, _skip_batch_check=True)
                # Pre-render filmstrip thumbnail (160x90)
                self.render_spectrogram_image(c_id, width=160, height=90, _skip_batch_check=True)
                # Pre-cache noise floor for instantaneous physical SNR estimation
                if c_id not in self._noise_floor_cache:
                    power_db, _, _, _ = self.compute_spectrogram(c_id)
                    linear_all_power = 10.0 ** (power_db / 10.0)
                    self._noise_floor_cache[c_id] = max(float(np.percentile(linear_all_power, 15)), 1e-12)
            except Exception:
                pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_render_task, range(len(self.chunks_meta))))

    def render_waterfall_video(self,
                               output_filepath: str,
                               fps: float = 10.0,
                               width: int = 1024,
                               height: int = 512,
                               add_overlay: bool = True) -> Dict[str, Any]:
        """
        Renders an animated waterfall MP4 video consisting of all overlapping chunk frames
        with time/frequency annotations overlay.
        Uses H.264/yuv420p video encoding for universal HTML5 browser playback.
        """
        import cv2
        import subprocess
        if not self.chunks_meta:
            raise ValueError("No chunks available to render waterfall video.")

        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        num_frames = len(self.chunks_meta)
        frames_bgr = []

        for c_id in range(num_frames):
            img_bytes, meta = self.render_spectrogram_image(c_id, width=width, height=height)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            frame_rgb = np.array(pil_img)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if add_overlay:
                t_start_ms = meta["start_time_us"] / 1000.0
                t_end_ms = meta["end_time_us"] / 1000.0
                hud_text1 = f"Chunk {c_id+1}/{num_frames} | Time: {t_start_ms:.2f} - {t_end_ms:.2f} ms | Fs: {self.fs/1e6:.2f} MHz"
                hud_text2 = f"Span: {meta['freq_min_mhz']:.2f} to {meta['freq_max_mhz']:.2f} MHz | NFFT: {self.nfft} | {self.colormap.upper()}"

                overlay = frame_bgr.copy()
                cv2.rectangle(overlay, (10, 10), (width - 10, 56), (20, 24, 33), -1)
                cv2.addWeighted(overlay, 0.75, frame_bgr, 0.25, 0, frame_bgr)
                cv2.rectangle(frame_bgr, (10, 10), (width - 10, 56), (0, 229, 255), 1)

                cv2.putText(frame_bgr, hud_text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1, cv2.LINE_AA)
                cv2.putText(frame_bgr, hud_text2, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 225), 1, cv2.LINE_AA)

            frames_bgr.append(frame_bgr)

        # Check for ffmpeg to encode browser-compatible H.264 (avc1/yuv420p) video
        ffmpeg_exe = None
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            import shutil
            ffmpeg_exe = shutil.which("ffmpeg")

        encoded_with_ffmpeg = False
        if ffmpeg_exe:
            try:
                cmd = [
                    ffmpeg_exe, '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{width}x{height}',
                    '-pix_fmt', 'bgr24',
                    '-r', str(fps),
                    '-i', '-',
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    output_filepath
                ]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                for f in frames_bgr:
                    proc.stdin.write(f.tobytes())
                proc.stdin.close()
                proc.wait()
                if proc.returncode == 0 and os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
                    encoded_with_ffmpeg = True
            except Exception as e:
                print(f"Warning: ffmpeg encoding failed ({e}), falling back to OpenCV VideoWriter.")

        if not encoded_with_ffmpeg:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_filepath, fourcc, float(fps), (width, height))
            for f in frames_bgr:
                video_writer.write(f)
            video_writer.release()

        file_size = os.path.getsize(output_filepath) if os.path.exists(output_filepath) else 0

        return {
            "video_filepath": output_filepath,
            "num_frames": num_frames,
            "fps": fps,
            "duration_s": round(num_frames / fps, 2) if fps > 0 else 0,
            "width": width,
            "height": height,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "codec": "libx264" if encoded_with_ffmpeg else "mp4v"
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return summary of current dataset and session."""
        total_duration_ms = (self.total_samples / self.fs * 1e3) if self.fs > 0 else 0.0
        return {
            "source_filename": self.source_filename,
            "source_format": self.source_format,
            "fs_hz": self.fs,
            "fs_mhz": self.fs / 1e6,
            "center_freq_hz": self.center_freq,
            "center_freq_mhz": self.center_freq / 1e6,
            "total_samples": self.total_samples,
            "total_duration_ms": total_duration_ms,
            "total_duration_us": total_duration_ms * 1000.0,
            "chunk_duration_ms": self.chunk_duration_ms,
            "overlap_duration_ms": self.overlap_duration_ms,
            "num_chunks": len(self.chunks_meta),
            "render_mode": self.render_mode,
            "eager_threshold": self.eager_threshold,
            "batch_size": self.batch_size,
            "active_batch_range": list(self.active_batch_range),
            "stft_config": {
                "nfft": self.nfft,
                "hop_length": self.hop_length,
                "window": self.window,
                "colormap": self.colormap,
                "db_min": self.db_min,
                "db_max": self.db_max,
                "supported_nfft": self.SUPPORTED_NFFT
            },
            "supported_fs": [f / 1e6 for f in self.SUPPORTED_FS],
            "supported_nfft": self.SUPPORTED_NFFT
        }

    def compute_spectrogram(self, chunk_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Compute STFT spectrogram for a specific chunk.
        Returns (power_db_matrix, time_array_us, freq_array_mhz, chunk_info).
        """
        if self.iq_data is None or chunk_id < 0 or chunk_id >= len(self.chunks_meta):
            raise IndexError(f"Chunk ID {chunk_id} out of bounds.")

        meta = self.chunks_meta[chunk_id]
        chunk_samples = self.iq_data[meta["start_idx"]:meta["end_idx"]]

        # If chunk is shorter than nfft, pad with zeros
        if len(chunk_samples) < self.nfft:
            chunk_samples = np.pad(chunk_samples, (0, self.nfft - len(chunk_samples)), 'constant')

        noverlap = self.nfft - self.hop_length
        f, t, Zxx = signal.spectrogram(
            chunk_samples,
            fs=self.fs,
            window=self.window,
            nperseg=self.nfft,
            noverlap=noverlap,
            nfft=self.nfft,
            return_onesided=False,
            mode='complex',
            scaling='spectrum'
        )

        # FFT shift along frequency axis to center DC
        f_shifted = np.fft.fftshift(f)
        Zxx_shifted = np.fft.fftshift(Zxx, axes=0)

        # Compute power spectral density in dBFS
        power = np.abs(Zxx_shifted) ** 2
        power_db = 10.0 * np.log10(power + 1e-12)

        # Frequency in MHz (including center frequency)
        freq_mhz = (f_shifted + self.center_freq) / 1e6

        # Time in microseconds relative to start of file
        time_us = meta["start_time_us"] + (t * 1e6)

        return power_db, time_us, freq_mhz, meta

    def render_spectrogram_image(self, chunk_id: int, width: Optional[int] = None, height: Optional[int] = None, _skip_batch_check: bool = False) -> Tuple[bytes, Dict[str, Any]]:
        """
        Render the STFT spectrogram of chunk_id as an RGB image buffer in PNG format
        with high-contrast auto-adaptive dynamic range and instant cache retrieval.
        """
        key = self._cache_key(chunk_id, width, height)
        if key in self._image_cache:
            return self._image_cache[key]

        if not _skip_batch_check and self.render_mode == "batched":
            b_start, b_end = self.active_batch_range
            if chunk_id < b_start or chunk_id >= b_end:
                self.ensure_batch_cached(chunk_id, width=width or 1024, height=height or 512)
                if key in self._image_cache:
                    return self._image_cache[key]

        power_db, time_us, freq_mhz, meta = self.compute_spectrogram(chunk_id)

        # Robust Auto-Adaptive Dynamic Range Contrast
        if self.db_min is None or self.db_max is None or self.db_min >= self.db_max or (self.db_min == -90.0 and self.db_max == 0.0):
            p_noise = float(np.percentile(power_db, 20))
            p_peak = float(np.percentile(power_db, 99.8))
            dyn_range = max(p_peak - p_noise, 25.0)
            p_min = p_noise - 3.0
            p_max = p_noise + dyn_range + 5.0
        else:
            p_min = self.db_min
            p_max = self.db_max

        norm_power = np.clip((power_db - p_min) / (p_max - p_min + 1e-6), 0.0, 1.0)

        # Flip vertically so high frequencies are at top, low frequencies at bottom (standard RF display)
        norm_power_flipped = np.flipud(norm_power)
        uint8_power = (norm_power_flipped * 255.0).astype(np.uint8)

        # Apply Colormap: OpenCV cv2.COLORMAP_HOT / cv2.COLORMAP_TURBO / etc.
        cmap_name = self.colormap.lower()
        if cmap_name in self.OPENCV_COLORMAPS:
            cv_cmap = self.OPENCV_COLORMAPS[cmap_name]
            bgr_img = cv2.applyColorMap(uint8_power, cv_cmap)
            if width and height and (bgr_img.shape[1] != width or bgr_img.shape[0] != height):
                bgr_img = cv2.resize(bgr_img, (width, height), interpolation=cv2.INTER_LINEAR)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_img)
        elif cmap_name == 'gray':
            if width and height and (uint8_power.shape[1] != width or uint8_power.shape[0] != height):
                uint8_power = cv2.resize(uint8_power, (width, height), interpolation=cv2.INTER_LINEAR)
            rgb_img = cv2.cvtColor(uint8_power, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(rgb_img)
        else:
            cmap = plt.get_cmap(self.colormap)
            rgba_img = (cmap(norm_power_flipped) * 255).astype(np.uint8)
            img = Image.fromarray(rgba_img)
            if width and height:
                img = img.resize((width, height), Image.Resampling.BILINEAR)

        # Save to PNG in memory with fast compression
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=1)
        img_bytes = buf.getvalue()

        render_meta = {
            **meta,
            "width": img.width,
            "height": img.height,
            "raw_stft_width": power_db.shape[1],
            "raw_stft_height": power_db.shape[0],
            "db_min": round(p_min, 1),
            "db_max": round(p_max, 1),
            "colormap": self.colormap
        }

        self._image_cache[key] = (img_bytes, render_meta)
        return img_bytes, render_meta

    def get_spectrogram_raw_image(self, chunk_id: int, width: Optional[int] = 1024, height: Optional[int] = 512) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Directly returns the uint8 normalized spectrogram image array and metadata without PNG compression,
        providing maximum performance for automated detectors and processing pipelines.
        """
        key = self._cache_key(chunk_id, width, height)
        if key in self._array_cache:
            return self._array_cache[key]

        power_db, time_us, freq_mhz, meta = self.compute_spectrogram(chunk_id)

        if self.db_min is None or self.db_max is None or self.db_min >= self.db_max or (self.db_min == -90.0 and self.db_max == 0.0):
            p_noise = float(np.percentile(power_db, 20))
            p_peak = float(np.percentile(power_db, 99.8))
            dyn_range = max(p_peak - p_noise, 25.0)
            p_min = p_noise - 3.0
            p_max = p_noise + dyn_range + 5.0
        else:
            p_min = self.db_min
            p_max = self.db_max

        norm_power = np.clip((power_db - p_min) / (p_max - p_min + 1e-6) * 255.0, 0.0, 255.0).astype(np.uint8)
        norm_power_flipped = np.flipud(norm_power)

        if width and height and (norm_power_flipped.shape[1] != width or norm_power_flipped.shape[0] != height):
            img_arr = cv2.resize(norm_power_flipped, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            img_arr = norm_power_flipped

        render_meta = {
            **meta,
            "width": img_arr.shape[1],
            "height": img_arr.shape[0],
            "raw_stft_width": power_db.shape[1],
            "raw_stft_height": power_db.shape[0],
            "db_min": round(p_min, 1),
            "db_max": round(p_max, 1),
            "colormap": self.colormap
        }

        self._array_cache[key] = (img_arr, render_meta)
        return img_arr, render_meta

    def calculate_bdw_parameters(self,
                                 chunk_id: int,
                                 bbox: List[float],
                                 img_width: int,
                                 img_height: int,
                                 signal_type: str = "Unknown",
                                 protocol: str = "Generic") -> Dict[str, Any]:
        """
        Convert pixel bounding box [x, y, w, h] to physical RF BDW parameters:
        TOA (Time of Arrival), TOD (Time of Departure), PW (Pulse Width),
        FC (Center Frequency), BW (Bandwidth), and estimated SNR (dB).
        """
        if chunk_id < 0 or chunk_id >= len(self.chunks_meta):
            raise IndexError(f"Chunk ID {chunk_id} out of range.")

        meta = self.chunks_meta[chunk_id]
        x, y, w, h = bbox

        # Time range mapping (X axis: left to right)
        t_start_chunk = meta["start_time_us"]
        t_end_chunk = meta["end_time_us"]
        t_span = t_end_chunk - t_start_chunk

        toa_us = t_start_chunk + (x / float(img_width)) * t_span
        tod_us = t_start_chunk + ((x + w) / float(img_width)) * t_span
        pw_us = max(tod_us - toa_us, 0.0)

        # Frequency range mapping (Y axis: top is freq_max, bottom is freq_min)
        f_min = meta["freq_min_mhz"]
        f_max = meta["freq_max_mhz"]
        f_span = f_max - f_min

        # Note: image y=0 is top (f_max), y=height is bottom (f_min)
        freq_high_mhz = f_max - (y / float(img_height)) * f_span
        freq_low_mhz = f_max - ((y + h) / float(img_height)) * f_span

        bw_mhz = max(freq_high_mhz - freq_low_mhz, 0.0)
        fc_mhz = (freq_high_mhz + freq_low_mhz) / 2.0

        # Estimate SNR from spectrogram with cached noise floor
        snr_db = self._estimate_box_snr(chunk_id, x, y, w, h, img_width, img_height)

        return {
            "toa_us": round(float(toa_us), 3),
            "tod_us": round(float(tod_us), 3),
            "pw_us": round(float(pw_us), 3),
            "fc_mhz": round(float(fc_mhz), 4),
            "bw_mhz": round(float(bw_mhz), 4),
            "freq_low_mhz": round(float(freq_low_mhz), 4),
            "freq_high_mhz": round(float(freq_high_mhz), 4),
            "snr_db": round(float(snr_db), 1),
            "type_of_signal": signal_type,
            "protocol": protocol,
            "data_source": self.source_filename
        }

    def _estimate_box_snr(self, chunk_id: int, x: float, y: float, w: float, h: float, img_width: int, img_height: int) -> float:
        """
        Calculates estimated Signal-to-Noise Ratio (SNR) in dB within the bounding box.
        Uses single-pass cached chunk noise floor for ultra-fast evaluation.
        """
        try:
            power_db, _, _, _ = self.compute_spectrogram(chunk_id)
            n_freq, n_time = power_db.shape

            # Map image coords to power_db matrix indices
            x_min_idx = int(np.clip((x / img_width) * n_time, 0, n_time - 1))
            x_max_idx = int(np.clip(((x + w) / img_width) * n_time, 1, n_time))

            # Since spectrogram Y is inverted relative to standard fftshift (row 0 is f_min):
            y_top_idx = int(np.clip(((img_height - y) / img_height) * n_freq, 1, n_freq))
            y_bot_idx = int(np.clip(((img_height - (y + h)) / img_height) * n_freq, 0, n_freq - 1))

            if y_top_idx <= y_bot_idx:
                y_top_idx = min(y_bot_idx + 1, n_freq)

            box_power_db = power_db[y_bot_idx:y_top_idx, x_min_idx:x_max_idx]
            if box_power_db.size == 0:
                return 15.0

            # Signal power: 80th percentile of linear power in box ROI only
            linear_box_power = 10.0 ** (box_power_db / 10.0)
            sig_power = float(np.percentile(linear_box_power, 80))

            # Noise floor power: cached 15th percentile of entire chunk spectrogram
            if chunk_id in self._noise_floor_cache:
                noise_power = self._noise_floor_cache[chunk_id]
            else:
                linear_all_power = 10.0 ** (power_db / 10.0)
                noise_power = max(float(np.percentile(linear_all_power, 15)), 1e-12)
                self._noise_floor_cache[chunk_id] = noise_power

            snr_linear = sig_power / noise_power
            snr_db = 10.0 * np.log10(max(snr_linear, 1.0))
            return float(np.clip(snr_db, 0.0, 50.0))
        except Exception:
            return 18.5
