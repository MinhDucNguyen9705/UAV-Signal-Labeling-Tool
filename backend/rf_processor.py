import io
import os
import re
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
import time
from backend.logger import logger

def power(iq: np.ndarray) -> float:
    """Calculates average linear signal power."""
    return float(np.mean(np.abs(iq) ** 2) + 1e-12)

def add_awgn(iq: np.ndarray, snr_db: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Injects Additive White Gaussian Noise (AWGN) to achieve a target SNR in dB.
    Optimized with direct float32 standard normal distribution generation.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise_power = power(iq) / (10 ** (float(snr_db) / 10.0))
    std = np.sqrt(noise_power / 2.0).astype(np.float32)
    noise_i = rng.standard_normal(iq.shape, dtype=np.float32) * std
    noise_q = rng.standard_normal(iq.shape, dtype=np.float32) * std
    return iq + (noise_i + 1j * noise_q)

class RFProcessor:
    """
    Processes RF IQ signals from .iq and .h5 files, performs time-chunking,
    computes STFT spectrograms, and calculates RF BDW parameters.
    Supports Dual-Mode Rendering (Eager for small datasets, Batched Sliding-Window with auto-eviction for large datasets).
    """

    SUPPORTED_FS = [245.76e6, 61.44e6, 30.72e6]
    COLORMAPS = ['turbo', 'hot', 'viridis', 'plasma', 'magma', 'inferno', 'jet', 'gray']
    SUPPORTED_ENGINES = ['opencv', 'matplotlib']
    DEFAULT_ENGINE = 'opencv'
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
                 drone_name: str = "DJI-MAVIC-PRO-3",
                 chunk_duration_ms: float = 30.0,
                 overlap_duration_ms: float = 10.0,
                 default_snr_db: float = 18.0,
                 render_width: int = 1024,
                 render_height: int = 512,
                 nfft: int = 1024,
                 hop_length: Optional[int] = None,
                 window: str = 'hann',
                 colormap: str = 'turbo',
                 colormap_engine: str = 'opencv',
                 db_min: float = -90.0,
                 db_max: float = 0.0,
                 eager_threshold: int = 20,
                 batch_size: int = 10):
        self.fs = float(fs)
        self.center_freq = float(center_freq)
        self.drone_name = str(drone_name) if drone_name else "DJI-MAVIC-PRO-3"
        self.chunk_duration_ms = float(chunk_duration_ms)
        self.overlap_duration_ms = float(overlap_duration_ms)
        self.default_snr_db = float(default_snr_db)
        self.render_width = int(render_width) if render_width else 1024
        self.render_height = int(render_height) if render_height else 512
        self.nfft = int(nfft)
        self.hop_length = int(hop_length) if hop_length else int(self.nfft // 4)
        self.window = str(window)
        self.colormap = str(colormap) if colormap in self.COLORMAPS else 'turbo'
        self.colormap_engine = str(colormap_engine).lower() if colormap_engine and str(colormap_engine).lower() in self.SUPPORTED_ENGINES else 'opencv'
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
        self._stft_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]] = {}
        self._noise_floor_cache: Dict[int, float] = {}

    @property
    def render_mode(self) -> str:
        """Returns 'eager' if chunk count <= eager_threshold, else 'batched'."""
        if len(self.chunks_meta) <= self.eager_threshold:
            return "eager"
        return "batched"

    def load_iq_file(self, filepath: str, iq_format: str = "float32", fs: Optional[float] = None, center_freq: Optional[float] = None, drone_name: Optional[str] = None, render_width: Optional[int] = None, render_height: Optional[int] = None, default_snr_db: Optional[float] = None, apply_awgn_snr_db: Optional[float] = None) -> Dict[str, Any]:
        """
        Load raw binary IQ file in int16 or float32 format with low-memory parsing.
        """
        self.iq_data = None
        if fs is not None:
            self.fs = float(fs)
        if center_freq is not None:
            self.center_freq = float(center_freq)
        if drone_name is not None:
            self.drone_name = str(drone_name).strip() or "DJI-MAVIC-PRO-3"
        if render_width is not None and render_width > 0:
            self.render_width = int(render_width)
        if render_height is not None and render_height > 0:
            self.render_height = int(render_height)
        if apply_awgn_snr_db is not None:
            self.default_snr_db = float(apply_awgn_snr_db)
        elif default_snr_db is not None:
            self.default_snr_db = float(default_snr_db)

        self.source_filename = os.path.basename(filepath)
        self.source_format = iq_format

        t0 = time.time()
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0.0
        logger.info(f"Loading IQ file '{filepath}' (Size: {file_size_mb:.2f} MB, format: {iq_format}, Fs: {self.fs / 1e6:.2f} MHz, Fc: {self.center_freq / 1e6:.2f} MHz, Res: {self.render_width}x{self.render_height})...")

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
                logger.warning(f"Detected large values in float32 file (>1000). Auto-fallback to int16 format.")
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

        if apply_awgn_snr_db is not None:
            self.iq_data = add_awgn(self.iq_data, snr_db=float(apply_awgn_snr_db))
            logger.info(f"Applied AWGN noise to IQ file (Target SNR: {apply_awgn_snr_db:.1f} dB).")

        self.total_samples = len(self.iq_data)
        elapsed_s = time.time() - t0
        logger.info(f"Loaded {self.total_samples:,} complex samples ({self.source_format}) in {elapsed_s:.3f}s. Calculating chunks...")
        self._calculate_chunks()
        self._initialize_cache_for_mode()

        return self.get_summary()

    def load_h5_file(self, filepath: str, dataset_name: Optional[str] = None, fs: Optional[float] = None, center_freq: Optional[float] = None, drone_name: Optional[str] = None, render_width: Optional[int] = None, render_height: Optional[int] = None, default_snr_db: Optional[float] = None, apply_awgn_snr_db: Optional[float] = None) -> Dict[str, Any]:
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
        t0 = time.time()

        if drone_name is not None:
            self.drone_name = str(drone_name).strip() or "DJI-MAVIC-PRO-3"
        if render_width is not None and render_width > 0:
            self.render_width = int(render_width)
        if render_height is not None and render_height > 0:
            self.render_height = int(render_height)
        if apply_awgn_snr_db is not None:
            self.default_snr_db = float(apply_awgn_snr_db)
        elif default_snr_db is not None:
            self.default_snr_db = float(default_snr_db)

        logger.info(f"Loading HDF5 file '{filepath}' (Fs: {self.fs / 1e6:.2f} MHz, Fc: {self.center_freq / 1e6:.2f} MHz, Res: {self.render_width}x{self.render_height})...")

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

        if self.iq_data is None:
            raise ValueError("No valid dataset or IQ data found in HDF5 file.")

        if apply_awgn_snr_db is not None:
            self.iq_data = add_awgn(self.iq_data, snr_db=float(apply_awgn_snr_db))
            logger.info(f"Applied AWGN noise to HDF5 signal (Target SNR: {apply_awgn_snr_db:.1f} dB).")

        self.total_samples = len(self.iq_data)
        elapsed_s = time.time() - t0
        logger.info(f"Loaded {self.total_samples:,} complex samples from HDF5 ({self.source_format}) in {elapsed_s:.3f}s. Calculating chunks...")
        self._calculate_chunks()
        self._initialize_cache_for_mode()

        return self.get_summary()

    def apply_awgn(self, snr_db: float, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """
        Injects Additive White Gaussian Noise (AWGN) into the currently loaded IQ data
        to achieve the specified target SNR in dB and recalculates spectrograms.
        """
        if self.iq_data is None:
            raise RuntimeError("No IQ data loaded to apply AWGN noise.")

        self.iq_data = add_awgn(self.iq_data, snr_db=float(snr_db), rng=rng)
        self.default_snr_db = float(snr_db)
        logger.info(f"Dynamically applied AWGN noise to IQ signal (Target SNR: {snr_db:.1f} dB).")

        self._image_cache.clear()
        self._array_cache.clear()
        self._stft_cache.clear()
        self._noise_floor_cache.clear()
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
                   drone_name: Optional[str] = None,
                   chunk_duration_ms: Optional[float] = None,
                   overlap_duration_ms: Optional[float] = None,
                   default_snr_db: Optional[float] = None,
                   render_width: Optional[int] = None,
                   render_height: Optional[int] = None,
                   nfft: Optional[int] = None,
                   hop_length: Optional[int] = None,
                   window: Optional[str] = None,
                   colormap: Optional[str] = None,
                   colormap_engine: Optional[str] = None,
                   db_min: Optional[float] = None,
                   db_max: Optional[float] = None):
        """Update signal and STFT processing parameters."""
        if fs is not None:
            self.fs = float(fs)
        if center_freq is not None:
            self.center_freq = float(center_freq)
        if drone_name is not None:
            self.drone_name = str(drone_name).strip() or "DJI-MAVIC-PRO-3"
        if chunk_duration_ms is not None:
            self.chunk_duration_ms = float(chunk_duration_ms)
        if overlap_duration_ms is not None:
            self.overlap_duration_ms = float(overlap_duration_ms)
        if default_snr_db is not None:
            self.default_snr_db = float(default_snr_db)
        if render_width is not None and render_width > 0:
            self.render_width = int(render_width)
        if render_height is not None and render_height > 0:
            self.render_height = int(render_height)
        if nfft is not None:
            self.nfft = int(nfft)
            self.hop_length = int(hop_length) if hop_length else int(self.nfft // 4)
        elif hop_length is not None:
            self.hop_length = int(hop_length)
        if window is not None:
            self.window = str(window)
        if colormap is not None and colormap in self.COLORMAPS:
            self.colormap = str(colormap)
        if colormap_engine is not None and str(colormap_engine).lower() in self.SUPPORTED_ENGINES:
            self.colormap_engine = str(colormap_engine).lower()
        if db_min is not None:
            self.db_min = float(db_min)
        if db_max is not None:
            self.db_max = float(db_max)

        self._image_cache.clear()
        self._array_cache.clear()
        self._stft_cache.clear()
        self._noise_floor_cache.clear()
        self.active_batch_range = (0, 0)
        if self.iq_data is not None:
            self._calculate_chunks()
            self._initialize_cache_for_mode()

    def get_formatted_filename(self, chunk_id: Optional[int] = None, extension: str = "png", drone_name: Optional[str] = None) -> str:
        """
        Generates standard formatted filename:
        <drone_name>_<frequency sample>_<frequency center>_<dtype>[_chunk_id].ext
        E.g.: DJI-MAVIC-PRO-3_100MHz_2450MHz_float.iq / .txt / .jpg
        """
        target_drone = drone_name or self.drone_name or "DJI-MAVIC-PRO-3"
        clean_drone = re.sub(r'[\s_]+', '-', target_drone.strip())
        
        fs_mhz = self.fs / 1e6
        fs_str = f"{fs_mhz:g}MHz"
        
        fc_mhz = self.center_freq / 1e6
        fc_str = f"{fc_mhz:g}MHz"
        
        dtype_lower = (self.source_format or "float32").lower()
        dtype_str = "float" if "float" in dtype_lower or "complex" in dtype_lower else "int16"
        
        base = f"{clean_drone}_{fs_str}_{fc_str}_{dtype_str}"
        total_chunks = len(self.chunks_meta)
        if total_chunks > 1 and chunk_id is not None:
            base = f"{base}_{chunk_id:04d}"
            
        ext = extension.lstrip('.')
        return f"{base}.{ext}" if ext else base

    def _calculate_chunks(self):
        """Divides the IQ data into overlapping time chunks."""
        self._image_cache.clear()
        self._array_cache.clear()
        self._stft_cache.clear()
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

        total_dur_ms = (self.total_samples / self.fs) * 1e3
        logger.info(f"Calculated {len(self.chunks_meta)} chunks ({total_dur_ms:.2f} ms total, chunk_dur: {self.chunk_duration_ms:.1f} ms, overlap: {self.overlap_duration_ms:.1f} ms, render_mode: {self.render_mode}).")

    def _cache_key(self, chunk_id: int, width: Optional[int], height: Optional[int], engine: Optional[str] = None, img_format: str = "PNG") -> str:
        active_engine = engine.lower() if engine and engine.lower() in self.SUPPORTED_ENGINES else self.colormap_engine
        w = width or self.render_width
        h = height or self.render_height
        return f"{chunk_id}_{self.nfft}_{self.hop_length}_{self.colormap}_{active_engine}_{self.window}_{self.db_min}_{self.db_max}_{w}_{h}_{img_format.upper()}"

    def _initialize_cache_for_mode(self):
        """
        Initializes and clears cache based on current render mode.
        """
        self._image_cache.clear()
        self._array_cache.clear()
        self._stft_cache.clear()
        self._noise_floor_cache.clear()
        self.active_batch_range = (0, 0)

    def _evict_inactive_chunks(self, keep_chunk_ids: set):
        """
        Evicts cached image buffers and array matrices for chunks not in keep_chunk_ids.
        Keeps RAM usage strictly bounded.
        """
        keys_to_del = [k for k in list(self._image_cache.keys()) if int(k.split('_')[0]) not in keep_chunk_ids]
        for k in keys_to_del:
            self._image_cache.pop(k, None)

        arr_keys_to_del = [k for k in list(self._array_cache.keys()) if int(k.replace('arr_', '').split('_')[0]) not in keep_chunk_ids]
        for k in arr_keys_to_del:
            self._array_cache.pop(k, None)

        stft_keys_to_del = [cid for cid in list(self._stft_cache.keys()) if cid not in keep_chunk_ids]
        for cid in stft_keys_to_del:
            self._stft_cache.pop(cid, None)

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

        target_w = width if width and width > 0 else self.render_width
        target_h = height if height and height > 0 else self.render_height

        # Calculate batch window
        batch_start = (target_chunk_id // self.batch_size) * self.batch_size
        batch_end = min(batch_start + self.batch_size, total_chunks)

        logger.debug(f"Prefetching chunk batch [{batch_start}:{batch_end}] ({target_w}x{target_h}, evicting other cached chunks)...")
        t0 = time.time()
        with self._lock:
            # Check if active batch range already covers this
            if self.active_batch_range == (batch_start, batch_end):
                target_key = self._cache_key(target_chunk_id, target_w, target_h)
                if target_key in self._image_cache:
                    return

            keep_ids = set(range(batch_start, batch_end))
            self._evict_inactive_chunks(keep_ids)
            self.active_batch_range = (batch_start, batch_end)

            def _render_chunk(c_id):
                try:
                    self.render_spectrogram_image(c_id, width=target_w, height=target_h, _skip_batch_check=True)
                    self.render_spectrogram_image(c_id, width=160, height=90, _skip_batch_check=True)
                    if c_id not in self._noise_floor_cache:
                        power_db, _, _, _ = self.compute_spectrogram(c_id)
                        linear_all_power = 10.0 ** (power_db / 10.0)
                        self._noise_floor_cache[c_id] = max(float(np.percentile(linear_all_power, 15)), 1e-12)
                except Exception as e:
                    logger.error(f"Error rendering chunk {c_id} in batch: {e}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(_render_chunk, range(batch_start, batch_end)))
        logger.debug(f"Finished caching batch [{batch_start}:{batch_end}] in {time.time() - t0:.3f}s.")

    def precompute_all_spectrograms(self, width: int = 1024, height: int = 512, max_workers: int = 8):
        """
        Precomputes and caches all spectrogram chunk images in parallel (for Eager mode)
        so user switching between chunks is instantaneous (0ms delay).
        """
        logger.info(f"Eager mode: Precomputing all {len(self.chunks_meta)} spectrogram chunks in parallel ({max_workers} workers)...")
        t0 = time.time()
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
            except Exception as e:
                logger.error(f"Error precomputing spectrogram chunk {c_id}: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_render_task, range(len(self.chunks_meta))))
        logger.info(f"Completed precomputing {len(self.chunks_meta)} chunks in {time.time() - t0:.3f}s.")

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
        total_dur_ms = (self.total_samples / self.fs * 1e3) if self.fs > 0 else 0.0
        return {
            "source_filename": self.source_filename,
            "source_format": self.source_format,
            "fs_hz": self.fs,
            "fs_mhz": self.fs / 1e6,
            "center_freq_hz": self.center_freq,
            "center_freq_mhz": self.center_freq / 1e6,
            "drone_name": self.drone_name,
            "render_width": self.render_width,
            "render_height": self.render_height,
            "total_samples": self.total_samples,
            "total_duration_ms": total_dur_ms,
            "total_duration_us": total_dur_ms * 1e3,
            "chunk_duration_ms": self.chunk_duration_ms,
            "overlap_duration_ms": self.overlap_duration_ms,
            "default_snr_db": self.default_snr_db,
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
                "colormap_engine": self.colormap_engine,
                "supported_engines": self.SUPPORTED_ENGINES,
                "db_min": self.db_min,
                "db_max": self.db_max,
                "supported_nfft": self.SUPPORTED_NFFT
            },
            "supported_fs": [f / 1e6 for f in self.SUPPORTED_FS],
            "supported_nfft": self.SUPPORTED_NFFT
        }

    def get_chunk_iq_data(self, chunk_id: int, target_snr_db: Optional[float] = None) -> np.ndarray:
        """
        Retrieves the 1D complex64 IQ slice for chunk_id, optionally injecting AWGN noise for target SNR.
        """
        if self.iq_data is None or chunk_id < 0 or chunk_id >= len(self.chunks_meta):
            raise IndexError(f"Chunk ID {chunk_id} out of bounds.")
        meta = self.chunks_meta[chunk_id]
        chunk_iq = self.iq_data[meta["start_idx"]:meta["end_idx"]].copy()
        if target_snr_db is not None:
            chunk_iq = add_awgn(chunk_iq, snr_db=float(target_snr_db))
        return chunk_iq

    def compute_spectrogram(self, chunk_id: int, target_snr_db: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Compute STFT spectrogram for a specific chunk with single-pass caching.
        Returns (power_db_matrix, time_array_us, freq_array_mhz, chunk_info).
        """
        if target_snr_db is None and chunk_id in self._stft_cache:
            return self._stft_cache[chunk_id]

        if self.iq_data is None or chunk_id < 0 or chunk_id >= len(self.chunks_meta):
            raise IndexError(f"Chunk ID {chunk_id} out of bounds.")

        meta = self.chunks_meta[chunk_id]
        chunk_samples = self.iq_data[meta["start_idx"]:meta["end_idx"]]
        if target_snr_db is not None:
            chunk_samples = add_awgn(chunk_samples, snr_db=float(target_snr_db))

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

        res = (power_db, time_us, freq_mhz, meta)
        if target_snr_db is None:
            self._stft_cache[chunk_id] = res
        return res

    def render_spectrogram_image(self,
                                 chunk_id: int,
                                 width: Optional[int] = None,
                                 height: Optional[int] = None,
                                 engine: Optional[str] = None,
                                 image_format: str = "PNG",
                                 quality: int = 95,
                                 target_snr_db: Optional[float] = None,
                                 _skip_batch_check: bool = False) -> Tuple[bytes, Dict[str, Any]]:
        """
        Render the STFT spectrogram of chunk_id as an RGB image buffer in PNG or JPEG format
        with high-contrast auto-adaptive dynamic range, selectable resolution, and instant cache retrieval.
        """
        active_engine = engine.lower() if engine and engine.lower() in self.SUPPORTED_ENGINES else self.colormap_engine
        target_w = width if width and width > 0 else self.render_width
        target_h = height if height and height > 0 else self.render_height
        img_fmt = image_format.upper() if image_format else "PNG"
        if img_fmt == "JPG":
            img_fmt = "JPEG"

        key = self._cache_key(chunk_id, target_w, target_h, engine=active_engine, img_format=img_fmt)
        if target_snr_db is None and key in self._image_cache:
            return self._image_cache[key]

        if not _skip_batch_check and self.render_mode == "batched" and target_snr_db is None:
            b_start, b_end = self.active_batch_range
            if chunk_id < b_start or chunk_id >= b_end:
                self.ensure_batch_cached(chunk_id, width=target_w, height=target_h)
                if key in self._image_cache:
                    return self._image_cache[key]

        power_db, time_us, freq_mhz, meta = self.compute_spectrogram(chunk_id, target_snr_db=target_snr_db)

        # Fast strided percentile sampling for auto-adaptive dynamic range contrast
        if self.db_min is None or self.db_max is None or self.db_min >= self.db_max or (self.db_min == -90.0 and self.db_max == 0.0):
            sub_power = power_db[::4, ::4] if power_db.size > 10000 else power_db
            p_noise = float(np.percentile(sub_power, 20))
            p_peak = float(np.percentile(sub_power, 99.8))
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

        # Colormap Rendering (OpenCV vs Matplotlib)
        if active_engine == 'matplotlib':
            try:
                cmap = plt.get_cmap(self.colormap)
            except Exception:
                cmap = plt.get_cmap('viridis')
            rgba_img = (cmap(norm_power_flipped) * 255).astype(np.uint8)
            img = Image.fromarray(rgba_img).convert("RGB")
            if img.width != target_w or img.height != target_h:
                img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        else:
            # OpenCV Rendering Engine (default / optimized for performance)
            cmap_name = self.colormap.lower()
            if cmap_name == 'gray':
                if uint8_power.shape[1] != target_w or uint8_power.shape[0] != target_h:
                    uint8_power = cv2.resize(uint8_power, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                rgb_img = cv2.cvtColor(uint8_power, cv2.COLOR_GRAY2RGB)
                img = Image.fromarray(rgb_img)
            else:
                cv_cmap = self.OPENCV_COLORMAPS.get(cmap_name, cv2.COLORMAP_TURBO)
                bgr_img = cv2.applyColorMap(uint8_power, cv_cmap)
                if bgr_img.shape[1] != target_w or bgr_img.shape[0] != target_h:
                    bgr_img = cv2.resize(bgr_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_img)

        # Save to memory buffer with requested format (PNG / JPEG)
        buf = io.BytesIO()
        if img_fmt == "JPEG":
            img.save(buf, format="JPEG", quality=quality)
        else:
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
            "colormap": self.colormap,
            "colormap_engine": active_engine,
            "image_format": img_fmt
        }

        self._image_cache[key] = (img_bytes, render_meta)
        return img_bytes, render_meta

    def render_spectrogram_image_array(self,
                                       chunk_id: int,
                                       width: Optional[int] = None,
                                       height: Optional[int] = None,
                                       engine: Optional[str] = None,
                                       _skip_batch_check: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Directly renders and returns the BGR uint8 image array without PNG encoding/decoding overhead,
        optimized for automated detector pipelines (ONNX / CFAR).
        """
        active_engine = engine.lower() if engine and engine.lower() in self.SUPPORTED_ENGINES else self.colormap_engine
        target_w = width if width and width > 0 else self.render_width
        target_h = height if height and height > 0 else self.render_height
        arr_key = f"arr_{self._cache_key(chunk_id, width, height, engine=active_engine)}"
        if arr_key in self._array_cache:
            return self._array_cache[arr_key]

        if not _skip_batch_check and self.render_mode == "batched":
            b_start, b_end = self.active_batch_range
            if chunk_id < b_start or chunk_id >= b_end:
                self.ensure_batch_cached(chunk_id, width=width or 1024, height=height or 512)
                if arr_key in self._array_cache:
                    return self._array_cache[arr_key]

        power_db, time_us, freq_mhz, meta = self.compute_spectrogram(chunk_id)

        # Fast strided percentile sampling
        if self.db_min is None or self.db_max is None or self.db_min >= self.db_max or (self.db_min == -90.0 and self.db_max == 0.0):
            sub_power = power_db[::4, ::4] if power_db.size > 10000 else power_db
            p_noise = float(np.percentile(sub_power, 20))
            p_peak = float(np.percentile(sub_power, 99.8))
            dyn_range = max(p_peak - p_noise, 25.0)
            p_min = p_noise - 3.0
            p_max = p_noise + dyn_range + 5.0
        else:
            p_min = self.db_min
            p_max = self.db_max

        norm_power = np.clip((power_db - p_min) / (p_max - p_min + 1e-6), 0.0, 1.0)
        norm_power_flipped = np.flipud(norm_power)
        uint8_power = (norm_power_flipped * 255.0).astype(np.uint8)

        cmap_name = self.colormap.lower()
        if cmap_name == 'gray':
            if width and height and (uint8_power.shape[1] != width or uint8_power.shape[0] != height):
                uint8_power = cv2.resize(uint8_power, (width, height), interpolation=cv2.INTER_LINEAR)
            bgr_img = cv2.cvtColor(uint8_power, cv2.COLOR_GRAY2BGR)
        else:
            cv_cmap = self.OPENCV_COLORMAPS.get(cmap_name, cv2.COLORMAP_TURBO)
            bgr_img = cv2.applyColorMap(uint8_power, cv_cmap)
            if width and height and (bgr_img.shape[1] != width or bgr_img.shape[0] != height):
                bgr_img = cv2.resize(bgr_img, (width, height), interpolation=cv2.INTER_LINEAR)

        render_meta = {
            **meta,
            "width": bgr_img.shape[1],
            "height": bgr_img.shape[0],
            "raw_stft_width": power_db.shape[1],
            "raw_stft_height": power_db.shape[0],
            "db_min": round(p_min, 1),
            "db_max": round(p_max, 1),
            "colormap": self.colormap,
            "colormap_engine": active_engine
        }

        self._array_cache[arr_key] = (bgr_img, render_meta)
        return bgr_img, render_meta

    def calculate_bdw_parameters(self,
                                 chunk_id: int,
                                 bbox: List[float],
                                 img_width: int,
                                 img_height: int,
                                 signal_type: str = "Unknown",
                                 protocol: str = "Generic",
                                 snr_db: Optional[float] = None) -> Dict[str, Any]:
        """
        Convert pixel bounding box [x, y, w, h] to physical RF BDW parameters:
        TOA (Time of Arrival), TOD (Time of Departure), PW (Pulse Width),
        FC (Center Frequency), BW (Bandwidth), and SNR (dB).
        Runs in O(1) instantaneous time using the user-defined/session SNR.
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

        # Use user-typed SNR or specified SNR
        final_snr_db = float(snr_db) if snr_db is not None else float(self.default_snr_db)

        return {
            "toa_us": round(float(toa_us), 3),
            "tod_us": round(float(tod_us), 3),
            "pw_us": round(float(pw_us), 3),
            "fc_mhz": round(float(fc_mhz), 4),
            "bw_mhz": round(float(bw_mhz), 4),
            "freq_low_mhz": round(float(freq_low_mhz), 4),
            "freq_high_mhz": round(float(freq_high_mhz), 4),
            "snr_db": round(final_snr_db, 1),
            "type_of_signal": signal_type,
            "protocol": protocol,
            "data_source": self.source_filename
        }
