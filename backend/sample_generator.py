import os
import math
import numpy as np
import h5py
from typing import Dict, Any, List, Tuple

class RFSampleGenerator:
    """
    Generates realistic synthetic RF IQ datasets (.h5 and .iq) for testing
    and benchmarking at 245.76 MHz, 61.44 MHz, and 30.72 MHz.
    """

    @staticmethod
    def generate_synthetic_rf(fs: float = 61.44e6,
                             center_freq: float = 2400.0e6,
                             duration_ms: float = 100.0,
                             iq_format: str = "float32",
                             snr_target_db: float = 20.0,
                             output_format: str = "h5",
                             output_path: str = "/home/dev/labelling_tool/samples/sample_capture.h5") -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generates multi-signal RF environment with FMCW, OFDM, FHSS, and CW pulses.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        total_samples = int(round(duration_ms * 1e-3 * fs))
        t = np.arange(total_samples) / fs

        # Base noise floor
        noise_i = np.random.normal(0, 0.05, total_samples).astype(np.float32)
        noise_q = np.random.normal(0, 0.05, total_samples).astype(np.float32)
        iq_signal = noise_i + 1j * noise_q

        ground_truth_boxes = []

        # Signal 1: FMCW Radar Chirps (e.g. 77GHz baseband chirp sweeping from -15 MHz to +15 MHz)
        # Multiple chirps repeating every 1.2 ms, pulse width 0.6 ms
        chirp_period_s = 1.2e-3
        chirp_pw_s = 0.6e-3
        chirp_f0 = -15e6
        chirp_f1 = 15e6
        k_chirp = (chirp_f1 - chirp_f0) / chirp_pw_s

        num_chirps = int(math.ceil((duration_ms * 1e-3) / chirp_period_s))
        for c in range(num_chirps):
            c_start_s = c * chirp_period_s + 0.1e-3
            c_end_s = c_start_s + chirp_pw_s
            if c_end_s > duration_ms * 1e-3:
                continue

            idx_start = int(round(c_start_s * fs))
            idx_end = int(round(c_end_s * fs))
            t_pulse = t[idx_start:idx_end] - c_start_s

            # Phase of linear chirp: phi(t) = 2*pi*(f0*t + 0.5*k*t^2)
            phase = 2.0 * np.pi * (chirp_f0 * t_pulse + 0.5 * k_chirp * (t_pulse ** 2))
            chirp_wave = 0.8 * np.exp(1j * phase)

            # Apply smooth Tukey / Hann window
            win = np.hanning(len(t_pulse))
            iq_signal[idx_start:idx_end] += chirp_wave * win

            ground_truth_boxes.append({
                "category_name": "FMCW_Radar",
                "toa_us": c_start_s * 1e6,
                "tod_us": c_end_s * 1e6,
                "pw_us": chirp_pw_s * 1e6,
                "fc_mhz": (chirp_f0 + chirp_f1) / (2.0 * 1e6),
                "bw_mhz": abs(chirp_f1 - chirp_f0) / 1e6,
                "type_of_signal": "FMCW",
                "protocol": "Automotive Radar",
                "snr_db": 22.5
            })

        # Signal 2: WiFi / OFDM Bursts at +20 MHz offset (bandwidth 20 MHz)
        # Bursts occur at t = 0.4ms to 1.1ms, and t = 2.5ms to 3.5ms
        ofdm_bursts = [
            (0.4e-3, 0.7e-3, 18e6, 20e6),
            (2.2e-3, 0.9e-3, 18e6, 20e6),
            (3.6e-3, 0.8e-3, 18e6, 20e6)
        ]

        for b_start_s, b_pw_s, b_fc, b_bw in ofdm_bursts:
            b_end_s = b_start_s + b_pw_s
            if b_end_s > duration_ms * 1e-3:
                continue

            idx_start = int(round(b_start_s * fs))
            idx_end = int(round(b_end_s * fs))
            num_pts = idx_end - idx_start
            t_b = t[idx_start:idx_end]

            # Generate synthetic multicarrier OFDM (64 subcarriers across 20 MHz)
            num_subcarriers = 64
            carrier_spacing = b_bw / num_subcarriers
            ofdm_signal = np.zeros(num_pts, dtype=np.complex64)

            for sc in range(-num_subcarriers // 2, num_subcarriers // 2):
                if abs(sc) < 2 or abs(sc) > 28:
                    continue  # Null carriers and DC guard
                sc_freq = b_fc + sc * carrier_spacing
                sc_phase = np.random.uniform(0, 2 * np.pi)
                ofdm_signal += (0.08 * np.exp(1j * (2.0 * np.pi * sc_freq * t_b + sc_phase))).astype(np.complex64)

            # Window
            win = np.ones(num_pts, dtype=np.float32)
            edge = min(100, num_pts // 10)
            if edge > 0:
                win[:edge] = np.linspace(0, 1, edge)
                win[-edge:] = np.linspace(1, 0, edge)

            iq_signal[idx_start:idx_end] += ofdm_signal * win

            ground_truth_boxes.append({
                "category_name": "WiFi_OFDM",
                "toa_us": b_start_s * 1e6,
                "tod_us": b_end_s * 1e6,
                "pw_us": b_pw_s * 1e6,
                "fc_mhz": b_fc / 1e6,
                "bw_mhz": b_bw / 1e6,
                "type_of_signal": "OFDM",
                "protocol": "802.11ax",
                "snr_db": 20.0
            })

        # Signal 3: Frequency Hopping Spread Spectrum (FHSS) Bluetooth-like pulses
        # Short 1 MHz pulses hopping at -22 MHz, -18 MHz, -10 MHz, -5 MHz, +5 MHz, +12 MHz
        hop_freqs = [-22e6, -18e6, -10e6, -5e6, 5e6, 12e6]
        hop_duration_s = 0.25e-3
        hop_gap_s = 0.15e-3

        current_t_s = 0.2e-3
        hop_idx = 0
        while current_t_s + hop_duration_s < duration_ms * 1e-3:
            f_hop = hop_freqs[hop_idx % len(hop_freqs)]
            h_start_s = current_t_s
            h_end_s = h_start_s + hop_duration_s

            idx_start = int(round(h_start_s * fs))
            idx_end = int(round(h_end_s * fs))
            t_h = t[idx_start:idx_end]

            # GFSK / pulse tone with slight Gaussian shaping
            phase = 2.0 * np.pi * f_hop * t_h + np.random.uniform(0, 2*np.pi)
            hop_sig = 0.6 * np.exp(1j * phase)

            num_pts = idx_end - idx_start
            win = np.hanning(num_pts)
            iq_signal[idx_start:idx_end] += (hop_sig * win).astype(np.complex64)

            ground_truth_boxes.append({
                "category_name": "Bluetooth_FHSS",
                "toa_us": h_start_s * 1e6,
                "tod_us": h_end_s * 1e6,
                "pw_us": hop_duration_s * 1e6,
                "fc_mhz": f_hop / 1e6,
                "bw_mhz": 1.5,
                "type_of_signal": "GFSK_FHSS",
                "protocol": "Bluetooth 5.0",
                "snr_db": 18.0
            })

            current_t_s += (hop_duration_s + hop_gap_s)
            hop_idx += 1

        # Normalize total signal amplitude to prevent clipping
        max_val = np.max(np.abs(iq_signal))
        if max_val > 0.95:
            iq_signal = (iq_signal / max_val) * 0.9

        # Save to output file
        if output_format.lower() == "h5" or output_path.endswith(".h5") or output_path.endswith(".hdf5"):
            with h5py.File(output_path, 'w') as h5f:
                h5f.create_dataset('iq', data=iq_signal.astype(np.complex64), compression="gzip")
                h5f.attrs['fs'] = fs
                h5f.attrs['sample_rate'] = fs
                h5f.attrs['center_freq'] = center_freq
                h5f.attrs['duration_ms'] = duration_ms
                h5f.attrs['format'] = iq_format
                h5f.attrs['description'] = "Synthetic Multi-Signal RF Spectrogram Dataset"
        else:
            # Raw IQ binary
            if iq_format == "int16":
                i_int = np.clip(np.real(iq_signal) * 32767.0, -32768, 32767).astype(np.int16)
                q_int = np.clip(np.imag(iq_signal) * 32767.0, -32768, 32767).astype(np.int16)
                interleaved = np.empty(total_samples * 2, dtype=np.int16)
                interleaved[0::2] = i_int
                interleaved[1::2] = q_int
                interleaved.tofile(output_path)
            else:
                i_flt = np.real(iq_signal).astype(np.float32)
                q_flt = np.imag(iq_signal).astype(np.float32)
                interleaved = np.empty(total_samples * 2, dtype=np.float32)
                interleaved[0::2] = i_flt
                interleaved[1::2] = q_flt
                interleaved.tofile(output_path)

        return output_path, ground_truth_boxes
