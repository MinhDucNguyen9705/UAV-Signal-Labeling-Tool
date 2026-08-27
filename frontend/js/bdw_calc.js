/**
 * BDW (Burst / Waveform Description) RF Physical Parameter Calculator
 * Handles bi-directional conversion between pixel coordinates and RF parameters:
 * TOA (Time of Arrival), TOD (Time of Departure), PW (Pulse Width),
 * FC (Center Frequency), BW (Bandwidth), and SNR (dB).
 */

export class BDWCalculator {
  /**
   * Convert pixel bounding box [x, y, w, h] to BDW physical parameters
   */
  static pixelsToBDW(bbox, imgWidth, imgHeight, chunkMeta, category = null, dataSource = "") {
    if (!chunkMeta) return null;

    const [x, y, w, h] = bbox;
    const tStart = chunkMeta.start_time_us || 0.0;
    const tEnd = chunkMeta.end_time_us || 1000.0;
    const tSpan = Math.max(tEnd - tStart, 1e-6);

    const fMin = chunkMeta.freq_min_mhz || -30.72;
    const fMax = chunkMeta.freq_max_mhz || 30.72;
    const fSpan = Math.max(fMax - fMin, 1e-6);

    // Time calculations (X axis)
    const toaUs = tStart + (x / imgWidth) * tSpan;
    const todUs = tStart + ((x + w) / imgWidth) * tSpan;
    const pwUs = Math.max(todUs - toaUs, 0.0);

    // Frequency calculations (Y axis: top is fMax, bottom is fMin)
    const fHigh = fMax - (y / imgHeight) * fSpan;
    const fLow = fMax - ((y + h) / imgHeight) * fSpan;
    const bwMhz = Math.max(fHigh - fLow, 0.0);
    const fcMhz = (fHigh + fLow) / 2.0;

    return {
      toa_us: Number(toaUs.toFixed(3)),
      tod_us: Number(todUs.toFixed(3)),
      pw_us: Number(pwUs.toFixed(3)),
      fc_mhz: Number(fcMhz.toFixed(4)),
      bw_mhz: Number(bwMhz.toFixed(4)),
      freq_low_mhz: Number(fLow.toFixed(4)),
      freq_high_mhz: Number(fHigh.toFixed(4)),
      snr_db: 20.0, // Default estimate, updated from server STFT
      type_of_signal: category?.type_of_signal || category?.name || "Unknown",
      protocol: category?.protocol || "Generic",
      data_source: dataSource || chunkMeta.source_filename || "capture.h5"
    };
  }

  /**
   * Convert physical RF parameters back to pixel bounding box [x, y, w, h]
   */
  static bdwToPixels(bdw, imgWidth, imgHeight, chunkMeta) {
    if (!chunkMeta || !bdw) return [0, 0, 100, 100];

    const tStart = chunkMeta.start_time_us || 0.0;
    const tEnd = chunkMeta.end_time_us || 1000.0;
    const tSpan = Math.max(tEnd - tStart, 1e-6);

    const fMin = chunkMeta.freq_min_mhz || -30.72;
    const fMax = chunkMeta.freq_max_mhz || 30.72;
    const fSpan = Math.max(fMax - fMin, 1e-6);

    const toa = Number(bdw.toa_us);
    const tod = Number(bdw.tod_us);
    const fc = Number(bdw.fc_mhz);
    const bw = Number(bdw.bw_mhz);

    const x = ((toa - tStart) / tSpan) * imgWidth;
    const w = ((tod - toa) / tSpan) * imgWidth;

    const fHigh = fc + bw / 2.0;
    const y = ((fMax - fHigh) / fSpan) * imgHeight;
    const h = (bw / fSpan) * imgHeight;

    return [
      Math.max(0, Math.round(x)),
      Math.max(0, Math.round(y)),
      Math.max(4, Math.round(w)),
      Math.max(4, Math.round(h))
    ];
  }

  /**
   * Convert single pixel point (px, py) to (time_us, freq_mhz)
   */
  static pixelToCoord(px, py, imgWidth, imgHeight, chunkMeta) {
    if (!chunkMeta) return { timeUs: 0, freqMhz: 0 };

    const tStart = chunkMeta.start_time_us || 0.0;
    const tEnd = chunkMeta.end_time_us || 1000.0;
    const tSpan = tEnd - tStart;

    const fMin = chunkMeta.freq_min_mhz || -30.72;
    const fMax = chunkMeta.freq_max_mhz || 30.72;
    const fSpan = fMax - fMin;

    const timeUs = tStart + (px / imgWidth) * tSpan;
    const freqMhz = fMax - (py / imgHeight) * fSpan;

    return {
      timeUs: Number(timeUs.toFixed(3)),
      timeMs: Number((timeUs / 1000.0).toFixed(4)),
      freqMhz: Number(freqMhz.toFixed(4))
    };
  }
}
