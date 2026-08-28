/**
 * Unified Export Manager for YOLO & COCO + BDW Datasets.
 * Provides modular component selection (images, labels, csv, metadata, raw iq, waterfall video),
 * AWGN noise SNR modification, customizable resolution, live manifest package preview,
 * and post-export download summary.
 */

export class ExportManager {
  constructor(app) {
    this.app = app;
    this.currentManifest = null;
    this.lastExportUrl = null;
    this.lastExportFilename = null;
    this.initEventListeners();
  }

  initEventListeners() {
    // Dropdown / Quick action buttons
    const exportToggleBtn = document.getElementById('exportToggleBtn');
    const openExportModalBtn = document.getElementById('openExportModalBtn');
    const exportYoloBtn = document.getElementById('exportYoloBtn');
    const exportCocoBtn = document.getElementById('exportCocoBtn');
    const exportZipBtn = document.getElementById('exportZipBtn');
    const exportZipWithIqBtn = document.getElementById('exportZipWithIqBtn');
    const exportWaterfallDropdownBtn = document.getElementById('exportWaterfallDropdownBtn');
    const importCocoBtn = document.getElementById('importCocoBtn');
    const importCocoDropdownBtn = document.getElementById('importCocoDropdownBtn');
    const importCocoFileInput = document.getElementById('importCocoFileInput');

    // Main Export Modal
    const exportModal = document.getElementById('exportModal');
    const closeExportModalBtn = document.getElementById('closeExportModalBtn');
    const cancelExportBtn = document.getElementById('cancelExportBtn');
    const confirmExportBtn = document.getElementById('confirmExportBtn');

    // Format selection cards
    const formatCardYolo = document.getElementById('formatCardYolo');
    const formatCardCoco = document.getElementById('formatCardCoco');
    const exportLabelTypeName = document.getElementById('exportLabelTypeName');

    // Controls
    const exportDroneName = document.getElementById('exportDroneName');
    const exportImgFormat = document.getElementById('exportImgFormat');
    const exportResPreset = document.getElementById('exportResolutionPreset');
    const exportCustomResRow = document.getElementById('exportCustomResRow');
    const exportRenderWidth = document.getElementById('exportRenderWidth');
    const exportRenderHeight = document.getElementById('exportRenderHeight');

    // Component Checkboxes
    const exportIncImages = document.getElementById('exportIncImages');
    const exportIncLabels = document.getElementById('exportIncLabels');
    const exportIncCsv = document.getElementById('exportIncCsv');
    const exportIncMetadata = document.getElementById('exportIncMetadata');
    const exportIncIq = document.getElementById('exportIncIq');
    const exportIncVideo = document.getElementById('exportIncVideo');

    // SNR Modification
    const exportApplyAwgn = document.getElementById('exportApplyAwgn');
    const exportAwgnFields = document.getElementById('exportAwgnFields');
    const exportAwgnSnr = document.getElementById('exportAwgnSnr');

    // Export Success Modal
    const exportSuccessModal = document.getElementById('exportSuccessModal');
    const closeExportSuccessBtn = document.getElementById('closeExportSuccessBtn');
    const reDownloadExportBtn = document.getElementById('reDownloadExportBtn');

    // Open Unified Export Modal
    const openModalHandler = (defaultFormat = 'yolo', includeIq = false) => {
      if (exportDroneName && this.app.sessionSummary?.drone_name) {
        exportDroneName.value = this.app.sessionSummary.drone_name;
      }
      if (exportRenderWidth) exportRenderWidth.value = this.app.sessionSummary?.render_width || 1024;
      if (exportRenderHeight) exportRenderHeight.value = this.app.sessionSummary?.render_height || 512;
      if (exportIncIq) exportIncIq.checked = includeIq;

      this.setFormat(defaultFormat);
      this.updateManifestPreview();
      exportModal?.classList.add('active');
    };

    if (openExportModalBtn) openExportModalBtn.addEventListener('click', () => openModalHandler('yolo'));
    if (exportYoloBtn) exportYoloBtn.addEventListener('click', () => openModalHandler('yolo', false));
    if (exportCocoBtn) exportCocoBtn.addEventListener('click', () => openModalHandler('coco', false));
    if (exportZipBtn) exportZipBtn.addEventListener('click', () => openModalHandler('coco', false));
    if (exportZipWithIqBtn) exportZipWithIqBtn.addEventListener('click', () => openModalHandler('coco', true));
    if (exportWaterfallDropdownBtn) {
      exportWaterfallDropdownBtn.addEventListener('click', () => {
        document.getElementById('openWaterfallBtn')?.click();
      });
    }

    // Format card selection
    formatCardYolo?.addEventListener('click', () => this.setFormat('yolo'));
    formatCardCoco?.addEventListener('click', () => this.setFormat('coco'));

    // Input changes trigger live manifest preview update
    const previewTriggerEls = [
      exportDroneName, exportImgFormat, exportRenderWidth, exportRenderHeight,
      exportIncImages, exportIncLabels, exportIncCsv, exportIncMetadata, exportIncIq, exportIncVideo,
      exportApplyAwgn, exportAwgnSnr
    ];
    previewTriggerEls.forEach(el => {
      el?.addEventListener('input', () => this.updateManifestPreview());
      el?.addEventListener('change', () => this.updateManifestPreview());
    });

    // Resolution Presets
    exportResPreset?.addEventListener('change', () => {
      const val = exportResPreset.value;
      if (val === 'custom') {
        if (exportCustomResRow) exportCustomResRow.style.display = 'flex';
      } else if (val === 'current') {
        if (exportCustomResRow) exportCustomResRow.style.display = 'none';
        if (exportRenderWidth) exportRenderWidth.value = this.app.sessionSummary?.render_width || 1024;
        if (exportRenderHeight) exportRenderHeight.value = this.app.sessionSummary?.render_height || 512;
      } else {
        if (exportCustomResRow) exportCustomResRow.style.display = 'none';
        const [w, h] = val.split('x').map(Number);
        if (exportRenderWidth) exportRenderWidth.value = w;
        if (exportRenderHeight) exportRenderHeight.value = h;
      }
      this.updateManifestPreview();
    });

    // AWGN Fields Visibility
    exportApplyAwgn?.addEventListener('change', () => {
      if (exportAwgnFields) {
        exportAwgnFields.style.display = exportApplyAwgn.checked ? 'block' : 'none';
      }
      this.updateManifestPreview();
    });

    // Modal Close buttons
    closeExportModalBtn?.addEventListener('click', () => exportModal?.classList.remove('active'));
    cancelExportBtn?.addEventListener('click', () => exportModal?.classList.remove('active'));

    // Confirm Download Export
    confirmExportBtn?.addEventListener('click', () => this.handleConfirmExport());

    // Success Modal Close / Re-download
    closeExportSuccessBtn?.addEventListener('click', () => exportSuccessModal?.classList.remove('active'));
    reDownloadExportBtn?.addEventListener('click', () => {
      if (this.lastExportUrl && this.lastExportFilename) {
        this.triggerDownloadUrl(this.lastExportUrl, this.lastExportFilename);
      }
    });

    // Import COCO JSON
    if (importCocoBtn && importCocoFileInput) {
      importCocoBtn.addEventListener('click', () => {
        importCocoFileInput.value = '';
        importCocoFileInput.click();
      });
    }
    if (importCocoDropdownBtn && importCocoFileInput) {
      importCocoDropdownBtn.addEventListener('click', () => {
        importCocoFileInput.value = '';
        importCocoFileInput.click();
      });
    }
    if (importCocoFileInput) {
      importCocoFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
          this.importCOCO(e.target.files[0]);
        }
      });
    }
  }

  setFormat(format) {
    const formatCardYolo = document.getElementById('formatCardYolo');
    const formatCardCoco = document.getElementById('formatCardCoco');
    const radioYolo = formatCardYolo?.querySelector('input[type="radio"]');
    const radioCoco = formatCardCoco?.querySelector('input[type="radio"]');
    const exportLabelTypeName = document.getElementById('exportLabelTypeName');
    const exportImgFormat = document.getElementById('exportImgFormat');

    if (format === 'yolo') {
      formatCardYolo?.classList.add('active');
      formatCardCoco?.classList.remove('active');
      if (radioYolo) radioYolo.checked = true;
      if (exportLabelTypeName) exportLabelTypeName.textContent = 'YOLO Labels + data.yaml';
      if (exportImgFormat && !exportImgFormat.dataset.userChanged) {
        exportImgFormat.value = 'jpg';
      }
    } else {
      formatCardCoco?.classList.add('active');
      formatCardYolo?.classList.remove('active');
      if (radioCoco) radioCoco.checked = true;
      if (exportLabelTypeName) exportLabelTypeName.textContent = 'COCO JSON (annotations_coco_bdw.json)';
      if (exportImgFormat && !exportImgFormat.dataset.userChanged) {
        exportImgFormat.value = 'png';
      }
    }
    this.updateManifestPreview();
  }

  getActiveFormat() {
    return document.getElementById('formatCardCoco')?.classList.contains('active') ? 'coco' : 'yolo';
  }

  async updateManifestPreview() {
    const format = this.getActiveFormat();
    const droneName = (document.getElementById('exportDroneName')?.value || 'DJI-MAVIC-PRO-3').trim().replace(/[\s_]+/g, '-');
    const imgFormat = document.getElementById('exportImgFormat')?.value || (format === 'yolo' ? 'jpg' : 'png');
    const renderWidth = parseInt(document.getElementById('exportRenderWidth')?.value || '1024', 10);
    const renderHeight = parseInt(document.getElementById('exportRenderHeight')?.value || '512', 10);
    
    const includeImages = document.getElementById('exportIncImages')?.checked ?? true;
    const includeLabels = document.getElementById('exportIncLabels')?.checked ?? true;
    const includeCsv = document.getElementById('exportIncCsv')?.checked ?? true;
    const includeMetadata = document.getElementById('exportIncMetadata')?.checked ?? true;
    const includeIq = document.getElementById('exportIncIq')?.checked ?? false;
    const includeVideo = document.getElementById('exportIncVideo')?.checked ?? false;

    const applyAwgn = document.getElementById('exportApplyAwgn')?.checked ?? false;
    const exportSnr = applyAwgn ? parseFloat(document.getElementById('exportAwgnSnr')?.value || '10.0') : null;

    const queryParams = new URLSearchParams({
      format_type: format,
      drone_name: droneName,
      include_images: includeImages,
      include_labels: includeLabels,
      include_csv: includeCsv,
      include_metadata: includeMetadata,
      include_iq: includeIq,
      include_video: includeVideo,
      width: renderWidth,
      height: renderHeight,
      img_format: imgFormat
    });
    if (exportSnr !== null && !isNaN(exportSnr)) {
      queryParams.set('export_snr_db', exportSnr);
    }

    try {
      const resp = await fetch(`/api/export/manifest?${queryParams.toString()}`);
      if (!resp.ok) return;
      const manifest = await resp.json();
      this.currentManifest = manifest;
      this.renderManifestPreview(manifest);
    } catch (e) {
      console.warn("Could not fetch export manifest:", e);
    }
  }

  renderManifestPreview(manifest) {
    const zipNameEl = document.getElementById('manifestZipName');
    const statsBadgeEl = document.getElementById('manifestStatsBadge');
    const treeListEl = document.getElementById('manifestTreeList');

    if (zipNameEl) zipNameEl.textContent = manifest.zip_filename;
    if (statsBadgeEl) {
      statsBadgeEl.textContent = `~${manifest.estimated_size_mb} MB | ${manifest.total_files} Files`;
    }

    if (!treeListEl) return;

    let treeHtml = `<div class="tree-root">📦 ${manifest.zip_filename}</div>`;
    
    if (manifest.summary.labels > 0 && manifest.format_type === 'yolo') {
      treeHtml += `<div style="padding-left: 14px;">├── 📄 data.yaml</div>`;
    }
    if (manifest.summary.labels > 0 && manifest.format_type === 'coco') {
      treeHtml += `<div style="padding-left: 14px;">├── 📄 annotations_coco_bdw.json</div>`;
    }
    if (manifest.summary.metadata > 0) {
      treeHtml += `<div style="padding-left: 14px;">├── 📄 metadata.json</div>`;
    }
    if (manifest.summary.csv > 0) {
      treeHtml += `<div style="padding-left: 14px;">├── 📄 signal_parameters_bdw.csv</div>`;
    }

    const imgDir = manifest.format_type === 'yolo' ? 'images' : 'spectrograms';
    if (manifest.summary.images > 0) {
      treeHtml += `<div style="padding-left: 14px;"><span class="tree-folder">├── 📁 ${imgDir}/</span> <span style="color: #64748b;">(${manifest.summary.images} items, ${manifest.resolution} ${manifest.img_format.toUpperCase()})</span></div>`;
    }
    if (manifest.summary.labels > 0 && manifest.format_type === 'yolo') {
      treeHtml += `<div style="padding-left: 14px;"><span class="tree-folder">├── 📁 labels/</span> <span style="color: #64748b;">(${manifest.summary.labels} normalized .txt labels)</span></div>`;
    }
    if (manifest.summary.iq_chunks > 0) {
      const snrLabel = manifest.export_snr_db !== null ? `, ${manifest.export_snr_db} dB SNR` : '';
      treeHtml += `<div style="padding-left: 14px;"><span class="tree-folder">├── 📁 iq/</span> <span style="color: #64748b;">(${manifest.summary.iq_chunks} raw complex .iq slices${snrLabel})</span></div>`;
    }
    if (manifest.summary.video > 0) {
      treeHtml += `<div style="padding-left: 14px;"><span class="tree-folder">└── 📁 video/</span> <span class="tree-file">waterfall_${manifest.drone_name}.mp4</span></div>`;
    }

    treeListEl.innerHTML = treeHtml;
  }

  async handleConfirmExport() {
    if (this.app.saveCurrentChunkAnnotations) {
      await this.app.saveCurrentChunkAnnotations();
    }

    const exportModal = document.getElementById('exportModal');
    const format = this.getActiveFormat();
    const droneName = (document.getElementById('exportDroneName')?.value || 'DJI-MAVIC-PRO-3').trim().replace(/[\s_]+/g, '-');
    const imgFormat = document.getElementById('exportImgFormat')?.value || (format === 'yolo' ? 'jpg' : 'png');
    const renderWidth = parseInt(document.getElementById('exportRenderWidth')?.value || '1024', 10);
    const renderHeight = parseInt(document.getElementById('exportRenderHeight')?.value || '512', 10);
    
    const includeImages = document.getElementById('exportIncImages')?.checked ?? true;
    const includeLabels = document.getElementById('exportIncLabels')?.checked ?? true;
    const includeCsv = document.getElementById('exportIncCsv')?.checked ?? true;
    const includeMetadata = document.getElementById('exportIncMetadata')?.checked ?? true;
    const includeIq = document.getElementById('exportIncIq')?.checked ?? false;
    const includeVideo = document.getElementById('exportIncVideo')?.checked ?? false;

    const applyAwgn = document.getElementById('exportApplyAwgn')?.checked ?? false;
    const exportSnr = applyAwgn ? parseFloat(document.getElementById('exportAwgnSnr')?.value || '10.0') : null;

    exportModal?.classList.remove('active');

    const formatName = format === 'yolo' ? 'YOLO Dataset' : 'COCO Bundle';
    const payload = {
      format_type: format,
      drone_name: droneName,
      include_images: includeImages,
      include_labels: includeLabels,
      include_csv: includeCsv,
      include_metadata: includeMetadata,
      include_iq: includeIq,
      include_video: includeVideo,
      width: renderWidth,
      height: renderHeight,
      img_format: imgFormat,
      export_snr_db: (exportSnr !== null && !isNaN(exportSnr)) ? exportSnr : null
    };

    // Show Real-Time Progress Bar
    this.app.showProgress(true, {
      title: `Exporting ${formatName}...`,
      current: 2,
      total: 100,
      stats: "Initializing export configuration & metadata..."
    });

    try {
      // 1. Start background export job
      const startResp = await fetch('/api/export/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!startResp.ok) {
        throw new Error(`Failed to start export job (HTTP ${startResp.status})`);
      }

      const startData = await startResp.json();
      const jobId = startData.job_id;

      // 2. Poll for progress until completed or error
      let jobStatus = null;
      while (true) {
        await new Promise(resolve => setTimeout(resolve, 150));
        const statusResp = await fetch(`/api/export/status/${jobId}`);
        if (!statusResp.ok) {
          throw new Error(`Error fetching export status (HTTP ${statusResp.status})`);
        }
        jobStatus = await statusResp.json();

        // Update progress UI
        this.app.showProgress(true, {
          title: `Exporting ${formatName}...`,
          current: jobStatus.progress || 0,
          total: 100,
          stats: jobStatus.detail || "Generating files..."
        });

        // Update detailed stats text
        const progDetail = document.getElementById('globalProgressDetail');
        const progStats = document.getElementById('globalProgressStats');
        if (progDetail && jobStatus.detail) {
          progDetail.textContent = jobStatus.detail;
        }
        if (progStats && jobStatus.stats) {
          progStats.innerHTML = `<i class="fas fa-microchip" style="color: var(--accent-cyan);"></i> <span>${jobStatus.stats}</span>`;
        }

        if (jobStatus.status === 'completed') {
          break;
        }
        if (jobStatus.status === 'error') {
          throw new Error(jobStatus.error || "Export job encountered an error");
        }
      }

      // 3. Download the completed ZIP bundle
      const downloadUrl = `/api/export/download/${jobId}`;
      const dlResp = await fetch(downloadUrl);
      if (!dlResp.ok) {
        throw new Error(`Failed to download exported package (HTTP ${dlResp.status})`);
      }

      const blob = await dlResp.blob();
      const filename = jobStatus.zip_filename || `dataset_${droneName}.zip`;

      this.lastExportUrl = downloadUrl;
      this.lastExportFilename = filename;

      this.downloadBlob(blob, filename);
      this.showExportSuccessSummary(format, droneName, renderWidth, renderHeight, imgFormat, includeIq, includeVideo, exportSnr, filename);
      this.app.showNotification(`${formatName} package exported successfully!`, "success");
    } catch (e) {
      console.error("Export error:", e);
      alert("Error generating export package: " + e.message);
    } finally {
      this.app.showProgress(false);
    }
  }

  showExportSuccessSummary(format, droneName, width, height, imgFormat, includeIq, includeVideo, exportSnr, filename) {
    const successModal = document.getElementById('exportSuccessModal');
    if (!successModal) return;

    const formatEl = document.getElementById('successStatFormat');
    const resEl = document.getElementById('successStatResolution');
    const filesEl = document.getElementById('successStatFiles');
    const sizeEl = document.getElementById('successStatSize');
    const snrEl = document.getElementById('successStatSnr');
    const iqEl = document.getElementById('successStatIq');
    const zipNameEl = document.getElementById('successZipFilename');

    if (formatEl) formatEl.textContent = format.toUpperCase();
    if (resEl) resEl.textContent = `${width}×${height} ${imgFormat.toUpperCase()}`;
    if (filesEl) filesEl.textContent = `${this.currentManifest?.total_files || 'All'} Files`;
    if (sizeEl) sizeEl.textContent = `~${this.currentManifest?.estimated_size_mb || '15'} MB`;
    if (snrEl) snrEl.textContent = exportSnr !== null ? `${exportSnr.toFixed(1)} dB` : 'Original';
    if (iqEl) iqEl.textContent = includeIq ? 'Yes (.iq)' : (includeVideo ? 'Video (MP4)' : 'No');
    if (zipNameEl) zipNameEl.textContent = filename;

    successModal.classList.add('active');
  }

  triggerDownloadUrl(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async importCOCO(file) {
    if (!file) return;
    this.app.showLoader(true, `Importing annotations from ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const resp = await fetch('/api/import/coco', {
        method: 'POST',
        body: formData
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: "Network response was not ok" }));
        throw new Error(err.detail || "Failed to import COCO JSON");
      }

      const data = await resp.json();
      if (data.status === 'success') {
        this.app.annotationsCache = data.annotations || {};
        if (data.classes && this.app.classManager) {
          this.app.classManager.setClasses(data.classes);
        }
        this.app.loadChunkData(this.app.navigation.currentChunkId);
        this.app.navigation.renderFilmstrip();

        const count = data.stats?.total_imported ?? 0;
        const chunkCount = data.stats?.chunks_updated ?? 0;
        this.app.showNotification(`Successfully imported ${count} bounding boxes across ${chunkCount} chunks!`);
      } else {
        alert("Import failed: " + (data.message || "Unknown error"));
      }
    } catch (e) {
      console.error("COCO import error:", e);
      alert("Error importing COCO annotations: " + e.message);
    } finally {
      this.app.showLoader(false);
    }
  }
}
