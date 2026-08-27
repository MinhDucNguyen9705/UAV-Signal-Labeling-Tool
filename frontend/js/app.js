import { SpectrogramCanvas } from './canvas.js?v=3.3';
import { ClassManager } from './classes.js?v=3.3';
import { NavigationManager } from './navigation.js?v=3.3';
import { ExportManager } from './export.js?v=3.3';
import { BDWCalculator } from './bdw_calc.js?v=3.3';
import { AutoLabelManager } from './autolabel.js?v=3.3';

class RFCVATApp {
  constructor() {
    this.sessionSummary = null;
    this.annotationsCache = {}; // { chunkId: [box, ...] }
    this.preloadedImages = new Map(); // { chunkId: HTMLImageElement }
    this.activeTab = 'bdw'; // 'classes', 'objects', 'bdw', 'stft'

    this.classManager = new ClassManager(this);
    this.navigation = new NavigationManager(this);
    this.exportManager = new ExportManager(this);
    this.canvas = new SpectrogramCanvas(this);
    this.autoLabelManager = new AutoLabelManager(this);

    this.initDOM();
    this.initHotkeys();
    this.loadInitialSession();
  }

  initDOM() {
    // Tool buttons (Select, Draw, Pan, Measure)
    document.querySelectorAll('.tool-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const mode = btn.dataset.tool;
        if (mode) this.setTool(mode);
      });
    });

    // Zoom buttons
    document.getElementById('zoomInBtn')?.addEventListener('click', () => this.canvas.zoomBy(1.2));
    document.getElementById('zoomOutBtn')?.addEventListener('click', () => this.canvas.zoomBy(0.8));
    document.getElementById('zoomFitBtn')?.addEventListener('click', () => this.canvas.fitToScreen());
    document.getElementById('zoomResetBtn')?.addEventListener('click', () => this.canvas.resetZoom());

    // Sidebar Tabs
    document.querySelectorAll('.sidebar-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        this.switchTab(tab);
      });
    });

    // Modals
    this.setupModals();

    // BDW Inspector Inputs
    this.setupBDWInspector();

    // Spectrogram Settings Inputs
    this.setupSTFTSettings();

    // Class Add Button in Sidebar
    document.getElementById('addNewClassBtn')?.addEventListener('click', () => {
      const name = prompt("Enter new class name:", "Radar_Signal");
      if (name) {
        this.classManager.addClass(name);
      }
    });

    // Export Dropdown Toggle
    const exportDropdown = document.getElementById('exportDropdown');
    const exportToggleBtn = document.getElementById('exportToggleBtn');
    exportToggleBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      exportDropdown.classList.toggle('active');
    });

    document.addEventListener('click', () => {
      exportDropdown?.classList.remove('active');
    });
  }

  setTool(toolName) {
    document.querySelectorAll('.tool-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tool === toolName);
    });
    this.canvas.setMode(toolName);
  }

  switchTab(tabName) {
    this.activeTab = tabName;
    document.querySelectorAll('.sidebar-tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === tabName);
    });
    document.querySelectorAll('.sidebar-tab-content').forEach(c => {
      c.classList.toggle('active', c.id === `tabContent_${tabName}`);
    });
  }

  setupBDWInspector() {
    const inputs = ['bdwToa', 'bdwTod', 'bdwFc', 'bdwBw', 'bdwSnr', 'bdwSignalType', 'bdwProtocol'];
    inputs.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('change', () => this.applyBDWInspectorChanges());
      }
    });

    document.getElementById('deleteActiveBoxBtn')?.addEventListener('click', () => {
      this.canvas.deleteSelectedBox();
    });
  }

  applyBDWInspectorChanges() {
    if (!this.canvas.selectedBoxId) return;
    const box = this.canvas.boxes.find(b => b.id === this.canvas.selectedBoxId);
    if (!box) return;

    this.canvas.pushHistory();

    const toa = parseFloat(document.getElementById('bdwToa').value);
    const tod = parseFloat(document.getElementById('bdwTod').value);
    const fc = parseFloat(document.getElementById('bdwFc').value);
    const bw = parseFloat(document.getElementById('bdwBw').value);
    const snr = parseFloat(document.getElementById('bdwSnr').value);
    const sigType = document.getElementById('bdwSignalType').value;
    const protocol = document.getElementById('bdwProtocol').value;

    const bdw = {
      ...box.bdw,
      toa_us: toa,
      tod_us: tod,
      pw_us: Math.max(0, tod - toa),
      fc_mhz: fc,
      bw_mhz: bw,
      snr_db: snr,
      type_of_signal: sigType,
      protocol: protocol
    };

    // Convert physical params back to pixels
    const [px, py, pw, ph] = BDWCalculator.bdwToPixels(bdw, this.canvas.imgWidth, this.canvas.imgHeight, this.canvas.chunkMeta);
    box.x = px;
    box.y = py;
    box.width = pw;
    box.height = ph;
    box.bdw = bdw;

    this.onBoxUpdated(box);
    this.canvas.redraw();
    this.onAnnotationsChanged(this.canvas.boxes);
  }

  setupSTFTSettings() {
    const colormapSelect = document.getElementById('stftColormap');
    const nfftSelect = document.getElementById('stftNfft');
    const chunkDurInput = document.getElementById('stftChunkDuration');
    const applyBtn = document.getElementById('applyStftBtn');

    applyBtn?.addEventListener('click', async () => {
      this.showLoader(true, "Updating STFT & Spectrogram Parameters...");
      try {
        const resp = await fetch('/api/session/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            colormap: colormapSelect.value,
            nfft: parseInt(nfftSelect.value, 10),
            chunk_duration_ms: parseFloat(chunkDurInput.value),
            overlap_duration_ms: parseFloat(document.getElementById('stftOverlapDuration')?.value || 10.0)
          })
        });
        const data = await resp.json();
        if (data.status === 'success') {
          this.sessionSummary = data.summary;
          this.navigation.setChunks(data.chunks);
          this.navigation.renderFilmstrip();
          this.preloadAllChunks();
          this.loadChunkData(this.navigation.currentChunkId);
          this.showNotification("STFT parameters updated & filmstrip synced!");
        }
      } catch (e) {
        alert("Failed to update STFT configuration: " + e.message);
      } finally {
        this.showLoader(false);
      }
    });
  }

  setupModals() {
    // Upload Modal
    const uploadModal = document.getElementById('uploadModal');
    const openUploadBtn = document.getElementById('openUploadBtn');
    const closeUploadBtn = document.getElementById('closeUploadBtn');
    const uploadDropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('fileInput');
    const submitUploadBtn = document.getElementById('submitUploadBtn');

    openUploadBtn?.addEventListener('click', () => uploadModal.classList.add('active'));
    closeUploadBtn?.addEventListener('click', () => uploadModal.classList.remove('active'));

    uploadDropzone?.addEventListener('click', () => fileInput.click());
    fileInput?.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        uploadDropzone.querySelector('.dropzone-text').textContent = e.target.files[0].name;
      }
    });

    submitUploadBtn?.addEventListener('click', async () => {
      if (!fileInput.files.length) {
        alert("Please select a file.");
        return;
      }
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('fs', document.getElementById('uploadFs').value);
      const fcMhz = parseFloat(document.getElementById('uploadCenterFreq')?.value || '2400.0');
      formData.append('center_freq', (isNaN(fcMhz) ? 2400e6 : fcMhz * 1e6).toString());
      formData.append('iq_format', document.getElementById('uploadFormat').value);
      formData.append('chunk_duration_ms', document.getElementById('uploadChunkDur').value);
      formData.append('overlap_duration_ms', document.getElementById('uploadOverlapDur')?.value || 10.0);
      formData.append('nfft', document.getElementById('uploadNfft')?.value || 1024);
      formData.append('colormap', document.getElementById('uploadColormap')?.value || 'turbo');
      formData.append('window', document.getElementById('uploadWindow')?.value || 'hann');

      this.showLoader(true, "Uploading & Processing RF Data...");
      uploadModal.classList.remove('active');

      try {
        const resp = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.status === 'success') {
          this.sessionSummary = data.summary;
          this.annotationsCache = {};
          this.navigation.setChunks(data.chunks);
          this.navigation.renderFilmstrip();
          this.preloadAllChunks();
          this.updateDatasetBadge();
          this.loadChunkData(0);
          this.showNotification("Uploaded and loaded successfully!");
        } else {
          alert("Upload failed: " + (data.detail || "Unknown error"));
        }
      } catch (e) {
        alert("Error during upload: " + e.message);
      } finally {
        this.showLoader(false);
      }
    });

    // Sample Generator Modal
    const sampleModal = document.getElementById('sampleModal');
    const openSampleBtn = document.getElementById('openSampleBtn');
    const closeSampleBtn = document.getElementById('closeSampleBtn');
    const generateSampleSubmitBtn = document.getElementById('generateSampleSubmitBtn');

    openSampleBtn?.addEventListener('click', () => sampleModal.classList.add('active'));
    closeSampleBtn?.addEventListener('click', () => sampleModal.classList.remove('active'));

    generateSampleSubmitBtn?.addEventListener('click', async () => {
      const formData = new FormData();
      formData.append('fs', document.getElementById('sampleFs').value);
      formData.append('iq_format', document.getElementById('sampleFormat').value);
      formData.append('duration_ms', document.getElementById('sampleDuration').value);
      formData.append('output_format', document.getElementById('sampleOutFormat').value);
      formData.append('chunk_duration_ms', document.getElementById('sampleChunkDur').value);
      formData.append('overlap_duration_ms', document.getElementById('sampleOverlapDur')?.value || 10.0);

      this.showLoader(true, "Generating Synthetic RF Dataset...");
      sampleModal.classList.remove('active');

      try {
        const resp = await fetch('/api/generate_sample', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.status === 'success') {
          this.sessionSummary = data.summary;
          this.annotationsCache = {};
          this.navigation.setChunks(data.chunks);
          this.navigation.renderFilmstrip();
          this.preloadAllChunks();
          this.updateDatasetBadge();
          this.loadChunkData(0);
          this.showNotification("Synthetic dataset generated & loaded!");
        }
      } catch (e) {
        alert("Error generating sample: " + e.message);
      } finally {
        this.showLoader(false);
      }
    });

    // Waterfall Video Modal
    const waterfallModal = document.getElementById('waterfallModal');
    const openWaterfallBtn = document.getElementById('openWaterfallBtn');
    const closeWaterfallBtn = document.getElementById('closeWaterfallBtn');
    const waterfallPlayer = document.getElementById('waterfallVideoPlayer');
    const waterfallSpinner = document.getElementById('waterfallLoadingSpinner');
    const waterfallFpsSelect = document.getElementById('waterfallFpsSelect');
    const reRenderWaterfallBtn = document.getElementById('reRenderWaterfallBtn');
    const downloadWaterfallMp4Btn = document.getElementById('downloadWaterfallMp4Btn');
    const exportWaterfallDropdownBtn = document.getElementById('exportWaterfallDropdownBtn');

    const loadWaterfallVideo = async () => {
      const fps = waterfallFpsSelect ? waterfallFpsSelect.value : 10;
      if (waterfallSpinner) waterfallSpinner.style.display = 'block';
      if (downloadWaterfallMp4Btn) {
        downloadWaterfallMp4Btn.href = `/api/export/waterfall_video?fps=${fps}&download=true`;
      }
      try {
        const resp = await fetch(`/api/export/waterfall_video?fps=${fps}&download=false`);
        const data = await resp.json();
        if (data.status === 'success' && waterfallPlayer) {
          waterfallPlayer.src = data.stream_url;
          waterfallPlayer.load();
          waterfallPlayer.play().catch(() => {});
        }
      } catch (e) {
        console.error("Error rendering waterfall video:", e);
      } finally {
        if (waterfallSpinner) waterfallSpinner.style.display = 'none';
      }
    };

    openWaterfallBtn?.addEventListener('click', () => {
      waterfallModal?.classList.add('active');
      loadWaterfallVideo();
    });
    exportWaterfallDropdownBtn?.addEventListener('click', () => {
      waterfallModal?.classList.add('active');
      loadWaterfallVideo();
    });
    closeWaterfallBtn?.addEventListener('click', () => {
      waterfallModal?.classList.remove('active');
      if (waterfallPlayer) waterfallPlayer.pause();
    });
    waterfallFpsSelect?.addEventListener('change', () => loadWaterfallVideo());
    reRenderWaterfallBtn?.addEventListener('click', () => loadWaterfallVideo());

    // Hotkeys Modal
    const hotkeysModal = document.getElementById('hotkeysModal');
    const openHotkeysBtn = document.getElementById('openHotkeysBtn');
    const closeHotkeysBtn = document.getElementById('closeHotkeysBtn');
    openHotkeysBtn?.addEventListener('click', () => hotkeysModal.classList.add('active'));
    closeHotkeysBtn?.addEventListener('click', () => hotkeysModal.classList.remove('active'));
  }

  initHotkeys() {
    window.addEventListener('keydown', (e) => {
      // Don't trigger hotkeys if typing in inputs
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        return;
      }

      if (e.key === 'Escape') {
        this.canvas.cancelCornerDrawing();
        document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active'));
      } else if (e.key === 'y' || e.key === 'Y') {
        // If candidate proposals are active, Y accepts current chunk proposals
        if (this.autoLabelManager && this.autoLabelManager.getProposalsForChunk(this.navigation.currentChunkId).length > 0 && !e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          this.autoLabelManager.agreeCurrentChunk();
        }
      } else if (e.key === 'n' || e.key === 'N') {
        // If candidate proposals are active, N declines current chunk proposals; else switch to Draw tool
        if (this.autoLabelManager && this.autoLabelManager.getProposalsForChunk(this.navigation.currentChunkId).length > 0 && !e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          this.autoLabelManager.declineCurrentChunk();
        } else {
          e.preventDefault();
          this.setTool('draw');
        }
      } else if (e.key === 'v' || e.key === 'V') {
        e.preventDefault();
        this.setTool('select');
      } else if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        this.setTool('pan');
      } else if (e.key === 'm' || e.key === 'M') {
        e.preventDefault();
        this.setTool('measure');
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        this.canvas.deleteHoveredOrSelectedBox();
      } else if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowLeft' || e.key === '[') {
        e.preventDefault();
        this.navigation.previousChunk();
      } else if (e.key === 'f' || e.key === 'F' || e.key === 'ArrowRight' || e.key === ']') {
        e.preventDefault();
        this.navigation.nextChunk();
      } else if (e.key === '0') {
        e.preventDefault();
        this.canvas.fitToScreen();
      } else if (e.key === '+' || e.key === '=') {
        e.preventDefault();
        this.canvas.zoomBy(1.2);
      } else if (e.key === '-' || e.key === '_') {
        e.preventDefault();
        this.canvas.zoomBy(0.8);
      } else if (e.key === '?') {
        e.preventDefault();
        document.getElementById('hotkeysModal')?.classList.toggle('active');
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        this.canvas.undo();
      } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.shiftKey && e.key === 'Z'))) {
        e.preventDefault();
        this.canvas.redo();
      } else if ((e.key === 'c' || e.key === 'C') && !e.ctrlKey && !e.metaKey) {
        if (this.autoLabelManager && this.autoLabelManager.getTotalProposalCount() > 0) {
          e.preventDefault();
          this.autoLabelManager.toggleComparisonMode();
        }
      } else if (e.key >= '1' && e.key <= '9') {
        const num = parseInt(e.key, 10);
        if (num <= this.classManager.classes.length) {
          this.classManager.setActiveClass(this.classManager.classes[num - 1].id);
        }
      }
    });
  }

  async loadInitialSession() {
    try {
      const resp = await fetch('/api/session');
      const data = await resp.json();

      this.sessionSummary = data.summary;
      this.classManager.setClasses(data.classes || []);
      this.navigation.setChunks(data.chunks || []);

      // Fetch all annotations
      const annResp = await fetch('/api/annotations');
      const annData = await annResp.json();
      this.annotationsCache = annData.annotations || {};

      this.updateDatasetBadge();
      this.updateSTFTForm();
      this.preloadAllChunks();

      if (data.chunks && data.chunks.length > 0) {
        this.loadChunkData(0);
      }
    } catch (e) {
      console.error("Error loading initial session:", e);
    }
  }

  preloadAllChunks(currentChunkId = 0) {
    this.preloadedImages.clear();
    const timestamp = Date.now();
    const isBatched = this.sessionSummary && this.sessionSummary.render_mode === 'batched';
    const batchSize = (this.sessionSummary && this.sessionSummary.batch_size) || 10;

    let targetChunks = this.navigation.chunks;
    if (isBatched) {
      const bStart = Math.floor(currentChunkId / batchSize) * batchSize;
      const bEnd = Math.min(bStart + batchSize, this.navigation.chunks.length);
      targetChunks = this.navigation.chunks.slice(bStart, bEnd);
    }

    targetChunks.forEach(chunk => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        this.preloadedImages.set(chunk.id, img);
      };
      img.src = `/api/chunks/${chunk.id}/spectrogram?t=${timestamp}`;
    });
  }

  updateDatasetBadge() {
    if (!this.sessionSummary) return;
    const s = this.sessionSummary;
    const nameEl = document.getElementById('datasetFileName');
    const fsEl = document.getElementById('datasetFs');
    const durEl = document.getElementById('datasetDuration');

    const modeText = s.render_mode === 'batched' ? ` | Mode: Batched (${s.batch_size}/batch)` : ` | Mode: Eager`;

    if (nameEl) nameEl.textContent = s.source_filename;
    if (fsEl) fsEl.textContent = `${s.fs_mhz} MHz (${s.source_format})${modeText}`;
    if (durEl) durEl.textContent = `${s.total_duration_ms.toFixed(2)} ms (${s.num_chunks} chunks, overlap: ${s.overlap_duration_ms || 10} ms)`;
  }

  updateSTFTForm() {
    if (!this.sessionSummary?.stft_config) return;
    const cfg = this.sessionSummary.stft_config;
    const cm = document.getElementById('stftColormap');
    const nfft = document.getElementById('stftNfft');
    const chunkDur = document.getElementById('stftChunkDuration');
    const overlapDur = document.getElementById('stftOverlapDuration');

    if (cm) cm.value = cfg.colormap;
    if (nfft) nfft.value = cfg.nfft;
    if (chunkDur) chunkDur.value = this.sessionSummary.chunk_duration_ms || 30.0;
    if (overlapDur) overlapDur.value = this.sessionSummary.overlap_duration_ms || 10.0;
  }

  async loadChunkData(chunkId) {
    const chunk = this.navigation.chunks[chunkId];
    if (!chunk) return;

    const isBatched = this.sessionSummary && this.sessionSummary.render_mode === 'batched';
    if (isBatched && !this.preloadedImages.has(chunkId)) {
      this.preloadAllChunks(chunkId);
    }

    // Load chunk annotations from cache or fetch
    let boxes = this.annotationsCache[String(chunkId)];
    if (!boxes) {
      try {
        const resp = await fetch(`/api/chunks/${chunkId}`);
        const data = await resp.json();
        boxes = data.annotations || [];
        this.annotationsCache[String(chunkId)] = boxes;
      } catch (e) {
        boxes = [];
      }
    }

    // Instant switch if image already preloaded in memory (0ms lag!)
    if (this.preloadedImages.has(chunkId)) {
      const preloadedImg = this.preloadedImages.get(chunkId);
      if (preloadedImg.complete && preloadedImg.naturalWidth > 0) {
        this.canvas.loadPreloadedImage(preloadedImg, chunk, boxes);
        return;
      }
    }

    const imgUrl = `/api/chunks/${chunkId}/spectrogram?t=${Date.now()}`;
    this.canvas.loadImage(imgUrl, chunk, boxes);
  }

  onChunkLoaded(boxes) {
    this.renderObjectsList(boxes);
    this.navigation.updateChunkBadge(this.navigation.currentChunkId, boxes.length);
    if (this.autoLabelManager) {
      this.autoLabelManager.syncCanvasProposals();
      this.autoLabelManager.updateReviewUI();
    }
  }

  onAnnotationsChanged(boxes) {
    const chunkId = this.navigation.currentChunkId;
    this.annotationsCache[String(chunkId)] = boxes;
    this.renderObjectsList(boxes);
    this.navigation.updateChunkBadge(chunkId, boxes.length);
    this.saveCurrentChunkAnnotations();
  }

  async saveCurrentChunkAnnotations() {
    const chunkId = this.navigation.currentChunkId;
    const boxes = this.canvas.boxes || [];
    try {
      await fetch('/api/annotations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chunk_id: chunkId,
          annotations: boxes
        })
      });
    } catch (e) {
      console.error("Failed to save annotations:", e);
    }
  }

  getChunkBoxCount(chunkId) {
    const boxes = this.annotationsCache[String(chunkId)];
    return boxes ? boxes.length : 0;
  }

  onBoxSelected(box) {
    this.updateBDWInspector(box);
    this.highlightObjectRow(box.id);
    this.switchTab('bdw');
  }

  onBoxDeselected() {
    this.clearBDWInspector();
    this.highlightObjectRow(null);
  }

  onBoxUpdated(box) {
    if (this.canvas.selectedBoxId === box.id) {
      this.updateBDWInspector(box);
    }
    this.renderObjectsList(this.canvas.boxes);
  }

  updateBDWInspector(box) {
    const bdw = box.bdw || {};
    const toaEl = document.getElementById('bdwToa');
    const todEl = document.getElementById('bdwTod');
    const pwEl = document.getElementById('bdwPw');
    const fcEl = document.getElementById('bdwFc');
    const bwEl = document.getElementById('bdwBw');
    const snrEl = document.getElementById('bdwSnr');
    const sigTypeEl = document.getElementById('bdwSignalType');
    const protoEl = document.getElementById('bdwProtocol');
    const delBtn = document.getElementById('deleteActiveBoxBtn');

    if (toaEl) toaEl.value = bdw.toa_us ?? '';
    if (todEl) todEl.value = bdw.tod_us ?? '';
    if (pwEl) pwEl.textContent = `${bdw.pw_us ?? 0} µs`;
    if (fcEl) fcEl.value = bdw.fc_mhz ?? '';
    if (bwEl) bwEl.value = bdw.bw_mhz ?? '';
    if (snrEl) snrEl.value = bdw.snr_db ?? '';
    if (sigTypeEl) sigTypeEl.value = bdw.type_of_signal || '';
    if (protoEl) protoEl.value = bdw.protocol || '';

    // Card Stats
    document.getElementById('statToa').textContent = `${bdw.toa_us ?? 0} µs`;
    document.getElementById('statPw').textContent = `${bdw.pw_us ?? 0} µs`;
    document.getElementById('statFc').textContent = `${bdw.fc_mhz ?? 0} MHz`;
    document.getElementById('statBw').textContent = `${bdw.bw_mhz ?? 0} MHz`;
    document.getElementById('statSnr').textContent = `${bdw.snr_db ?? 0} dB`;

    if (delBtn) delBtn.disabled = false;
  }

  clearBDWInspector() {
    ['bdwToa', 'bdwTod', 'bdwFc', 'bdwBw', 'bdwSnr', 'bdwSignalType', 'bdwProtocol'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.value = '';
    });
    ['statToa', 'statPw', 'statFc', 'statBw', 'statSnr'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = '--';
    });
    const delBtn = document.getElementById('deleteActiveBoxBtn');
    if (delBtn) delBtn.disabled = true;
  }

  renderObjectsList(boxes) {
    const listEl = document.getElementById('objectsListContainer');
    if (!listEl) return;
    listEl.innerHTML = '';

    const countEl = document.getElementById('objectsCountBadge');
    if (countEl) countEl.textContent = `${boxes.length} items`;

    if (boxes.length === 0) {
      listEl.innerHTML = `<div style="text-align: center; color: var(--text-muted); font-size: 11px; padding: 20px 0;">No annotations on this chunk.<br>Press 'N' to draw a bounding box.</div>`;
      return;
    }

    boxes.forEach(box => {
      const cls = this.classManager.classes.find(c => c.id === box.category_id) || {
        name: "Unknown",
        color: "#00e5ff"
      };

      const row = document.createElement('div');
      row.className = `object-item ${box.id === this.canvas.selectedBoxId ? 'selected' : ''}`;
      row.id = `obj_row_${box.id}`;

      const pw = box.bdw ? `${box.bdw.pw_us}µs` : '';
      const bw = box.bdw ? `${box.bdw.bw_mhz}MHz` : '';
      const snr = box.bdw ? `${box.bdw.snr_db}dB` : '';

      row.innerHTML = `
        <div class="object-color-pill" style="background-color: ${cls.color}"></div>
        <div class="object-info">
          <div class="object-title" style="color: ${cls.color}">${cls.name}</div>
          <div class="object-subtext">PW: ${pw} | BW: ${bw} | SNR: ${snr}</div>
        </div>
        <div class="object-actions">
          <button class="icon-btn lock-btn ${box.isLocked ? 'active' : ''}" title="${box.isLocked ? 'Unlock' : 'Lock'}">
            <i class="fas fa-${box.isLocked ? 'lock' : 'lock-open'}"></i>
          </button>
          <button class="icon-btn hide-btn ${box.isHidden ? 'active' : ''}" title="${box.isHidden ? 'Show' : 'Hide'}">
            <i class="fas fa-${box.isHidden ? 'eye-slash' : 'eye'}"></i>
          </button>
          <button class="icon-btn del-btn" title="Delete">
            <i class="fas fa-trash"></i>
          </button>
        </div>
      `;

      row.addEventListener('click', (e) => {
        if (!e.target.closest('.object-actions')) {
          this.canvas.selectBox(box.id);
        }
      });

      row.querySelector('.lock-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        box.isLocked = !box.isLocked;
        this.canvas.redraw();
        this.renderObjectsList(this.canvas.boxes);
      });

      row.querySelector('.hide-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        box.isHidden = !box.isHidden;
        this.canvas.redraw();
        this.renderObjectsList(this.canvas.boxes);
      });

      row.querySelector('.del-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        this.canvas.selectBox(box.id);
        this.canvas.deleteSelectedBox();
      });

      listEl.appendChild(row);
    });
  }

  highlightObjectRow(boxId) {
    document.querySelectorAll('.object-item').forEach(row => {
      row.classList.toggle('selected', row.id === `obj_row_${boxId}`);
    });
  }

  showLoader(show, text = "Processing...") {
    const loader = document.getElementById('globalLoader');
    const textEl = document.getElementById('globalLoaderText');
    const progContainer = document.getElementById('globalProgressBarContainer');
    if (loader) {
      loader.classList.toggle('active', show);
      if (textEl && text) textEl.textContent = text;
      if (!show && progContainer) {
        progContainer.style.display = 'none';
      }
    }
  }

  showProgress(visible, { title = "Running Auto-Labeling...", current = 0, total = 1, stats = "" } = {}) {
    const loader = document.getElementById('globalLoader');
    const titleEl = document.getElementById('globalLoaderText');
    const progContainer = document.getElementById('globalProgressBarContainer');
    const progDetail = document.getElementById('globalProgressDetail');
    const progPercent = document.getElementById('globalProgressPercent');
    const progFill = document.getElementById('globalProgressBarFill');
    const progStats = document.getElementById('globalProgressStats');

    if (!loader) return;

    if (visible) {
      loader.classList.add('active');
      if (titleEl && title) titleEl.textContent = title;
      if (progContainer) {
        progContainer.style.display = 'block';
        const safeTotal = Math.max(1, total);
        const pct = Math.min(100, Math.round((current / safeTotal) * 100));
        if (progDetail) progDetail.textContent = `Chunk ${current} / ${safeTotal}`;
        if (progPercent) progPercent.textContent = `${pct}%`;
        if (progFill) progFill.style.width = `${pct}%`;
        if (progStats) {
          progStats.innerHTML = `<i class="fas fa-bolt" style="color: var(--accent-cyan);"></i> <span>${stats || 'Analyzing RF signals...'}</span>`;
        }
      }
    } else {
      loader.classList.remove('active');
      if (progContainer) progContainer.style.display = 'none';
    }
  }

  showNotification(msg) {
    const notif = document.getElementById('appNotification');
    if (!notif) return;
    notif.textContent = msg;
    notif.style.display = 'block';
    setTimeout(() => {
      notif.style.display = 'none';
    }, 3000);
  }
}

// Instantiate on load
window.addEventListener('DOMContentLoaded', () => {
  window.rfApp = new RFCVATApp();
});
