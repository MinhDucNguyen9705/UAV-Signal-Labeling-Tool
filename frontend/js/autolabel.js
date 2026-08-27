/**
 * Auto-Label Manager: CFAR Detection & ONNX AI Model Inference with Agree/Decline Side-by-Side Review System
 */

export class AutoLabelManager {
  constructor(app) {
    this.app = app;
    this.proposalsByChunk = {}; // { [chunkId]: [proposalBoxes] }
    this.activeTab = 'cfar'; // 'cfar' | 'onnx'
    this.loadedOnnxModel = null;

    this.initElements();
    this.initEventListeners();
    this.checkActiveModel();
  }

  initElements() {
    this.modal = document.getElementById('autoLabelModal');
    this.openBtn = document.getElementById('openAutoLabelBtn');
    this.closeBtn = document.getElementById('closeAutoLabelBtn');
    this.cancelBtn = document.getElementById('cancelAutoLabelBtn');
    this.runBtn = document.getElementById('runAutoLabelBtn');

    this.tabCfarBtn = document.getElementById('tabCfarBtn');
    this.tabOnnxBtn = document.getElementById('tabOnnxBtn');
    this.tabCfarContent = document.getElementById('tabCfarContent');
    this.tabOnnxContent = document.getElementById('tabOnnxContent');

    // 2D CA-CFAR inputs
    this.cfarClass = document.getElementById('cfarClassSelect');
    this.cfarThresholdFactor = document.getElementById('cfarThresholdFactorInput');
    this.cfarThresholdFactorVal = document.getElementById('cfarThresholdFactorVal');
    this.cfarGuardRows = document.getElementById('cfarGuardRowsInput');
    this.cfarGuardCols = document.getElementById('cfarGuardColsInput');
    this.cfarTrainRows = document.getElementById('cfarTrainRowsInput');
    this.cfarTrainCols = document.getElementById('cfarTrainColsInput');
    this.cfarMorphKernel = document.getElementById('cfarMorphKernelInput');
    this.cfarMinArea = document.getElementById('cfarMinAreaInput');
    this.cfarMaxBoxesInput = document.getElementById('cfarMaxBoxesInput');

    // ONNX inputs
    this.onnxDropzone = document.getElementById('onnxDropzone');
    this.onnxFileInput = document.getElementById('onnxFileInput');
    this.onnxModelCard = document.getElementById('onnxModelCard');
    this.onnxModelName = document.getElementById('onnxModelName');
    this.onnxModelDetails = document.getElementById('onnxModelDetails');
    this.onnxConfInput = document.getElementById('onnxConfInput');
    this.onnxConfValue = document.getElementById('onnxConfValue');
    this.onnxIouInput = document.getElementById('onnxIouInput');
    this.onnxIouValue = document.getElementById('onnxIouValue');
    this.onnxDefaultClass = document.getElementById('onnxDefaultClassSelect');
    this.onnxClassMappingSection = document.getElementById('onnxClassMappingSection');
    this.onnxClassMappingList = document.getElementById('onnxClassMappingList');
    this.syncModelClassesBtn = document.getElementById('syncModelClassesBtn');
    this.onnxSingleClassFallback = document.getElementById('onnxSingleClassFallback');

    // Review bar elements
    this.reviewBar = document.getElementById('proposalReviewBar');
    this.proposalChunkBadge = document.getElementById('proposalChunkBadge');
    this.proposalCountBadge = document.getElementById('proposalCountBadge');
    this.proposalTotalBadge = document.getElementById('proposalTotalBadge');
    this.reviewPrevChunkBtn = document.getElementById('reviewPrevChunkBtn');
    this.reviewNextChunkBtn = document.getElementById('reviewNextChunkBtn');
    this.toggleComparisonBtn = document.getElementById('toggleComparisonViewBtn');
    this.comparisonToggleText = document.getElementById('comparisonToggleText');
    this.acceptChunkBtn = document.getElementById('acceptChunkProposalsBtn');
    this.rejectChunkBtn = document.getElementById('rejectChunkProposalsBtn');
    this.acceptAllBtn = document.getElementById('acceptAllProposalsBtn');
    this.rejectAllBtn = document.getElementById('rejectAllProposalsBtn');
  }

  initEventListeners() {
    if (this.openBtn) {
      this.openBtn.addEventListener('click', () => this.openModal());
    }
    if (this.closeBtn) {
      this.closeBtn.addEventListener('click', () => this.closeModal());
    }
    if (this.cancelBtn) {
      this.cancelBtn.addEventListener('click', () => this.closeModal());
    }

    if (this.syncModelClassesBtn) {
      this.syncModelClassesBtn.addEventListener('click', () => this.syncModelClassesToPalette());
    }

    // Tab switching
    if (this.tabCfarBtn && this.tabOnnxBtn) {
      this.tabCfarBtn.addEventListener('click', () => this.setTab('cfar'));
      this.tabOnnxBtn.addEventListener('click', () => this.setTab('onnx'));
    }

    // 2D CA-CFAR Threshold listener
    if (this.cfarThresholdFactor && this.cfarThresholdFactorVal) {
      this.cfarThresholdFactor.addEventListener('input', (e) => {
        this.cfarThresholdFactorVal.textContent = parseFloat(e.target.value).toFixed(2);
      });
    }

    // ONNX Threshold sliders
    if (this.onnxConfInput && this.onnxConfValue) {
      this.onnxConfInput.addEventListener('input', (e) => {
        this.onnxConfValue.textContent = parseFloat(e.target.value).toFixed(2);
      });
    }
    if (this.onnxIouInput && this.onnxIouValue) {
      this.onnxIouInput.addEventListener('input', (e) => {
        this.onnxIouValue.textContent = parseFloat(e.target.value).toFixed(2);
      });
    }

    // ONNX file upload dropzone
    if (this.onnxDropzone && this.onnxFileInput) {
      this.onnxDropzone.addEventListener('click', () => this.onnxFileInput.click());
      this.onnxFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
          this.uploadOnnxModel(e.target.files[0]);
        }
      });

      this.onnxDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        this.onnxDropzone.style.borderColor = 'var(--accent-orange)';
      });
      this.onnxDropzone.addEventListener('dragleave', () => {
        this.onnxDropzone.style.borderColor = 'var(--accent-cyan)';
      });
      this.onnxDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        this.onnxDropzone.style.borderColor = 'var(--accent-cyan)';
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          this.uploadOnnxModel(e.dataTransfer.files[0]);
        }
      });
    }

    // Run Auto-Label
    if (this.runBtn) {
      this.runBtn.addEventListener('click', () => this.runAutoLabel());
    }

    // Quick review chunk navigation
    if (this.reviewPrevChunkBtn) {
      this.reviewPrevChunkBtn.addEventListener('click', () => {
        if (this.app.navigation) this.app.navigation.previousChunk();
      });
    }
    if (this.reviewNextChunkBtn) {
      this.reviewNextChunkBtn.addEventListener('click', () => {
        if (this.app.navigation) this.app.navigation.nextChunk();
      });
    }

    // Side-by-Side View Mode toggle
    if (this.toggleComparisonBtn) {
      this.toggleComparisonBtn.addEventListener('click', () => this.toggleComparisonMode());
    }

    // Review bar action buttons
    if (this.acceptChunkBtn) {
      this.acceptChunkBtn.addEventListener('click', () => this.agreeCurrentChunk());
    }
    if (this.rejectChunkBtn) {
      this.rejectChunkBtn.addEventListener('click', () => this.declineCurrentChunk());
    }
    if (this.acceptAllBtn) {
      this.acceptAllBtn.addEventListener('click', () => this.agreeAllChunks());
    }
    if (this.rejectAllBtn) {
      this.rejectAllBtn.addEventListener('click', () => this.declineAllChunks());
    }
  }

  openModal() {
    this.populateClassDropdowns();
    this.renderClassMappingUI();
    if (this.modal) this.modal.classList.add('active');
  }

  closeModal() {
    if (this.modal) this.modal.classList.remove('active');
  }

  setTab(tab) {
    this.activeTab = tab;
    if (this.tabCfarBtn && this.tabOnnxBtn) {
      this.tabCfarBtn.classList.toggle('active', tab === 'cfar');
      this.tabOnnxBtn.classList.toggle('active', tab === 'onnx');
    }
    if (this.tabCfarContent && this.tabOnnxContent) {
      this.tabCfarContent.style.display = tab === 'cfar' ? 'block' : 'none';
      this.tabOnnxContent.style.display = tab === 'onnx' ? 'block' : 'none';
    }
    if (tab === 'onnx') {
      this.renderClassMappingUI();
    }
  }

  populateClassDropdowns() {
    const classes = this.app.classManager ? this.app.classManager.classes : [];
    if (this.cfarClass) {
      this.cfarClass.innerHTML = '';
      classes.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.id;
        opt.textContent = `${c.name} (${c.type_of_signal})`;
        this.cfarClass.appendChild(opt);
      });
    }
    if (this.onnxDefaultClass) {
      this.onnxDefaultClass.innerHTML = '';
      classes.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.id;
        opt.textContent = `${c.name} (${c.type_of_signal})`;
        this.onnxDefaultClass.appendChild(opt);
      });
    }
  }

  async checkActiveModel() {
    try {
      const resp = await fetch('/api/models/info');
      const data = await resp.json();
      if (data.loaded && data.model_info) {
        this.loadedOnnxModel = data.model_info;
        this.updateModelCard(data.model_info);
      }
    } catch (e) {
      console.warn("Could not check ONNX model info:", e);
    }
  }

  async uploadOnnxModel(file) {
    if (!file.name.toLowerCase().endsWith('.onnx')) {
      this.app.showNotification("Error: Please select a valid .onnx model file.", "danger");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    this.app.showLoader(true, `Uploading & Initializing ONNX Model '${file.name}'...`);
    try {
      const resp = await fetch('/api/models/upload', {
        method: 'POST',
        body: formData
      });
      const data = await resp.json();
      if (data.status === 'success' && data.model_info) {
        this.loadedOnnxModel = data.model_info;
        this.updateModelCard(data.model_info);
        this.app.showNotification(`Loaded ONNX model '${data.model_info.model_name}' (${data.model_info.num_classes} classes) successfully!`, "success");
      } else {
        this.app.showNotification(data.detail || "Failed to load ONNX model", "danger");
      }
    } catch (e) {
      this.app.showNotification(`Error uploading ONNX model: ${e.message}`, "danger");
    } finally {
      this.app.showLoader(false);
    }
  }

  updateModelCard(info) {
    if (!this.onnxModelCard) return;
    this.onnxModelCard.style.display = 'block';
    if (this.onnxModelName) this.onnxModelName.textContent = info.model_name;
    if (this.onnxModelDetails) {
      const clsList = info.class_names && info.class_names.length > 0 ? info.class_names : Array.from({length: info.num_classes}, (_, i) => `Class ${i}`);
      this.onnxModelDetails.textContent = `Input: ${info.input_width}x${info.input_height} | Classes (${info.num_classes}): ${clsList.join(', ')}`;
    }
    this.renderClassMappingUI();
  }

  renderClassMappingUI() {
    if (!this.onnxClassMappingList) return;
    this.onnxClassMappingList.innerHTML = '';

    if (!this.loadedOnnxModel) {
      if (this.onnxClassMappingSection) this.onnxClassMappingSection.style.display = 'none';
      return;
    }

    if (this.onnxClassMappingSection) this.onnxClassMappingSection.style.display = 'block';

    const numClasses = this.loadedOnnxModel.num_classes || 1;
    const modelClassNames = this.loadedOnnxModel.class_names || [];
    const appClasses = this.app.classManager ? this.app.classManager.classes : [];

    for (let i = 0; i < numClasses; i++) {
      const defaultName = modelClassNames[i] || `Class_${i}`;
      
      // Look for a matching app class by name, or match by index
      let matchedAppClass = appClasses.find(c => c.name.toLowerCase() === defaultName.toLowerCase()) ||
                            appClasses[i % appClasses.length];

      const row = document.createElement('div');
      row.className = 'onnx-class-mapping-row';
      row.id = `onnx_map_row_${i}`;
      row.style.cssText = 'display: flex; align-items: center; gap: 8px; background: rgba(255,255,255,0.03); padding: 8px 10px; border-radius: var(--radius-sm); border: 1px solid var(--border-color);';

      row.innerHTML = `
        <div style="font-family: var(--font-mono); font-size: 11px; font-weight: 700; color: var(--accent-cyan); width: 60px; flex-shrink: 0;">Class #${i}</div>
        <div style="flex: 1; display: flex; flex-direction: column; gap: 2px;">
          <span style="font-size: 10px; color: var(--text-muted);">Name for Class ${i}</span>
          <input type="text" class="form-input onnx-class-name-input form-input-mono" data-class-idx="${i}" value="${defaultName}" placeholder="e.g. WiFi_OFDM" style="padding: 4px 8px; font-size: 12px;">
        </div>
        <div style="flex: 1; display: flex; flex-direction: column; gap: 2px;">
          <span style="font-size: 10px; color: var(--text-muted);">Annotation Palette Class</span>
          <div style="display: flex; align-items: center; gap: 6px;">
            <div class="class-color-dot" id="onnx_color_dot_${i}" style="width: 12px; height: 12px; border-radius: 50%; background: ${matchedAppClass ? matchedAppClass.color : '#00e5ff'}; flex-shrink: 0;"></div>
            <select class="form-select onnx-class-select" data-class-idx="${i}" style="padding: 4px 8px; font-size: 12px; flex: 1;">
              ${appClasses.map(c => `<option value="${c.id}" ${matchedAppClass && matchedAppClass.id === c.id ? 'selected' : ''}>${c.name} (${c.type_of_signal})</option>`).join('')}
              <option value="__create_new__">+ Create New Class with Name Above</option>
            </select>
          </div>
        </div>
      `;

      const selectEl = row.querySelector('.onnx-class-select');
      const dotEl = row.querySelector(`#onnx_color_dot_${i}`);
      const inputEl = row.querySelector('.onnx-class-name-input');

      selectEl.addEventListener('change', (e) => {
        if (e.target.value === '__create_new__') {
          dotEl.style.background = 'var(--accent-orange)';
        } else {
          const selectedId = parseInt(e.target.value, 10);
          const found = appClasses.find(c => c.id === selectedId);
          if (found) {
            dotEl.style.background = found.color;
            if (inputEl.value.startsWith('Class_')) {
              inputEl.value = found.name;
            }
          }
        }
      });

      this.onnxClassMappingList.appendChild(row);
    }
  }

  async syncModelClassesToPalette() {
    if (!this.loadedOnnxModel || !this.app.classManager) return;
    const numClasses = this.loadedOnnxModel.num_classes || 1;
    let addedCount = 0;

    for (let i = 0; i < numClasses; i++) {
      const inputEl = document.querySelector(`.onnx-class-name-input[data-class-idx="${i}"]`);
      const customName = inputEl ? inputEl.value.trim() : `Class_${i}`;

      let existing = this.app.classManager.classes.find(c => c.name.toLowerCase() === customName.toLowerCase());
      if (!existing) {
        existing = this.app.classManager.addClass(customName, null, "Unknown", "Generic");
        addedCount++;
      }
    }

    this.populateClassDropdowns();
    this.renderClassMappingUI();

    if (addedCount > 0) {
      this.app.showNotification(`Synced and created ${addedCount} new classes in the annotation palette!`, "success");
    } else {
      this.app.showNotification("Model classes are already synced with the annotation palette.", "info");
    }
  }

  async runAutoLabel() {
    const scopeEl = document.querySelector('input[name="autoLabelScope"]:checked');
    const scope = scopeEl ? scopeEl.value : 'all';
    const currentChunkId = this.app.navigation ? this.app.navigation.currentChunkId : 0;

    this.closeModal();

    if (this.activeTab === 'cfar') {
      await this.runCFARAutoLabel(scope, currentChunkId);
    } else {
      await this.runONNXAutoLabel(scope, currentChunkId);
    }
  }

  async readStreamEvents(response, onProgress, onComplete) {
    if (!response.body) return;
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep trailing incomplete line

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const evt = JSON.parse(trimmed);
          if (evt.type === 'progress') {
            if (typeof onProgress === 'function') onProgress(evt);
          } else if (evt.type === 'complete') {
            if (typeof onComplete === 'function') onComplete(evt);
          }
        } catch (e) {
          console.warn("Could not parse NDJSON stream event:", trimmed, e);
        }
      }
    }

    if (buffer && buffer.trim()) {
      try {
        const evt = JSON.parse(buffer.trim());
        if (evt.type === 'complete') {
          if (typeof onComplete === 'function') onComplete(evt);
        } else if (evt.type === 'progress') {
          if (typeof onProgress === 'function') onProgress(evt);
        }
      } catch (e) {
        console.warn("Could not parse final buffer chunk:", buffer, e);
      }
    }
  }

  async runCFARAutoLabel(scope, currentChunkId) {
    const targetCatId = parseInt(this.cfarClass ? this.cfarClass.value : 1, 10) || 1;
    const body = {
      scope: scope,
      chunk_id: currentChunkId,
      target_category_id: targetCatId,
      threshold_factor: parseFloat(this.cfarThresholdFactor ? this.cfarThresholdFactor.value : 1.10) || 1.10,
      guard_rows: parseInt(this.cfarGuardRows ? this.cfarGuardRows.value : 12, 10) || 12,
      guard_cols: parseInt(this.cfarGuardCols ? this.cfarGuardCols.value : 25, 10) || 25,
      train_rows: parseInt(this.cfarTrainRows ? this.cfarTrainRows.value : 15, 10) || 15,
      train_cols: parseInt(this.cfarTrainCols ? this.cfarTrainCols.value : 15, 10) || 15,
      morph_kernel: parseInt(this.cfarMorphKernel ? this.cfarMorphKernel.value : 5, 10) || 5,
      min_area: parseInt(this.cfarMinArea ? this.cfarMinArea.value : 20, 10) || 20,
      max_boxes: parseInt(this.cfarMaxBoxesInput ? this.cfarMaxBoxesInput.value : 32, 10) || 32,
      stream: true
    };

    const totalChunksEstimate = scope === 'all' ? (this.app.navigation?.chunks?.length || 1) : 1;
    this.app.showProgress(true, {
      title: "Running 2D CA-CFAR Detection...",
      current: 0,
      total: totalChunksEstimate,
      stats: "Initializing parallel CFAR workers..."
    });

    try {
      const resp = await fetch('/api/autolabel/cfar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      let finalResult = null;
      await this.readStreamEvents(
        resp,
        (prog) => {
          this.app.showProgress(true, {
            title: "Running 2D CA-CFAR Detection...",
            current: prog.current,
            total: prog.total,
            stats: `${prog.total_proposals} candidate signals detected`
          });
        },
        (comp) => {
          finalResult = comp;
        }
      );

      if (finalResult && finalResult.status === 'success') {
        this.setProposals(finalResult.proposals_by_chunk);
        this.app.showNotification(`2D CA-CFAR detected ${finalResult.total_proposals} candidate signals! Review side-by-side below.`, "success");
      } else {
        this.app.showNotification(finalResult?.detail || "2D CA-CFAR Auto-Labeling completed.", "warning");
      }
    } catch (e) {
      this.app.showNotification(`CFAR error: ${e.message}`, "danger");
    } finally {
      this.app.showProgress(false);
    }
  }

  async runONNXAutoLabel(scope, currentChunkId) {
    if (!this.loadedOnnxModel) {
      this.app.showNotification("Please upload an ONNX model file (.onnx) first!", "warning");
      this.openModal();
      this.setTab('onnx');
      return;
    }

    const numClasses = this.loadedOnnxModel.num_classes || 1;
    const classMapping = {};
    let defaultCatId = 1;

    for (let i = 0; i < numClasses; i++) {
      const inputEl = document.querySelector(`.onnx-class-name-input[data-class-idx="${i}"]`);
      const selectEl = document.querySelector(`.onnx-class-select[data-class-idx="${i}"]`);
      const customName = inputEl ? inputEl.value.trim() : `Class_${i}`;
      let targetCatId = selectEl ? selectEl.value : '__create_new__';

      if (targetCatId === '__create_new__' || !targetCatId) {
        let existing = this.app.classManager.classes.find(c => c.name.toLowerCase() === customName.toLowerCase());
        if (!existing) {
          existing = this.app.classManager.addClass(customName, null, "Unknown", "Generic");
        }
        targetCatId = existing.id;
      } else {
        targetCatId = parseInt(targetCatId, 10);
      }

      classMapping[i] = targetCatId;
      if (i === 0) defaultCatId = targetCatId;
    }

    const body = {
      scope: scope,
      chunk_id: currentChunkId,
      conf_thresh: parseFloat(this.onnxConfInput.value) || 0.25,
      iou_thresh: parseFloat(this.onnxIouInput.value) || 0.45,
      default_category_id: defaultCatId,
      class_mapping: classMapping,
      stream: true
    };

    const totalChunksEstimate = scope === 'all' ? (this.app.navigation?.chunks?.length || 1) : 1;
    this.app.showProgress(true, {
      title: `Running ONNX Inference '${this.loadedOnnxModel.model_name}'...`,
      current: 0,
      total: totalChunksEstimate,
      stats: "Executing neural network detection..."
    });

    try {
      const resp = await fetch('/api/autolabel/onnx', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      let finalResult = null;
      await this.readStreamEvents(
        resp,
        (prog) => {
          this.app.showProgress(true, {
            title: `Running ONNX Inference '${this.loadedOnnxModel.model_name}'...`,
            current: prog.current,
            total: prog.total,
            stats: `${prog.total_proposals} bounding boxes predicted`
          });
        },
        (comp) => {
          finalResult = comp;
        }
      );

      if (finalResult && finalResult.status === 'success') {
        this.setProposals(finalResult.proposals_by_chunk);
        this.app.showNotification(`AI Model proposed ${finalResult.total_proposals} bounding boxes! Review side-by-side below.`, "success");
      } else {
        this.app.showNotification(finalResult?.detail || "ONNX Inference completed.", "warning");
      }
    } catch (e) {
      this.app.showNotification(`ONNX error: ${e.message}`, "danger");
    } finally {
      this.app.showProgress(false);
    }
  }

  setProposals(proposalsByChunk) {
    this.proposalsByChunk = proposalsByChunk || {};
    const totalProps = this.getTotalProposalCount();

    if (totalProps > 0) {
      // Automatically activate Side-by-Side Comparison View for intuitive review
      if (this.app.canvas) {
        this.app.canvas.setComparisonMode(true);
      }
      this.updateComparisonToggleUI(true);
    } else {
      if (this.app.canvas) {
        this.app.canvas.setComparisonMode(false);
      }
      this.updateComparisonToggleUI(false);
    }

    this.updateReviewUI();
    this.syncCanvasProposals();
    this.updateFilmstripProposalBadges();
  }

  toggleComparisonMode() {
    if (!this.app.canvas) return;
    const nextMode = !this.app.canvas.isComparisonMode;
    this.app.canvas.setComparisonMode(nextMode);
    this.updateComparisonToggleUI(nextMode);
  }

  updateComparisonToggleUI(isComparison) {
    if (this.toggleComparisonBtn) {
      this.toggleComparisonBtn.classList.toggle('active-toggle', isComparison);
      this.toggleComparisonBtn.classList.toggle('single-view', !isComparison);
    }
    if (this.comparisonToggleText) {
      this.comparisonToggleText.textContent = isComparison ? "Side-by-Side" : "Single View";
    }
  }

  getProposalsForChunk(chunkId) {
    return this.proposalsByChunk[String(chunkId)] || [];
  }

  getTotalProposalCount() {
    let total = 0;
    Object.values(this.proposalsByChunk).forEach(props => {
      total += props.length;
    });
    return total;
  }

  syncCanvasProposals() {
    const chunkId = this.app.navigation ? this.app.navigation.currentChunkId : 0;
    const proposals = this.getProposalsForChunk(chunkId);
    if (this.app.canvas) {
      this.app.canvas.setProposals(proposals);
    }
  }

  updateReviewUI() {
    const chunkId = this.app.navigation ? this.app.navigation.currentChunkId : 0;
    const currentProps = this.getProposalsForChunk(chunkId);
    const totalProps = this.getTotalProposalCount();

    if (!this.reviewBar) return;

    if (totalProps > 0) {
      this.reviewBar.style.display = 'flex';
      if (this.proposalChunkBadge) this.proposalChunkBadge.textContent = `#${chunkId + 1}`;
      if (this.proposalCountBadge) this.proposalCountBadge.textContent = `${currentProps.length}`;
      if (this.proposalTotalBadge) this.proposalTotalBadge.textContent = `${totalProps}`;

      if (this.reviewPrevChunkBtn) {
        this.reviewPrevChunkBtn.disabled = chunkId <= 0;
      }
      if (this.reviewNextChunkBtn && this.app.navigation?.chunks) {
        this.reviewNextChunkBtn.disabled = chunkId >= this.app.navigation.chunks.length - 1;
      }
    } else {
      this.reviewBar.style.display = 'none';
      if (this.app.canvas && this.app.canvas.isComparisonMode) {
        this.app.canvas.setComparisonMode(false);
      }
    }
  }

  updateFilmstripProposalBadges() {
    if (!this.app.navigation || !this.app.navigation.chunks) return;
    this.app.navigation.chunks.forEach(chunk => {
      const props = this.getProposalsForChunk(chunk.id);
      const badge = document.getElementById(`badge_${chunk.id}`);
      if (badge) {
        if (props.length > 0) {
          badge.textContent = `✨${props.length}`;
          badge.style.display = 'inline-block';
          badge.style.background = 'linear-gradient(135deg, #a855f7 0%, #00e5ff 100%)';
          badge.style.color = '#fff';
        } else {
          const normalCount = this.app.getChunkBoxCount ? this.app.getChunkBoxCount(chunk.id) : 0;
          badge.textContent = normalCount > 0 ? normalCount : '';
          badge.style.display = normalCount > 0 ? 'inline-block' : 'none';
          badge.style.background = '';
          badge.style.color = '';
        }
      }
    });
  }

  agreeCurrentChunk() {
    const chunkId = this.app.navigation ? this.app.navigation.currentChunkId : 0;
    const proposals = this.getProposalsForChunk(chunkId);
    if (proposals.length === 0) {
      this.app.showNotification("No candidate proposals on this chunk to agree.", "info");
      return;
    }

    // Convert proposals to confirmed annotations
    const currentBoxes = this.app.canvas ? [...this.app.canvas.boxes] : [];
    proposals.forEach(prop => {
      const confirmedBox = {
        ...prop,
        isProposal: false,
        id: `box_${Date.now()}_${Math.floor(Math.random() * 10000)}`
      };
      currentBoxes.push(confirmedBox);
    });

    // Update app state
    this.app.annotationsCache[String(chunkId)] = currentBoxes;
    if (this.app.canvas) {
      this.app.canvas.boxes = currentBoxes;
      this.app.canvas.setProposals([]);
    }

    // Clear proposals for this chunk
    delete this.proposalsByChunk[String(chunkId)];

    this.app.saveCurrentChunkAnnotations();
    this.app.onChunkLoaded(currentBoxes);
    this.updateReviewUI();
    this.updateFilmstripProposalBadges();
    this.app.showNotification(`Agreed & added ${proposals.length} annotations to Chunk #${chunkId + 1}!`, "success");

    // Check if remaining proposals exist on another chunk and navigate if appropriate
    const remainingKeys = Object.keys(this.proposalsByChunk).filter(k => this.proposalsByChunk[k]?.length > 0);
    if (remainingKeys.length > 0) {
      const nextChunkWithProps = remainingKeys.map(k => parseInt(k, 10)).find(id => id > chunkId) || parseInt(remainingKeys[0], 10);
      if (nextChunkWithProps !== undefined && nextChunkWithProps !== chunkId && this.app.navigation) {
        this.app.navigation.goToChunk(nextChunkWithProps);
      }
    } else {
      if (this.app.canvas) {
        this.app.canvas.setComparisonMode(false);
      }
    }
  }

  declineCurrentChunk() {
    const chunkId = this.app.navigation ? this.app.navigation.currentChunkId : 0;
    const count = this.getProposalsForChunk(chunkId).length;
    if (count === 0) return;

    delete this.proposalsByChunk[String(chunkId)];
    if (this.app.canvas) {
      this.app.canvas.setProposals([]);
    }
    this.updateReviewUI();
    this.updateFilmstripProposalBadges();
    this.app.showNotification(`Declined ${count} proposals on Chunk #${chunkId + 1}.`, "info");

    const remainingKeys = Object.keys(this.proposalsByChunk).filter(k => this.proposalsByChunk[k]?.length > 0);
    if (remainingKeys.length > 0) {
      const nextChunkWithProps = remainingKeys.map(k => parseInt(k, 10)).find(id => id > chunkId) || parseInt(remainingKeys[0], 10);
      if (nextChunkWithProps !== undefined && nextChunkWithProps !== chunkId && this.app.navigation) {
        this.app.navigation.goToChunk(nextChunkWithProps);
      }
    } else {
      if (this.app.canvas) {
        this.app.canvas.setComparisonMode(false);
      }
    }
  }

  async agreeAllChunks() {
    let totalCommitted = 0;
    const savePromises = [];

    for (const [cIdStr, proposals] of Object.entries(this.proposalsByChunk)) {
      if (!proposals || proposals.length === 0) continue;
      const cId = parseInt(cIdStr, 10);
      let existingBoxes = this.app.annotationsCache[cIdStr] || [];

      proposals.forEach(prop => {
        const confirmedBox = {
          ...prop,
          isProposal: false,
          id: `box_${Date.now()}_${Math.floor(Math.random() * 10000)}`
        };
        existingBoxes.push(confirmedBox);
        totalCommitted++;
      });

      this.app.annotationsCache[cIdStr] = existingBoxes;

      savePromises.push(
        fetch('/api/annotations', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chunk_id: cId,
            annotations: existingBoxes
          })
        })
      );
    }

    await Promise.all(savePromises);

    this.proposalsByChunk = {};
    if (this.app.canvas) {
      const currentCId = this.app.navigation ? this.app.navigation.currentChunkId : 0;
      this.app.canvas.boxes = this.app.annotationsCache[String(currentCId)] || [];
      this.app.canvas.setProposals([]);
      this.app.canvas.setComparisonMode(false);
      this.app.onChunkLoaded(this.app.canvas.boxes);
    }

    this.updateReviewUI();
    this.updateFilmstripProposalBadges();
    this.app.showNotification(`Agreed & committed all ${totalCommitted} proposals across all chunks!`, "success");
  }

  declineAllChunks() {
    const count = this.getTotalProposalCount();
    this.proposalsByChunk = {};
    if (this.app.canvas) {
      this.app.canvas.setProposals([]);
      this.app.canvas.setComparisonMode(false);
    }
    this.updateReviewUI();
    this.updateFilmstripProposalBadges();
    this.app.showNotification(`Declined all ${count} proposals.`, "info");
  }
}
