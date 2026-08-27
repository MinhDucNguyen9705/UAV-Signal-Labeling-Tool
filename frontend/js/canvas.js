import { BDWCalculator } from './bdw_calc.js?v=3.3';

export class SpectrogramCanvas {
  constructor(app) {
    this.app = app;

    // Single Canvas Elements
    this.container = document.getElementById('canvasContainer');
    this.singleWrapper = document.getElementById('singleCanvasWrapper');
    this.canvas = document.getElementById('spectrogramCanvas');
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;

    // Side-by-Side Comparison Split Elements
    this.splitWrapper = document.getElementById('comparisonSplitWrapper');
    this.beforePane = document.getElementById('comparisonPaneBefore');
    this.afterPane = document.getElementById('comparisonPaneAfter');
    this.canvasBefore = document.getElementById('spectrogramCanvasBefore');
    this.ctxBefore = this.canvasBefore ? this.canvasBefore.getContext('2d') : null;
    this.canvasAfter = document.getElementById('spectrogramCanvasAfter');
    this.ctxAfter = this.canvasAfter ? this.canvasAfter.getContext('2d') : null;
    this.beforeCountBadge = document.getElementById('beforeCountBadge');
    this.afterCountBadge = document.getElementById('afterCountBadge');
    this.isComparisonMode = false;

    // Dual Axis Rulers
    this.rulerTop = document.getElementById('rulerTop');
    this.rulerLeft = document.getElementById('rulerLeft');
    this.rulerTopCtx = this.rulerTop?.getContext('2d');
    this.rulerLeftCtx = this.rulerLeft?.getContext('2d');

    // Live HUD Elements
    this.hudTime = document.getElementById('hudTime');
    this.hudFreq = document.getElementById('hudFreq');

    // Spectrogram Image State
    this.image = new Image();
    this.isLoaded = false;
    this.imgWidth = 1024;
    this.imgHeight = 512;
    this.chunkMeta = null;

    // View Transformation (Pan & Zoom) - Synchronized across both panes
    this.scale = 1.0;
    this.offsetX = 0;
    this.offsetY = 0;
    this.minScale = 0.2;
    this.maxScale = 20.0;

    // Interaction State
    this.currentMode = 'draw'; // 'select', 'draw', 'pan', 'measure'
    this.isMouseDown = false;
    this.isPanning = false;
    this.panStartX = 0;
    this.panStartY = 0;
    this.activeEventCanvas = null;

    // Drawing Box State (2-Corner Click Mode)
    this.cornerClick1 = null; // { x, y }
    this.isWaitingSecondCorner = false;
    this.hoverImgCoords = null;
    this.drawStartX = 0;
    this.drawStartY = 0;
    this.drawCurrentX = 0;
    this.drawCurrentY = 0;
    this.isDrawingBox = false;

    // Resizing & Dragging State
    this.boxes = []; // [{ id, category_id, x, y, width, height, isLocked, isHidden, bdw }]
    this.proposals = []; // Auto-label candidate proposals
    this.selectedBoxId = null;
    this.hoveredBoxId = null;
    this.activeHandle = null; // 'nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'
    this.isDraggingBox = false;
    this.dragOffset = { x: 0, y: 0 };
    this.originalBoxState = null;

    // Measure Tool State
    this.measureStart = null;
    this.measureEnd = null;

    // History for Undo / Redo
    this.undoStack = [];
    this.redoStack = [];
    this.hasInitialFit = false;

    this.initElements();
    this.initEvents();
    this.resizeCanvas();
  }

  initElements() {
    if (!this.container) this.container = document.getElementById('canvasContainer');
    if (!this.singleWrapper) this.singleWrapper = document.getElementById('singleCanvasWrapper');
    if (!this.splitWrapper) this.splitWrapper = document.getElementById('comparisonSplitWrapper');
    if (!this.canvas) this.canvas = document.getElementById('spectrogramCanvas');
    if (!this.ctx && this.canvas) this.ctx = this.canvas.getContext('2d');
    if (!this.beforePane) this.beforePane = document.getElementById('comparisonPaneBefore');
    if (!this.afterPane) this.afterPane = document.getElementById('comparisonPaneAfter');
    if (!this.canvasBefore) this.canvasBefore = document.getElementById('spectrogramCanvasBefore');
    if (!this.ctxBefore && this.canvasBefore) this.ctxBefore = this.canvasBefore.getContext('2d');
    if (!this.canvasAfter) this.canvasAfter = document.getElementById('spectrogramCanvasAfter');
    if (!this.ctxAfter && this.canvasAfter) this.ctxAfter = this.canvasAfter.getContext('2d');
    if (!this.beforeCountBadge) this.beforeCountBadge = document.getElementById('beforeCountBadge');
    if (!this.afterCountBadge) this.afterCountBadge = document.getElementById('afterCountBadge');
  }

  initEvents() {
    window.addEventListener('resize', () => this.resizeCanvas());

    this.initElements();
    const canvases = [this.canvas, this.canvasBefore, this.canvasAfter].filter(Boolean);

    canvases.forEach(c => {
      c.addEventListener('mousedown', (e) => {
        this.activeEventCanvas = c;
        this.onMouseDown(e, c);
      });
      c.addEventListener('mouseleave', () => {
        this.hoveredBoxId = null;
        this.hoverImgCoords = null;
        this.updateCursor();
        this.redraw();
      });
      c.addEventListener('wheel', (e) => this.onWheel(e, c), { passive: false });
      c.addEventListener('contextmenu', (e) => e.preventDefault());
    });

    window.addEventListener('mousemove', (e) => this.onMouseMove(e));
    window.addEventListener('mouseup', (e) => this.onMouseUp(e));

    if (this.container) {
      this.container.addEventListener('wheel', (e) => {
        if (!canvases.includes(e.target)) {
          this.onWheel(e, this.isComparisonMode ? (this.canvasAfter || this.canvasBefore) : this.canvas);
        }
      }, { passive: false });
    }
  }

  setComparisonMode(enabled) {
    this.isComparisonMode = Boolean(enabled);
    this.initElements();

    if (this.container) {
      this.container.classList.toggle('comparison-mode-active', this.isComparisonMode);
    }
    if (document.body) {
      document.body.classList.toggle('comparison-active', this.isComparisonMode);
    }

    if (this.splitWrapper) {
      this.splitWrapper.style.display = this.isComparisonMode ? 'flex' : 'none';
    }
    if (this.singleWrapper) {
      this.singleWrapper.style.display = this.isComparisonMode ? 'none' : 'block';
    }

    // Force synchronous resize & fit, then refine on next animation frame
    this.resizeCanvas();
    this.fitToScreen();
    this.redraw();

    requestAnimationFrame(() => {
      this.resizeCanvas();
      this.fitToScreen();
      this.redraw();
    });
  }

  resizeCanvas() {
    if (!this.container) return;
    this.initElements();
    const dpr = window.devicePixelRatio || 1;

    const containerRect = this.container.getBoundingClientRect();
    const contW = Math.max(10, containerRect.width || 1024);
    const contH = Math.max(10, containerRect.height || 512);

    // 1. Size Single Canvas
    if (this.canvas && this.ctx) {
      this.canvas.width = Math.max(1, contW * dpr);
      this.canvas.height = Math.max(1, contH * dpr);
      this.canvas.style.width = `${contW}px`;
      this.canvas.style.height = `${contH}px`;
      this.ctx.setTransform(1, 0, 0, 1, 0, 0);
      this.ctx.scale(dpr, dpr);
    }

    // 2. Size Comparison Canvases (Left & Right)
    let paneW = Math.floor((contW - 14) / 2);
    let paneH = contH;
    if (this.beforePane) {
      const bRect = this.beforePane.getBoundingClientRect();
      if (bRect.width > 20) paneW = bRect.width;
      if (bRect.height > 20) paneH = bRect.height;
    }

    if (this.canvasBefore && this.ctxBefore) {
      this.canvasBefore.width = Math.max(1, paneW * dpr);
      this.canvasBefore.height = Math.max(1, paneH * dpr);
      this.canvasBefore.style.width = `${paneW}px`;
      this.canvasBefore.style.height = `${paneH}px`;
      this.ctxBefore.setTransform(1, 0, 0, 1, 0, 0);
      this.ctxBefore.scale(dpr, dpr);
    }

    if (this.canvasAfter && this.ctxAfter) {
      this.canvasAfter.width = Math.max(1, paneW * dpr);
      this.canvasAfter.height = Math.max(1, paneH * dpr);
      this.canvasAfter.style.width = `${paneW}px`;
      this.canvasAfter.style.height = `${paneH}px`;
      this.ctxAfter.setTransform(1, 0, 0, 1, 0, 0);
      this.ctxAfter.scale(dpr, dpr);
    }

    // 3. Size Dual Axis Rulers
    if (this.rulerTop && this.rulerTopCtx) {
      this.rulerTop.width = Math.max(1, contW * dpr);
      this.rulerTop.height = 24 * dpr;
      this.rulerTop.style.width = `${contW}px`;
      this.rulerTop.style.height = `24px`;
      this.rulerTopCtx.setTransform(1, 0, 0, 1, 0, 0);
      this.rulerTopCtx.scale(dpr, dpr);
    }

    if (this.rulerLeft && this.rulerLeftCtx) {
      this.rulerLeft.width = 54 * dpr;
      this.rulerLeft.height = Math.max(1, contH * dpr);
      this.rulerLeft.style.width = `54px`;
      this.rulerLeft.style.height = `${contH}px`;
      this.rulerLeftCtx.setTransform(1, 0, 0, 1, 0, 0);
      this.rulerLeftCtx.scale(dpr, dpr);
    }

    this.redraw();
  }

  setMode(mode) {
    if (mode !== 'draw') {
      this.cancelCornerDrawing();
    }
    this.currentMode = mode;
    this.updateCursor();
    this.redraw();
  }

  cancelCornerDrawing() {
    this.cornerClick1 = null;
    this.isWaitingSecondCorner = false;
    this.isDrawingBox = false;
    this.redraw();
  }

  updateCursor(handle = null) {
    const targetCanvases = [this.canvas, this.canvasBefore, this.canvasAfter].filter(Boolean);

    if (handle) {
      const handleCursors = {
        nw: 'nwse-resize', se: 'nwse-resize',
        ne: 'nesw-resize', sw: 'nesw-resize',
        n: 'ns-resize', s: 'ns-resize',
        e: 'ew-resize', w: 'ew-resize'
      };
      const cur = handleCursors[handle] || 'default';
      targetCanvases.forEach(c => c.style.cursor = cur);
      return;
    }

    let cur = 'default';
    if (this.isPanning) {
      cur = 'grabbing';
    } else if (this.currentMode === 'pan') {
      cur = 'grab';
    } else if (this.currentMode === 'draw') {
      if (!this.isWaitingSecondCorner && this.hoverImgCoords) {
        const box = this.getBoxAt(this.hoverImgCoords.x, this.hoverImgCoords.y);
        if (box) {
          cur = 'pointer';
          targetCanvases.forEach(c => c.style.cursor = cur);
          return;
        }
      }
      cur = 'crosshair';
    } else if (this.currentMode === 'measure') {
      cur = 'cell';
    } else if (this.currentMode === 'select') {
      if (this.hoverImgCoords) {
        const box = this.getBoxAt(this.hoverImgCoords.x, this.hoverImgCoords.y);
        if (box) {
          cur = 'move';
          targetCanvases.forEach(c => c.style.cursor = cur);
          return;
        }
      }
      cur = 'grab';
    }

    targetCanvases.forEach(c => c.style.cursor = cur);
  }

  loadPreloadedImage(imageEl, chunkMeta, existingBoxes = []) {
    this.image = imageEl;
    this.isLoaded = true;
    this.chunkMeta = chunkMeta;
    this.imgWidth = this.image.naturalWidth || 1024;
    this.imgHeight = this.image.naturalHeight || 512;
    this.boxes = existingBoxes || [];
    this.selectedBoxId = null;
    this.undoStack = [];
    this.redoStack = [];

    this.fitToScreen();
    this.showLoader(false);
    this.app.onChunkLoaded(this.boxes);
  }

  loadImage(imageUrl, chunkMeta, existingBoxes = []) {
    if (!this.image) this.isLoaded = false;
    this.chunkMeta = chunkMeta;
    this.showLoader(true);

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      this.image = img;
      this.isLoaded = true;
      this.imgWidth = this.image.naturalWidth || 1024;
      this.imgHeight = this.image.naturalHeight || 512;
      this.boxes = existingBoxes || [];
      this.selectedBoxId = null;
      this.undoStack = [];
      this.redoStack = [];

      this.fitToScreen();
      this.showLoader(false);
      this.app.onChunkLoaded(this.boxes);
    };
    img.onerror = () => {
      this.showLoader(false);
    };
    img.src = imageUrl;
  }

  setProposals(proposals) {
    this.proposals = proposals || [];
    this.redraw();
  }

  showLoader(show) {
    const loader = document.getElementById('canvasLoader');
    if (loader) {
      loader.classList.toggle('active', show);
    }
  }

  getActiveViewportWidth() {
    if (this.isComparisonMode) {
      if (this.beforePane) {
        const w = this.beforePane.getBoundingClientRect().width;
        if (w > 20) return w;
      }
      if (this.container) {
        const w = this.container.getBoundingClientRect().width;
        if (w > 20) return Math.floor((w - 14) / 2);
      }
    }
    return this.container ? (this.container.getBoundingClientRect().width || 1024) : 1024;
  }

  getActiveViewportHeight() {
    if (this.isComparisonMode) {
      if (this.beforePane) {
        const h = this.beforePane.getBoundingClientRect().height;
        if (h > 20) return h;
      }
    }
    return this.container ? (this.container.getBoundingClientRect().height || 512) : 512;
  }

  fitToScreen() {
    const viewW = this.getActiveViewportWidth();
    const viewH = this.getActiveViewportHeight();
    if (!viewW || !viewH || !this.imgWidth || !this.imgHeight) return;

    const pad = 18;
    const availW = Math.max(10, viewW - pad * 2);
    const availH = Math.max(10, viewH - pad * 2);

    const scaleX = availW / this.imgWidth;
    const scaleY = availH / this.imgHeight;
    this.scale = Math.min(scaleX, scaleY, 1.5);

    this.offsetX = Math.round((viewW - this.imgWidth * this.scale) / 2);
    this.offsetY = Math.round((viewH - this.imgHeight * this.scale) / 2);
    this.redraw();
  }

  resetZoom() {
    const viewW = this.getActiveViewportWidth();
    const viewH = this.getActiveViewportHeight();

    this.scale = 1.0;
    this.offsetX = Math.round((viewW - this.imgWidth) / 2);
    this.offsetY = Math.round((viewH - this.imgHeight) / 2);
    this.redraw();
  }

  zoom(factor, centerX = null, centerY = null) {
    const newScale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * factor));
    if (newScale === this.scale) return;

    const viewW = this.getActiveViewportWidth();
    const viewH = this.getActiveViewportHeight();

    const cx = centerX !== null ? centerX : viewW / 2;
    const cy = centerY !== null ? centerY : viewH / 2;

    this.offsetX = cx - (cx - this.offsetX) * (newScale / this.scale);
    this.offsetY = cy - (cy - this.offsetY) * (newScale / this.scale);
    this.scale = newScale;

    this.redraw();
  }

  zoomBy(factor, centerX = null, centerY = null) {
    this.zoom(factor, centerX, centerY);
  }

  // Coordinate Conversion: Screen -> Image Pixels
  screenToImageCoords(screenX, screenY) {
    const imgX = (screenX - this.offsetX) / this.scale;
    const imgY = (screenY - this.offsetY) / this.scale;
    return {
      x: Math.max(0, Math.min(this.imgWidth, imgX)),
      y: Math.max(0, Math.min(this.imgHeight, imgY)),
      rawX: imgX,
      rawY: imgY
    };
  }

  // Coordinate Conversion: Image Pixels -> Screen
  imageToScreenCoords(imgX, imgY) {
    return {
      x: imgX * this.scale + this.offsetX,
      y: imgY * this.scale + this.offsetY
    };
  }

  getTargetCanvas(e) {
    if (e && e.target && e.target.tagName === 'CANVAS') {
      return e.target;
    }
    if (this.activeEventCanvas) {
      return this.activeEventCanvas;
    }
    if (this.isComparisonMode) {
      return this.canvasAfter || this.canvasBefore || this.canvas;
    }
    return this.canvas;
  }

  onMouseDown(e, targetCanvas) {
    if (!this.isLoaded && !this.image) return;
    const canvasEl = targetCanvas || this.getTargetCanvas(e);
    if (!canvasEl) return;

    const rect = canvasEl.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    const imgCoords = this.screenToImageCoords(screenX, screenY);

    this.isMouseDown = true;
    this.activeEventCanvas = canvasEl;

    // Pan mode (Middle click, Spacebar, or Pan tool)
    if (e.button === 1 || e.spaceKey || this.currentMode === 'pan') {
      this.isPanning = true;
      this.panStartX = e.clientX - this.offsetX;
      this.panStartY = e.clientY - this.offsetY;
      this.updateCursor();
      return;
    }

    if (e.button !== 0) return; // Only left click for drawing/selection

    // Measure tool
    if (this.currentMode === 'measure') {
      this.measureStart = imgCoords;
      this.measureEnd = imgCoords;
      this.redraw();
      return;
    }

    // If clicking inside an existing box while in DRAW mode -> switch to SELECT/MOVE mode
    const clickedBox = this.getBoxAt(imgCoords.x, imgCoords.y);
    if (clickedBox && this.currentMode === 'draw' && !this.isWaitingSecondCorner) {
      this.cancelCornerDrawing();
      this.app.setTool('select');
      this.selectBox(clickedBox.id);
      if (!clickedBox.isLocked) {
        this.isDraggingBox = true;
        this.dragOffset = {
          x: imgCoords.x - clickedBox.x,
          y: imgCoords.y - clickedBox.y
        };
        this.originalBoxState = { ...clickedBox };
        this.pushHistory();
      }
      this.redraw();
      return;
    }

    // 2-Corner Click Mode for adding bounding box
    if (this.currentMode === 'draw') {
      if (!this.isWaitingSecondCorner) {
        // First click: Top-Left Corner
        this.cornerClick1 = { x: imgCoords.x, y: imgCoords.y };
        this.isWaitingSecondCorner = true;
        this.selectedBoxId = null;
        this.redraw();
        return;
      } else {
        // Second click: Bottom-Right Corner
        const p1 = this.cornerClick1;
        const p2 = { x: imgCoords.x, y: imgCoords.y };
        const x1 = Math.min(p1.x, p2.x);
        const y1 = Math.min(p1.y, p2.y);
        const x2 = Math.max(p1.x, p2.x);
        const y2 = Math.max(p1.y, p2.y);
        const width = x2 - x1;
        const height = y2 - y1;

        this.cornerClick1 = null;
        this.isWaitingSecondCorner = false;

        if (width >= 4 && height >= 4) {
          this.pushHistory();
          const activeClass = this.app.classManager.getActiveClass();
          const boxId = `box_${Date.now()}_${Math.floor(Math.random() * 1000)}`;

          const newBox = {
            id: boxId,
            category_id: activeClass.id,
            x: Math.round(x1),
            y: Math.round(y1),
            width: Math.round(width),
            height: Math.round(height),
            isLocked: false,
            isHidden: false,
            bdw: null
          };

          newBox.bdw = BDWCalculator.pixelsToBDW(
            [newBox.x, newBox.y, newBox.width, newBox.height],
            this.imgWidth,
            this.imgHeight,
            this.chunkMeta,
            activeClass
          );

          this.boxes.push(newBox);
          this.selectBox(newBox.id);
          this.fetchServerSNR(newBox);
          this.app.onAnnotationsChanged(this.boxes);
        }
        this.redraw();
        return;
      }
    }

    // Check if clicking a handle on selected box
    if (this.selectedBoxId) {
      const selBox = this.boxes.find(b => b.id === this.selectedBoxId);
      if (selBox && !selBox.isLocked) {
        const handle = this.getHandleAt(selBox, screenX, screenY);
        if (handle) {
          this.activeHandle = handle;
          this.originalBoxState = { ...selBox };
          this.pushHistory();
          return;
        }
      }
    }

    // Check if clicking inside an existing box
    if (clickedBox && (this.currentMode === 'select' || e.shiftKey)) {
      this.selectBox(clickedBox.id);
      if (!clickedBox.isLocked) {
        this.isDraggingBox = true;
        this.dragOffset = {
          x: imgCoords.x - clickedBox.x,
          y: imgCoords.y - clickedBox.y
        };
        this.originalBoxState = { ...clickedBox };
        this.pushHistory();
      }
      this.redraw();
      return;
    }

    // Click empty space: deselect and start pan
    if (this.currentMode === 'select') {
      if (this.selectedBoxId) {
        this.selectedBoxId = null;
        this.app.onBoxDeselected();
      }
      this.isPanning = true;
      this.panStartX = e.clientX - this.offsetX;
      this.panStartY = e.clientY - this.offsetY;
      this.updateCursor();
      this.redraw();
    }
  }

  onMouseMove(e) {
    if (!this.isLoaded && !this.image) return;

    // Panning
    if (this.isPanning) {
      this.offsetX = e.clientX - this.panStartX;
      this.offsetY = e.clientY - this.panStartY;
      this.redraw();
      return;
    }

    const canvasEl = this.getTargetCanvas(e);
    if (!canvasEl) return;
    const rect = canvasEl.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    const isInsideCanvas = (
      e.clientX >= rect.left && e.clientX <= rect.right &&
      e.clientY >= rect.top && e.clientY <= rect.bottom
    );

    const imgCoords = this.screenToImageCoords(screenX, screenY);
    this.hoverImgCoords = isInsideCanvas ? imgCoords : null;

    // Update Live HUD
    if (this.chunkMeta && isInsideCanvas) {
      const coord = BDWCalculator.pixelToCoord(imgCoords.x, imgCoords.y, this.imgWidth, this.imgHeight, this.chunkMeta);
      if (this.hudTime) this.hudTime.textContent = `${coord.timeUs.toFixed(3)} µs`;
      if (this.hudFreq) this.hudFreq.textContent = `${coord.freqMhz.toFixed(3)} MHz`;
    }

    // Hover box detection
    let hovered = null;
    if (isInsideCanvas) {
      hovered = this.getBoxAt(imgCoords.x, imgCoords.y);
    }
    const newHoveredId = hovered ? hovered.id : null;
    const hoverChanged = newHoveredId !== this.hoveredBoxId;
    this.hoveredBoxId = newHoveredId;

    // 2-Corner Click Mode Active Hover
    if (this.currentMode === 'draw' && this.isWaitingSecondCorner) {
      this.redraw();
      return;
    }

    // Measure tool active drag
    if (this.currentMode === 'measure' && this.isMouseDown && this.measureStart) {
      this.measureEnd = imgCoords;
      this.redraw();
      return;
    }

    // Handle cursor for resize handles
    if (this.selectedBoxId && !this.isMouseDown && isInsideCanvas) {
      const selBox = this.boxes.find(b => b.id === this.selectedBoxId);
      if (selBox && !selBox.isLocked) {
        const handle = this.getHandleAt(selBox, screenX, screenY);
        if (handle) {
          this.updateCursor(handle);
          if (hoverChanged) this.redraw();
          return;
        }
      }
    }

    if (!this.isMouseDown) {
      this.updateCursor();
    }

    // Resizing selected box
    if (this.activeHandle && this.selectedBoxId) {
      const selBox = this.boxes.find(b => b.id === this.selectedBoxId);
      if (selBox) {
        this.resizeBoxWithHandle(selBox, this.activeHandle, imgCoords.x, imgCoords.y);
        this.updateBDW(selBox);
        this.redraw();
        return;
      }
    }

    // Dragging box
    if (this.isDraggingBox && this.selectedBoxId) {
      const selBox = this.boxes.find(b => b.id === this.selectedBoxId);
      if (selBox) {
        selBox.x = Math.max(0, Math.min(this.imgWidth - selBox.width, imgCoords.x - this.dragOffset.x));
        selBox.y = Math.max(0, Math.min(this.imgHeight - selBox.height, imgCoords.y - this.dragOffset.y));
        this.updateBDW(selBox);
        this.redraw();
        return;
      }
    }

    if (hoverChanged) {
      this.redraw();
    }
  }

  onMouseUp(e) {
    if (this.isPanning) {
      this.isPanning = false;
      this.updateCursor();
    }

    if (this.activeHandle || this.isDraggingBox) {
      this.activeHandle = null;
      this.isDraggingBox = false;
      const selBox = this.boxes.find(b => b.id === this.selectedBoxId);
      if (selBox) {
        this.fetchServerSNR(selBox);
        this.app.onAnnotationsChanged(this.boxes);
      }
    }

    this.isMouseDown = false;
    this.activeEventCanvas = null;
    this.redraw();
  }

  onWheel(e, targetCanvas) {
    e.preventDefault();
    if (!this.isLoaded && !this.image) return;

    const canvasEl = targetCanvas || this.getTargetCanvas(e);
    if (!canvasEl) return;
    const rect = canvasEl.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    let delta = e.deltaY;
    if (e.deltaMode === 1) {
      delta *= 20;
    } else if (e.deltaMode === 2) {
      delta *= 500;
    }

    const factor = Math.exp(-delta * 0.0015);
    this.zoom(factor, mouseX, mouseY);
  }

  selectBox(boxId) {
    this.selectedBoxId = boxId;
    const box = this.boxes.find(b => b.id === boxId);
    if (box) {
      this.app.onBoxSelected(box);
    } else {
      this.app.onBoxDeselected();
    }
    this.redraw();
  }

  deleteBoxById(boxId) {
    if (!boxId) return;

    // Check if proposal
    const propIdx = this.proposals.findIndex(p => p.id === boxId);
    if (propIdx !== -1) {
      this.proposals.splice(propIdx, 1);
      if (this.app.autoLabelManager) {
        const chunkId = this.app.navigation ? this.app.navigation.currentChunkId : 0;
        this.app.autoLabelManager.proposalsByChunk[String(chunkId)] = this.proposals;
        this.app.autoLabelManager.updateReviewUI();
        this.app.autoLabelManager.updateFilmstripProposalBadges();
      }
      if (this.hoveredBoxId === boxId) this.hoveredBoxId = null;
      this.redraw();
      return;
    }

    const box = this.boxes.find(b => b.id === boxId);
    if (!box) return;

    this.pushHistory();
    this.boxes = this.boxes.filter(b => b.id !== boxId);

    if (this.selectedBoxId === boxId) {
      this.selectedBoxId = null;
      this.app.onBoxDeselected();
    }
    if (this.hoveredBoxId === boxId) {
      this.hoveredBoxId = null;
    }

    this.app.onAnnotationsChanged(this.boxes);
    this.redraw();
  }

  deleteHoveredOrSelectedBox() {
    let targetBoxId = null;
    if (this.hoverImgCoords) {
      const hoveredBox = this.getBoxAt(this.hoverImgCoords.x, this.hoverImgCoords.y);
      if (hoveredBox) {
        targetBoxId = hoveredBox.id;
      }
    }

    if (!targetBoxId && this.selectedBoxId) {
      targetBoxId = this.selectedBoxId;
    }

    if (!targetBoxId) return;
    this.deleteBoxById(targetBoxId);
  }

  deleteSelectedBox() {
    if (this.selectedBoxId) {
      this.deleteBoxById(this.selectedBoxId);
    }
  }

  updateBDW(box) {
    const cat = this.app.classManager.classes.find(c => c.id === box.category_id);
    box.bdw = BDWCalculator.pixelsToBDW(
      [box.x, box.y, box.width, box.height],
      this.imgWidth,
      this.imgHeight,
      this.chunkMeta,
      cat
    );
    this.app.onBoxUpdated(box);
  }

  async fetchServerSNR(box) {
    try {
      const cat = this.app.classManager.classes.find(c => c.id === box.category_id);
      const sigType = cat?.type_of_signal || box.bdw?.type_of_signal || "Unknown";
      const proto = cat?.protocol || box.bdw?.protocol || "Generic";

      const resp = await fetch('/api/annotations/calculate_bdw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chunk_id: this.app.navigation.currentChunkId,
          bbox: [box.x, box.y, box.width, box.height],
          img_width: this.imgWidth,
          img_height: this.imgHeight,
          category_id: box.category_id,
          signal_type: sigType,
          protocol: proto
        })
      });
      const data = await resp.json();
      if (data.status === 'success' && data.bdw) {
        box.bdw = {
          ...data.bdw,
          type_of_signal: sigType,
          protocol: proto
        };
        this.app.onBoxUpdated(box);
        this.redraw();
      }
    } catch (e) {
      console.warn("Failed to fetch server SNR:", e);
    }
  }

  getBoxAt(x, y) {
    // Reverse search so topmost is picked
    for (let i = this.boxes.length - 1; i >= 0; i--) {
      const b = this.boxes[i];
      if (!b.isHidden && x >= b.x && x <= b.x + b.width && y >= b.y && y <= b.y + b.height) {
        return b;
      }
    }
    // Search candidate proposals
    for (let i = this.proposals.length - 1; i >= 0; i--) {
      const p = this.proposals[i];
      if (x >= p.x && x <= p.x + p.width && y >= p.y && y <= p.y + p.height) {
        return p;
      }
    }
    return null;
  }

  getHandleAt(box, screenX, screenY) {
    const handleSize = 8;
    const handles = this.getHandleCoordinates(box);

    for (const [handleKey, pt] of Object.entries(handles)) {
      if (Math.abs(screenX - pt.x) <= handleSize && Math.abs(screenY - pt.y) <= handleSize) {
        return handleKey;
      }
    }
    return null;
  }

  getHandleCoordinates(box) {
    const p1 = this.imageToScreenCoords(box.x, box.y);
    const p2 = this.imageToScreenCoords(box.x + box.width, box.y + box.height);
    const midX = (p1.x + p2.x) / 2;
    const midY = (p1.y + p2.y) / 2;

    return {
      nw: { x: p1.x, y: p1.y },
      n:  { x: midX, y: p1.y },
      ne: { x: p2.x, y: p1.y },
      e:  { x: p2.x, y: midY },
      se: { x: p2.x, y: p2.y },
      s:  { x: midX, y: p2.y },
      sw: { x: p1.x, y: p2.y },
      w:  { x: p1.x, y: midY }
    };
  }

  resizeBoxWithHandle(box, handle, imgX, imgY) {
    const minSize = 4;
    let { x, y, width, height } = box;
    let right = x + width;
    let bottom = y + height;

    if (handle.includes('e')) right = Math.max(x + minSize, imgX);
    if (handle.includes('w')) x = Math.min(right - minSize, imgX);
    if (handle.includes('s')) bottom = Math.max(y + minSize, imgY);
    if (handle.includes('n')) y = Math.min(bottom - minSize, imgY);

    box.x = Math.max(0, Math.round(x));
    box.y = Math.max(0, Math.round(y));
    box.width = Math.min(this.imgWidth - box.x, Math.round(right - x));
    box.height = Math.min(this.imgHeight - box.y, Math.round(bottom - y));
  }

  pushHistory() {
    this.undoStack.push(JSON.stringify(this.boxes));
    if (this.undoStack.length > 50) this.undoStack.shift();
    this.redoStack = [];
  }

  undo() {
    if (this.undoStack.length === 0) return;
    this.redoStack.push(JSON.stringify(this.boxes));
    const prev = this.undoStack.pop();
    this.boxes = JSON.parse(prev);
    this.selectedBoxId = null;
    this.app.onAnnotationsChanged(this.boxes);
    this.redraw();
  }

  redo() {
    if (this.redoStack.length === 0) return;
    this.undoStack.push(JSON.stringify(this.boxes));
    const next = this.redoStack.pop();
    this.boxes = JSON.parse(next);
    this.selectedBoxId = null;
    this.app.onAnnotationsChanged(this.boxes);
    this.redraw();
  }

  onActiveClassChanged() {
    if (this.selectedBoxId) {
      const box = this.boxes.find(b => b.id === this.selectedBoxId);
      if (box) {
        this.pushHistory();
        box.category_id = this.app.classManager.activeClassId;
        this.updateBDW(box);
        this.app.onAnnotationsChanged(this.boxes);
        this.redraw();
      }
    }
  }

  drawRoundedRect(ctx, x, y, width, height, radius = 4) {
    if (typeof ctx.roundRect === 'function') {
      ctx.beginPath();
      ctx.roundRect(x, y, width, height, radius);
      return;
    }
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  redraw() {
    this.initElements();

    if (this.isComparisonMode) {
      // 1. Redraw Left Pane: BEFORE (Current confirmed annotations only)
      if (this.ctxBefore && this.canvasBefore) {
        this.redrawCanvas(this.ctxBefore, this.canvasBefore, false);
      }
      // 2. Redraw Right Pane: AFTER (Current annotations + candidate proposals)
      if (this.ctxAfter && this.canvasAfter) {
        this.redrawCanvas(this.ctxAfter, this.canvasAfter, true);
      }

      // Update badge texts in pane headers
      if (this.beforeCountBadge) {
        const cnt = this.boxes ? this.boxes.length : 0;
        this.beforeCountBadge.textContent = `${cnt} box${cnt === 1 ? '' : 'es'}`;
      }
      if (this.afterCountBadge) {
        const propCnt = this.proposals ? this.proposals.length : 0;
        const totalCnt = (this.boxes ? this.boxes.length : 0) + propCnt;
        this.afterCountBadge.textContent = `+${propCnt} proposed (${totalCnt} total)`;
      }
    } else {
      // Redraw Single Canvas
      if (this.ctx && this.canvas) {
        this.redrawCanvas(this.ctx, this.canvas, true);
      }
    }

    // Redraw Dual Axis Rulers
    this.renderRulers();
  }

  redrawCanvas(ctx, canvasEl, includeProposals = true) {
    if (!ctx || !canvasEl) return;
    const rect = canvasEl.getBoundingClientRect();
    const w = rect.width || (canvasEl.width / (window.devicePixelRatio || 1)) || 500;
    const h = rect.height || (canvasEl.height / (window.devicePixelRatio || 1)) || 500;

    ctx.clearRect(0, 0, w, h);

    if (!this.isLoaded || !this.image) return;

    // Draw Spectrogram Image
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.translate(this.offsetX, this.offsetY);
    ctx.scale(this.scale, this.scale);
    ctx.drawImage(this.image, 0, 0, this.imgWidth, this.imgHeight);

    // Image Boundary Stroke
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.3)';
    ctx.lineWidth = 1 / this.scale;
    ctx.strokeRect(0, 0, this.imgWidth, this.imgHeight);
    ctx.restore();

    // Draw Confirmed Existing Bounding Boxes
    if (this.boxes && this.boxes.length > 0) {
      this.boxes.forEach(box => {
        if (!box.isHidden) {
          const isSelected = box.id === this.selectedBoxId;
          const isHovered = box.id === this.hoveredBoxId;
          this.renderBoundingBoxOn(ctx, box, isSelected, isHovered);
        }
      });
    }

    // Draw Candidate Proposals (After view only)
    if (includeProposals && this.proposals && this.proposals.length > 0) {
      this.proposals.forEach(prop => {
        const isHovered = prop.id === this.hoveredBoxId;
        this.renderProposalBoxOn(ctx, prop, isHovered);
      });
    }

    // Draw 2-Corner Click Mode Preview
    if (this.currentMode === 'draw' && includeProposals) {
      this.renderDrawPreviewOn(ctx, { width: w, height: h });
    }

    // Draw Measure Tool Line
    if (this.currentMode === 'measure' && this.measureStart && this.measureEnd) {
      this.renderMeasureToolOn(ctx);
    }
  }

  renderBoundingBoxOn(ctx, box, isSelected, isHovered = false) {
    const cls = this.app.classManager.classes.find(c => c.id === box.category_id) || {
      name: "Unknown",
      color: "#00e5ff"
    };

    const p = this.imageToScreenCoords(box.x, box.y);
    const scrW = box.width * this.scale;
    const scrH = box.height * this.scale;

    ctx.save();

    // Box Fill & Stroke
    const fillAlpha = isSelected ? 0.28 : (isHovered ? 0.22 : 0.15);
    ctx.fillStyle = this.hexToRgba(cls.color, fillAlpha);
    ctx.fillRect(p.x, p.y, scrW, scrH);

    ctx.strokeStyle = cls.color;
    ctx.lineWidth = isSelected ? 2.5 : (isHovered ? 2.0 : 1.5);
    if (isSelected || isHovered) {
      ctx.shadowColor = cls.color;
      ctx.shadowBlur = isSelected ? 10 : 6;
    }
    ctx.strokeRect(p.x, p.y, scrW, scrH);
    ctx.restore();

    // Label Tag
    this.renderBoxTagOn(ctx, box, cls, p.x, p.y, scrW, scrH, isSelected, isHovered);

    // 8 Resize Handles if selected
    if (isSelected && !box.isLocked) {
      this.renderResizeHandlesOn(ctx, box, cls.color);
    }
  }

  renderProposalBoxOn(ctx, box, isHovered) {
    if (!box) return;
    const p = this.imageToScreenCoords(box.x, box.y);
    const scrW = box.width * this.scale;
    const scrH = box.height * this.scale;

    let cls = null;
    if (this.app.classManager) {
      if (typeof this.app.classManager.getClassById === 'function') {
        cls = this.app.classManager.getClassById(box.category_id);
      } else if (Array.isArray(this.app.classManager.classes)) {
        cls = this.app.classManager.classes.find(c => c.id === Number(box.category_id));
      }
    }
    if (!cls) {
      cls = { color: '#a855f7', name: 'Proposal' };
    }
    const propColor = cls.color || '#a855f7';

    ctx.save();
    // Translucent fill
    ctx.fillStyle = isHovered ? this.hexToRgba(propColor, 0.35) : this.hexToRgba(propColor, 0.20);
    ctx.fillRect(p.x, p.y, scrW, scrH);

    // Dashed glowing border
    ctx.strokeStyle = propColor;
    ctx.lineWidth = isHovered ? 2.5 : 1.8;
    ctx.setLineDash([6, 3]);
    ctx.shadowColor = propColor;
    ctx.shadowBlur = isHovered ? 12 : 6;
    ctx.strokeRect(p.x, p.y, scrW, scrH);
    ctx.restore();

    // Proposal Tag - Only Class Name
    const tagText = cls.name || 'Proposal';

    ctx.save();
    ctx.font = 'bold 10px JetBrains Mono, monospace';
    const textWidth = ctx.measureText(tagText).width;
    const tagHeight = 18;

    let tagX = p.x;
    let tagY = p.y - tagHeight - 2;
    if (tagY < 0) tagY = p.y + 2;

    ctx.fillStyle = isHovered ? this.hexToRgba(propColor, 0.95) : 'rgba(25, 15, 40, 0.92)';
    ctx.strokeStyle = propColor;
    ctx.lineWidth = isHovered ? 1.5 : 1.0;

    this.drawRoundedRect(ctx, tagX, tagY, textWidth + 12, tagHeight, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = isHovered ? '#ffffff' : '#e9d5ff';
    ctx.fillText(tagText, tagX + 6, tagY + 13);
    ctx.restore();
  }

  renderBoxTagOn(ctx, box, cls, screenX, screenY, screenW, screenH, isSelected, isHovered = false) {
    // Annotation Tag - Only Class Name
    const tagText = cls.name || 'Signal';

    ctx.save();
    ctx.font = 'bold 10px JetBrains Mono, monospace';
    const textWidth = ctx.measureText(tagText).width;
    const tagHeight = 18;

    let tagX = screenX;
    let tagY = screenY - tagHeight - 2;
    if (tagY < 0) tagY = screenY + 2;

    ctx.fillStyle = isSelected ? cls.color : (isHovered ? 'rgba(30, 45, 75, 0.95)' : 'rgba(16, 21, 34, 0.9)');
    ctx.strokeStyle = cls.color;
    ctx.lineWidth = isHovered || isSelected ? 1.5 : 1;

    this.drawRoundedRect(ctx, tagX, tagY, textWidth + 12, tagHeight, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = isSelected ? '#000000' : '#ffffff';
    ctx.fillText(tagText, tagX + 6, tagY + 13);
    ctx.restore();
  }

  renderResizeHandlesOn(ctx, box, color) {
    const handles = this.getHandleCoordinates(box);
    const size = 7;

    ctx.save();
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;

    for (const pt of Object.values(handles)) {
      ctx.beginPath();
      ctx.rect(pt.x - size / 2, pt.y - size / 2, size, size);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  renderDrawPreviewOn(ctx, rect) {
    const activeClass = this.app.classManager.getActiveClass() || { color: '#00e5ff', name: 'Signal' };
    const clsColor = activeClass.color || '#00e5ff';

    if (this.isWaitingSecondCorner && this.cornerClick1 && this.hoverImgCoords) {
      const p1_img = this.cornerClick1;
      const p2_img = this.hoverImgCoords;

      const x1 = Math.min(p1_img.x, p2_img.x);
      const y1 = Math.min(p1_img.y, p2_img.y);
      const x2 = Math.max(p1_img.x, p2_img.x);
      const y2 = Math.max(p1_img.y, p2_img.y);
      const w = x2 - x1;
      const h = y2 - y1;

      const pTopLeft = this.imageToScreenCoords(x1, y1);
      const scrW = w * this.scale;
      const scrH = h * this.scale;

      const p1_scr = this.imageToScreenCoords(p1_img.x, p1_img.y);
      const p2_scr = this.imageToScreenCoords(p2_img.x, p2_img.y);

      ctx.save();

      // 1. Dashed Box Preview
      ctx.strokeStyle = clsColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.fillStyle = this.hexToRgba(clsColor, 0.2);
      ctx.fillRect(pTopLeft.x, pTopLeft.y, scrW, scrH);
      ctx.strokeRect(pTopLeft.x, pTopLeft.y, scrW, scrH);

      // 2. Corner 1 Beacon (Anchor)
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(p1_scr.x, p1_scr.y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = clsColor;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Pulse ring around Corner 1
      ctx.beginPath();
      ctx.arc(p1_scr.x, p1_scr.y, 11, 0, 2 * Math.PI);
      ctx.strokeStyle = clsColor;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // 3. Corner 2 Crosshair (Mouse Pointer)
      ctx.beginPath();
      ctx.arc(p2_scr.x, p2_scr.y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = clsColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      // 4. Dimension Badge (Δt, Δf)
      if (this.chunkMeta && w > 0 && h > 0) {
        const dt_us = (w / this.imgWidth) * (this.chunkMeta.duration_ms * 1000.0);
        const df_mhz = (h / this.imgHeight) * (this.chunkMeta.fs_mhz);
        const dimText = `Δt: ${dt_us.toFixed(1)} µs | Δf: ${df_mhz.toFixed(2)} MHz`;

        ctx.font = 'bold 11px JetBrains Mono, monospace';
        const dimW = ctx.measureText(dimText).width;
        const badgeX = Math.max(10, Math.min((rect.width || 500) - dimW - 24, pTopLeft.x + (scrW - dimW) / 2));
        const badgeY = pTopLeft.y > 30 ? pTopLeft.y - 12 : pTopLeft.y + scrH + 22;

        ctx.fillStyle = 'rgba(10, 14, 23, 0.9)';
        ctx.strokeStyle = clsColor;
        ctx.lineWidth = 1;
        this.drawRoundedRect(ctx, badgeX - 8, badgeY - 14, dimW + 16, 20, 4);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.fillText(dimText, badgeX, badgeY);
      }

      // 5. Instruction banner at top of canvas
      const instText = "📍 Corner 1 Set. Click Corner 2 to place box | Esc to cancel";
      ctx.font = '12px Inter, sans-serif';
      const instW = ctx.measureText(instText).width;
      const bannerX = ((rect.width || 500) - instW) / 2;

      ctx.fillStyle = 'rgba(16, 21, 34, 0.92)';
      ctx.strokeStyle = clsColor;
      ctx.lineWidth = 1;
      this.drawRoundedRect(ctx, bannerX - 14, 12, instW + 28, 26, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#00e5ff';
      ctx.fillText(instText, bannerX, 29);

      ctx.restore();
    } else if (!this.isWaitingSecondCorner && this.hoverImgCoords) {
      ctx.save();
      const hintText = `[Draw Mode: ${activeClass.name}] Click 1: Select Top-Left Corner`;
      ctx.font = '12px Inter, sans-serif';
      const hintW = ctx.measureText(hintText).width;
      const bannerX = ((rect.width || 500) - hintW) / 2;

      ctx.fillStyle = 'rgba(16, 21, 34, 0.85)';
      ctx.strokeStyle = 'rgba(0, 229, 255, 0.4)';
      ctx.lineWidth = 1;
      this.drawRoundedRect(ctx, bannerX - 12, 12, hintW + 24, 24, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(hintText, bannerX, 28);
      ctx.restore();
    }
  }

  renderMeasureToolOn(ctx) {
    const p1 = this.imageToScreenCoords(this.measureStart.x, this.measureStart.y);
    const p2 = this.imageToScreenCoords(this.measureEnd.x, this.measureEnd.y);

    const c1 = BDWCalculator.pixelToCoord(this.measureStart.x, this.measureStart.y, this.imgWidth, this.imgHeight, this.chunkMeta);
    const c2 = BDWCalculator.pixelToCoord(this.measureEnd.x, this.measureEnd.y, this.imgWidth, this.imgHeight, this.chunkMeta);

    const deltaT = Math.abs(c2.timeUs - c1.timeUs).toFixed(3);
    const deltaF = Math.abs(c2.freqMhz - c1.freqMhz).toFixed(3);

    ctx.save();
    ctx.strokeStyle = '#ffff00';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);

    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();

    // Text badge
    const midX = (p1.x + p2.x) / 2;
    const midY = (p1.y + p2.y) / 2;
    const info = `Δt: ${deltaT} µs | Δf: ${deltaF} MHz`;

    ctx.font = 'bold 11px JetBrains Mono, monospace';
    const tw = ctx.measureText(info).width;

    ctx.fillStyle = 'rgba(0,0,0,0.85)';
    ctx.fillRect(midX - tw / 2 - 6, midY - 20, tw + 12, 22);
    ctx.strokeStyle = '#ffff00';
    ctx.strokeRect(midX - tw / 2 - 6, midY - 20, tw + 12, 22);

    ctx.fillStyle = '#ffff00';
    ctx.fillText(info, midX - tw / 2, midY - 5);
    ctx.restore();
  }

  renderRulers() {
    if (!this.chunkMeta || !this.rulerTopCtx || !this.rulerLeftCtx) return;

    const w = this.rulerTop.width / (window.devicePixelRatio || 1);
    const h = this.rulerLeft.height / (window.devicePixelRatio || 1);

    // Top Time Ruler (Horizontal)
    this.rulerTopCtx.clearRect(0, 0, w, 24);
    this.rulerTopCtx.fillStyle = '#9aa8c7';
    this.rulerTopCtx.font = '9px JetBrains Mono, monospace';

    const numTicksX = 8;
    for (let i = 0; i <= numTicksX; i++) {
      const imgX = (i / numTicksX) * this.imgWidth;
      const screenX = imgX * this.scale + this.offsetX;
      if (screenX >= 0 && screenX <= w) {
        const coord = BDWCalculator.pixelToCoord(imgX, 0, this.imgWidth, this.imgHeight, this.chunkMeta);
        this.rulerTopCtx.fillRect(screenX, 16, 1, 8);
        this.rulerTopCtx.fillText(`${coord.timeUs}µs`, screenX + 3, 12);
      }
    }

    // Left Frequency Ruler (Vertical)
    this.rulerLeftCtx.clearRect(0, 0, 54, h);
    this.rulerLeftCtx.fillStyle = '#9aa8c7';
    this.rulerLeftCtx.font = '9px JetBrains Mono, monospace';

    const numTicksY = 8;
    for (let i = 0; i <= numTicksY; i++) {
      const imgY = (i / numTicksY) * this.imgHeight;
      const screenY = imgY * this.scale + this.offsetY;
      if (screenY >= 0 && screenY <= h) {
        const coord = BDWCalculator.pixelToCoord(0, imgY, this.imgWidth, this.imgHeight, this.chunkMeta);
        this.rulerLeftCtx.fillRect(46, screenY, 8, 1);
        this.rulerLeftCtx.fillText(`${coord.freqMhz.toFixed(1)}M`, 4, screenY - 2);
      }
    }
  }

  hexToRgba(hex, alpha = 1.0) {
    let c = hex.replace('#', '');
    if (c.length === 3) c = c.split('').map(x => x + x).join('');
    const num = parseInt(c, 16);
    return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`;
  }
}
