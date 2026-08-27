/**
 * Export Manager for COCO + BDW JSON and Dataset ZIP Bundle
 */

export class ExportManager {
  constructor(app) {
    this.app = app;
    this.initEventListeners();
  }

  initEventListeners() {
    const exportCocoBtn = document.getElementById('exportCocoBtn');
    const exportZipBtn = document.getElementById('exportZipBtn');
    const exportZipWithIqBtn = document.getElementById('exportZipWithIqBtn');
    const importCocoBtn = document.getElementById('importCocoBtn');
    const importCocoDropdownBtn = document.getElementById('importCocoDropdownBtn');
    const importCocoFileInput = document.getElementById('importCocoFileInput');

    if (exportCocoBtn) {
      exportCocoBtn.addEventListener('click', () => this.exportCOCO());
    }
    if (exportZipBtn) {
      exportZipBtn.addEventListener('click', () => this.exportZIP(false));
    }
    if (exportZipWithIqBtn) {
      exportZipWithIqBtn.addEventListener('click', () => this.exportZIP(true));
    }
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
        // Update annotations cache
        this.app.annotationsCache = data.annotations || {};

        // Update classes if returned
        if (data.classes && this.app.classManager) {
          this.app.classManager.setClasses(data.classes);
        }

        // Re-load and draw bounding boxes on current canvas
        this.app.loadChunkData(this.app.navigation.currentChunkId);

        // Re-render filmstrip timeline badges
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

  async exportCOCO() {
    if (this.app.saveCurrentChunkAnnotations) {
      await this.app.saveCurrentChunkAnnotations();
    }

    try {
      const resp = await fetch('/api/export/coco');
      if (!resp.ok) throw new Error("Failed to export COCO JSON");

      const data = await resp.json();
      const jsonBlob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const filename = `annotations_coco_bdw_${this.app.sessionSummary?.source_filename || 'rf_dataset'}.json`;

      this.downloadBlob(jsonBlob, filename);
      this.app.showNotification("COCO JSON with BDW structure exported successfully!");
    } catch (e) {
      console.error("Export error:", e);
      alert("Error exporting COCO dataset: " + e.message);
    }
  }

  async exportZIP(includeIq = false) {
    if (this.app.saveCurrentChunkAnnotations) {
      await this.app.saveCurrentChunkAnnotations();
    }

    const loaderMsg = includeIq
      ? "Generating complete dataset bundle (Raw .iq Chunks + Spectrograms + COCO JSON + CSV)..."
      : "Generating dataset ZIP bundle (Spectrograms + COCO JSON + CSV)...";

    this.app.showLoader(true, loaderMsg);

    try {
      const resp = await fetch(`/api/export/zip?include_iq=${includeIq}`);
      if (!resp.ok) throw new Error("Failed to export dataset ZIP");

      const blob = await resp.blob();
      const suffix = includeIq ? '_with_iq' : '';
      const cleanSource = (this.app.sessionSummary?.source_filename || 'dataset').replace(/\.[^/.]+$/, "");
      const filename = `rf_spectrogram_bundle_${cleanSource}${suffix}.zip`;

      this.downloadBlob(blob, filename);
      const notifMsg = includeIq
        ? "Complete dataset bundle with .iq chunks downloaded successfully!"
        : "Dataset bundle downloaded successfully!";
      this.app.showNotification(notifMsg, "success");
    } catch (e) {
      console.error("ZIP Export error:", e);
      alert("Error generating ZIP package: " + e.message);
    } finally {
      this.app.showLoader(false);
    }
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
}
