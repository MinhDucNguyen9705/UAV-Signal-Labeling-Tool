/**
 * Navigation Manager: Top Chunk Controls & Bottom Filmstrip
 */

export class NavigationManager {
  constructor(app) {
    this.app = app;
    this.currentChunkId = 0;
    this.chunks = [];

    this.prevBtn = document.getElementById('prevChunkBtn');
    this.nextBtn = document.getElementById('nextChunkBtn');
    this.selectEl = document.getElementById('chunkSelect');
    this.timeBadge = document.getElementById('chunkTimeBadge');
    this.filmstrip = document.getElementById('filmstripContainer');

    this.initEventListeners();
  }

  initEventListeners() {
    if (this.prevBtn) {
      this.prevBtn.addEventListener('click', () => this.previousChunk());
    }
    if (this.nextBtn) {
      this.nextBtn.addEventListener('click', () => this.nextChunk());
    }
    if (this.selectEl) {
      this.selectEl.addEventListener('change', (e) => {
        this.goToChunk(parseInt(e.target.value, 10));
      });
    }
  }

  setChunks(chunksList) {
    this.chunks = chunksList || [];
    this.populateSelect();
    this.renderFilmstrip();
    if (this.chunks.length > 0) {
      if (this.currentChunkId >= this.chunks.length) {
        this.currentChunkId = 0;
      }
      this.updateUI();
    }
  }

  populateSelect() {
    if (!this.selectEl) return;
    this.selectEl.innerHTML = '';

    this.chunks.forEach((chunk, i) => {
      const opt = document.createElement('option');
      opt.value = chunk.id;
      const startMs = (chunk.start_time_us / 1000.0).toFixed(2);
      const endMs = (chunk.end_time_us / 1000.0).toFixed(2);
      opt.textContent = `Chunk ${i + 1}/${this.chunks.length} [${startMs} - ${endMs} ms]`;
      this.selectEl.appendChild(opt);
    });
  }

  goToChunk(chunkId) {
    if (chunkId < 0 || chunkId >= this.chunks.length) return;
    if (chunkId === this.currentChunkId && this.app.canvas?.isLoaded) return;

    // Auto-save annotations of current chunk before switching
    if (this.app.saveCurrentChunkAnnotations) {
      this.app.saveCurrentChunkAnnotations();
    }

    this.currentChunkId = chunkId;
    this.updateUI();
    this.app.loadChunkData(this.currentChunkId);
  }

  previousChunk() {
    if (this.currentChunkId > 0) {
      this.goToChunk(this.currentChunkId - 1);
    }
  }

  nextChunk() {
    if (this.currentChunkId < this.chunks.length - 1) {
      this.goToChunk(this.currentChunkId + 1);
    }
  }

  updateUI() {
    const chunk = this.chunks[this.currentChunkId];
    if (!chunk) return;

    if (this.prevBtn) {
      this.prevBtn.disabled = (this.currentChunkId <= 0);
    }
    if (this.nextBtn) {
      this.nextBtn.disabled = (this.currentChunkId >= this.chunks.length - 1);
    }
    if (this.selectEl) {
      this.selectEl.value = this.currentChunkId;
    }
    if (this.timeBadge) {
      const startMs = (chunk.start_time_us / 1000.0).toFixed(3);
      const endMs = (chunk.end_time_us / 1000.0).toFixed(3);
      this.timeBadge.textContent = `${startMs} ms → ${endMs} ms (Δ ${(endMs - startMs).toFixed(3)} ms)`;
    }

    // Update filmstrip active highlight and scroll into view
    if (this.filmstrip) {
      const items = this.filmstrip.querySelectorAll('.filmstrip-item');
      items.forEach((item, idx) => {
        if (idx === this.currentChunkId) {
          item.classList.add('active');
          item.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
        } else {
          item.classList.remove('active');
        }
      });
    }
  }

  renderFilmstrip() {
    if (!this.filmstrip) return;
    this.filmstrip.innerHTML = '';

    if (this.thumbObserver) {
      this.thumbObserver.disconnect();
    }

    const isLargeDataset = this.chunks.length > 20;
    if (isLargeDataset && 'IntersectionObserver' in window) {
      this.thumbObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const target = entry.target;
            const src = target.getAttribute('data-bg');
            if (src) {
              target.style.backgroundImage = `url('${src}')`;
              target.removeAttribute('data-bg');
            }
            this.thumbObserver.unobserve(target);
          }
        });
      }, { root: this.filmstrip, rootMargin: '120px' });
    }

    this.chunks.forEach((chunk, idx) => {
      const item = document.createElement('div');
      item.className = `filmstrip-item ${idx === this.currentChunkId ? 'active' : ''}`;
      item.title = `Jump to Chunk ${idx + 1} (${(chunk.start_time_us/1000).toFixed(2)} ms)`;

      const startMs = (chunk.start_time_us / 1000.0).toFixed(1);
      const annCount = this.app.getChunkBoxCount ? this.app.getChunkBoxCount(chunk.id) : 0;

      item.innerHTML = `
        <div class="filmstrip-thumb" id="thumb_${chunk.id}"></div>
        <div class="filmstrip-label">
          <span>#${idx + 1} (${startMs}ms)</span>
          <span class="filmstrip-badge" id="badge_${chunk.id}">${annCount > 0 ? annCount : ''}</span>
        </div>
      `;

      item.addEventListener('click', () => {
        this.goToChunk(chunk.id);
      });

      this.filmstrip.appendChild(item);

      const thumbEl = item.querySelector('.filmstrip-thumb');
      const timestamp = Date.now();
      const thumbUrl = `/api/chunks/${chunk.id}/spectrogram?width=160&height=90&t=${timestamp}`;

      if (isLargeDataset && this.thumbObserver) {
        thumbEl.setAttribute('data-bg', thumbUrl);
        this.thumbObserver.observe(thumbEl);
      } else {
        thumbEl.style.backgroundImage = `url('${thumbUrl}')`;
      }
    });
  }

  updateChunkBadge(chunkId, count) {
    const badge = document.getElementById(`badge_${chunkId}`);
    if (badge) {
      badge.textContent = count > 0 ? count : '';
      badge.style.display = count > 0 ? 'inline-block' : 'none';
    }
  }
}
