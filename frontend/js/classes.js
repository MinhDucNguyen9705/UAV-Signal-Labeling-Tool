/**
 * Class and Color Palette Manager
 */

export class ClassManager {
  constructor(app) {
    this.app = app;
    this.classes = [];
    this.activeClassId = 1;
    this.container = document.getElementById('classesListContainer');
    this.quickPicker = document.getElementById('classQuickPicker');
  }

  setClasses(classesList) {
    this.classes = classesList;
    if (!this.classes.some(c => c.id === this.activeClassId) && this.classes.length > 0) {
      this.activeClassId = this.classes[0].id;
    }
    this.render();
    this.renderQuickPicker();
  }

  getActiveClass() {
    return this.classes.find(c => c.id === this.activeClassId) || this.classes[0] || {
      id: 1,
      name: "Default_Signal",
      color: "#00e5ff",
      type_of_signal: "Unknown",
      protocol: "Generic"
    };
  }

  getClassById(classId) {
    const numId = Number(classId);
    return this.classes.find(c => c.id === numId) || this.classes[0] || {
      id: 1,
      name: "Default_Signal",
      color: "#00e5ff",
      type_of_signal: "Unknown",
      protocol: "Generic"
    };
  }

  setActiveClass(classId) {
    this.activeClassId = Number(classId);
    this.render();
    this.renderQuickPicker();
    if (this.app.canvas) {
      this.app.canvas.onActiveClassChanged();
    }
  }

  addClass(name, color, type_of_signal = "Unknown", protocol = "Generic") {
    const nextId = this.classes.length > 0 ? Math.max(...this.classes.map(c => c.id)) + 1 : 1;
    const newClass = {
      id: nextId,
      name: name || `Class_${nextId}`,
      color: color || this.getRandomColor(),
      type_of_signal: type_of_signal || "Unknown",
      protocol: protocol || "Generic"
    };
    this.classes.push(newClass);
    this.activeClassId = nextId;
    this.syncBackend();
    this.render();
    this.renderQuickPicker();
    return newClass;
  }

  updateClass(classId, updates) {
    const idx = this.classes.findIndex(c => c.id === classId);
    if (idx !== -1) {
      this.classes[idx] = { ...this.classes[idx], ...updates };
      this.syncBackend();
      this.render();
      this.renderQuickPicker();
      if (this.app.canvas) {
        this.app.canvas.redraw();
      }
    }
  }

  deleteClass(classId) {
    if (this.classes.length <= 1) {
      alert("At least one class is required.");
      return;
    }
    this.classes = this.classes.filter(c => c.id !== classId);
    if (this.activeClassId === classId) {
      this.activeClassId = this.classes[0].id;
    }
    this.syncBackend();
    this.render();
    this.renderQuickPicker();
    if (this.app.canvas) {
      this.app.canvas.redraw();
    }
  }

  async syncBackend() {
    try {
      await fetch('/api/classes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ classes: this.classes })
      });
    } catch (e) {
      console.error("Failed to sync classes with backend:", e);
    }
  }

  render() {
    if (!this.container) return;
    this.container.innerHTML = '';

    this.classes.forEach((cls, idx) => {
      const row = document.createElement('div');
      row.className = `class-item-row ${cls.id === this.activeClassId ? 'active' : ''}`;
      if (cls.id === this.activeClassId) {
        row.style.borderColor = cls.color;
      }

      row.innerHTML = `
        <input type="color" class="class-color-input" value="${cls.color}" title="Change color">
        <div class="class-name-tag" style="color: ${cls.color}">${cls.name}</div>
        <span style="font-family: var(--font-mono); font-size: 10px; color: var(--text-muted);">[${idx + 1}]</span>
        <button class="icon-btn delete-btn" title="Delete class"><i class="fas fa-trash"></i></button>
      `;

      // Color picker change
      const colorInput = row.querySelector('.class-color-input');
      colorInput.addEventListener('input', (e) => {
        this.updateClass(cls.id, { color: e.target.value });
      });

      // Select active class on click
      row.addEventListener('click', (e) => {
        if (!e.target.classList.contains('class-color-input') && !e.target.closest('.delete-btn')) {
          this.setActiveClass(cls.id);
        }
      });

      // Delete button
      const delBtn = row.querySelector('.delete-btn');
      delBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.deleteClass(cls.id);
      });

      this.container.appendChild(row);
    });
  }

  renderQuickPicker() {
    if (!this.quickPicker) return;
    this.quickPicker.innerHTML = '';

    this.classes.forEach((cls, idx) => {
      const dot = document.createElement('div');
      dot.className = `class-dot-btn ${cls.id === this.activeClassId ? 'active' : ''}`;
      dot.style.backgroundColor = cls.color;
      dot.title = `${cls.name} (Key: ${idx + 1})`;
      dot.innerText = idx < 9 ? (idx + 1) : '';

      dot.addEventListener('click', () => {
        this.setActiveClass(cls.id);
      });

      this.quickPicker.appendChild(dot);
    });
  }

  getRandomColor() {
    const palette = ['#00e5ff', '#ff007f', '#00ff66', '#ffaa00', '#bd00ff', '#ff334b', '#ffff00', '#00e1d9'];
    return palette[Math.floor(Math.random() * palette.length)];
  }
}
