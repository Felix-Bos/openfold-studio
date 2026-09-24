import { Tooltip, onResize, setupCanvas } from "./canvas.js";

/**
 * Square matrix drawn as a heatmap (one cell per matrix entry), with a hover
 * tooltip and an optional click callback. Used for the PDE matrix and the
 * attention maps.
 *
 * options.color(value, max) -> [r, g, b]
 * options.tooltip(row, col, value) -> HTML string
 * options.onClick(row, col, value)
 */
export class MatrixCanvas {
  constructor(container, { color, tooltip, onClick } = {}) {
    this.container = container;
    this.canvas = container.querySelector("canvas");
    this.tooltip = new Tooltip(container);
    this.options = { color, tooltip, onClick };
    this.matrix = null;
    this.max = 1;
    this.highlight = null; // row index to outline
    this.image = document.createElement("canvas");
    this.#bindEvents();
    onResize(container, () => this.draw());
  }

  setMatrix(matrix, max = null) {
    this.matrix = matrix;
    this.max = max ?? (Math.max(...matrix.map((row) => Math.max(...row))) || 1);
    this.#renderImage();
    this.draw();
  }

  setHighlightedRow(row) {
    this.highlight = row;
    this.draw();
  }

  /** Repaints the cached image (e.g. after a theme change). */
  refresh() {
    if (!this.matrix) return;
    this.#renderImage();
    this.draw();
  }

  #renderImage() {
    const n = this.matrix.length;
    this.image.width = n;
    this.image.height = n;
    const context = this.image.getContext("2d");
    const data = context.createImageData(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const [r, g, b] = this.options.color(this.matrix[i][j], this.max);
        data.data.set([r, g, b, 255], (i * n + j) * 4);
      }
    }
    context.putImageData(data, 0, 0);
  }

  draw() {
    if (!this.matrix) return;
    const width = this.canvas.clientWidth;
    const { context } = setupCanvas(this.canvas, width);
    context.imageSmoothingEnabled = false;
    context.drawImage(this.image, 0, 0, width, width);
    if (this.highlight !== null) {
      const cell = width / this.matrix.length;
      context.strokeStyle = "#ffffff";
      context.lineWidth = 1.5;
      context.strokeRect(0, this.highlight * cell, width, Math.max(cell, 2));
    }
  }

  #cellAt(event) {
    if (!this.matrix) return null;
    const rect = this.canvas.getBoundingClientRect();
    const n = this.matrix.length;
    const col = Math.floor(((event.clientX - rect.left) / rect.width) * n);
    const row = Math.floor(((event.clientY - rect.top) / rect.height) * n);
    if (row < 0 || col < 0 || row >= n || col >= n) return null;
    return { row, col, value: this.matrix[row][col], x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  #bindEvents() {
    this.canvas.addEventListener("mousemove", (event) => {
      const cell = this.#cellAt(event);
      if (!cell || !this.options.tooltip) return this.tooltip.hide();
      this.tooltip.show(cell.x, cell.y, this.options.tooltip(cell.row, cell.col, cell.value));
    });
    this.canvas.addEventListener("mouseleave", () => this.tooltip.hide());
    this.canvas.addEventListener("click", (event) => {
      const cell = this.#cellAt(event);
      if (cell && this.options.onClick) this.options.onClick(cell.row, cell.col, cell.value);
    });
  }
}
