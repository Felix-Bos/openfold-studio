import { Tooltip, onResize, setupCanvas } from "../lib/canvas.js";
import { COLORS } from "../lib/palette.js";

const HEIGHT = 150;
const PADDING = { top: 12, right: 8, bottom: 22, left: 34 };

/**
 * Bar chart of the mean attention entropy of each block of a family.
 * Low bar = focused attention (few partners), high bar = diffuse attention.
 * A dashed line marks the maximum entropy, log(N) = perfectly uniform
 * attention. Clicking a bar calls `onSelect(block)`.
 */
export class EntropyChart {
  constructor(container, { onSelect } = {}) {
    this.container = container;
    this.canvas = container.querySelector("canvas");
    this.tooltip = new Tooltip(container);
    this.onSelect = onSelect;
    this.values = [];
    this.maxEntropy = 1;
    this.current = 0;
    this.#bindEvents();
    onResize(container, () => this.draw());
  }

  setData(values, maxEntropy, current) {
    this.values = values || [];
    this.maxEntropy = maxEntropy || Math.max(...this.values, 1);
    this.current = current;
    this.draw();
  }

  setCurrent(block) {
    this.current = block;
    this.draw();
  }

  #geometry(width) {
    const plotWidth = width - PADDING.left - PADDING.right;
    return { plotWidth, barWidth: plotWidth / Math.max(this.values.length, 1) };
  }

  #y(value) {
    return PADDING.top + (1 - value / this.maxEntropy) * (HEIGHT - PADDING.top - PADDING.bottom);
  }

  draw() {
    const { context: ctx, width } = setupCanvas(this.canvas, HEIGHT);
    ctx.clearRect(0, 0, width, HEIGHT);
    if (!this.values.length) return;
    const { barWidth } = this.#geometry(width);
    const minimum = Math.min(...this.values);

    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillStyle = COLORS.inkFaint;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(this.maxEntropy.toFixed(1), PADDING.left - 6, this.#y(this.maxEntropy));
    ctx.fillText("0", PADDING.left - 6, this.#y(0));

    ctx.strokeStyle = COLORS.inkFaint;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(PADDING.left, this.#y(this.maxEntropy));
    ctx.lineTo(width - PADDING.right, this.#y(this.maxEntropy));
    ctx.stroke();
    ctx.setLineDash([]);

    this.values.forEach((value, block) => {
      const x = PADDING.left + block * barWidth;
      const y = this.#y(value);
      ctx.fillStyle = block === this.current ? COLORS.accent : value === minimum ? COLORS.veryLow : COLORS.line;
      ctx.fillRect(x + 1, y, Math.max(barWidth - 2, 1), this.#y(0) - y);
    });

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = COLORS.inkFaint;
    const step = this.values.length > 30 ? 8 : 4;
    for (let block = 0; block < this.values.length; block += step) {
      ctx.fillText(String(block), PADDING.left + (block + 0.5) * barWidth, HEIGHT - PADDING.bottom + 6);
    }
  }

  #blockAt(event) {
    const rect = this.canvas.getBoundingClientRect();
    const { barWidth } = this.#geometry(rect.width);
    const block = Math.floor((event.clientX - rect.left - PADDING.left) / barWidth);
    return block >= 0 && block < this.values.length ? block : null;
  }

  #bindEvents() {
    this.canvas.style.cursor = "pointer";
    this.canvas.addEventListener("mousemove", (event) => {
      const block = this.#blockAt(event);
      if (block === null) return this.tooltip.hide();
      const rect = this.canvas.getBoundingClientRect();
      const { barWidth } = this.#geometry(rect.width);
      const value = this.values[block];
      this.tooltip.show(
        PADDING.left + (block + 0.5) * barWidth,
        this.#y(value),
        `block ${block} · entropy <b>${value.toFixed(2)}</b> (${Math.round((value / this.maxEntropy) * 100)}% of uniform)`,
      );
    });
    this.canvas.addEventListener("mouseleave", () => this.tooltip.hide());
    this.canvas.addEventListener("click", (event) => {
      const block = this.#blockAt(event);
      if (block !== null && this.onSelect) this.onSelect(block);
    });
  }
}
