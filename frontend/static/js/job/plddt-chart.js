import { Tooltip, onResize, setupCanvas } from "../lib/canvas.js";
import { COLORS, plddtColor } from "../lib/palette.js";

const HEIGHT = 190;
const PADDING = { top: 10, right: 12, bottom: 24, left: 34 };
const BANDS = [
  { from: 90, to: 100, color: () => COLORS.veryHigh },
  { from: 70, to: 90, color: () => COLORS.high },
  { from: 50, to: 70, color: () => COLORS.low },
  { from: 0, to: 50, color: () => COLORS.veryLow },
];

/**
 * pLDDT of every residue along the sequence. The background shows the four
 * confidence bands; the curve is coloured by band. Hovering shows the
 * residue, clicking calls `onSelect(residueIndex)` (1-based).
 */
export class PlddtChart {
  constructor(container, { onSelect } = {}) {
    this.container = container;
    this.canvas = container.querySelector("canvas");
    this.tooltip = new Tooltip(container);
    this.onSelect = onSelect;
    this.residues = [];
    this.selected = null;
    this.#bindEvents();
    onResize(container, () => this.draw());
  }

  setResidues(residues) {
    this.residues = residues;
    this.draw();
  }

  select(index) {
    this.selected = index;
    this.draw();
  }

  #x(index, width) {
    const span = Math.max(this.residues.length - 1, 1);
    return PADDING.left + ((index - 1) / span) * (width - PADDING.left - PADDING.right);
  }

  #y(value) {
    return PADDING.top + (1 - value / 100) * (HEIGHT - PADDING.top - PADDING.bottom);
  }

  draw() {
    const { context: ctx, width } = setupCanvas(this.canvas, HEIGHT);
    ctx.clearRect(0, 0, width, HEIGHT);
    if (!this.residues.length) return;

    // Confidence bands
    for (const band of BANDS) {
      ctx.fillStyle = band.color();
      ctx.globalAlpha = 0.08;
      ctx.fillRect(PADDING.left, this.#y(band.to), width - PADDING.left - PADDING.right, this.#y(band.from) - this.#y(band.to));
    }
    ctx.globalAlpha = 1;

    // Axis labels and grid lines
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillStyle = COLORS.inkFaint;
    ctx.strokeStyle = COLORS.line;
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tick of [0, 50, 70, 90, 100]) {
      const y = this.#y(tick);
      ctx.fillText(String(tick), PADDING.left - 6, y);
      ctx.beginPath();
      ctx.moveTo(PADDING.left, y);
      ctx.lineTo(width - PADDING.right, y);
      ctx.stroke();
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const step = Math.max(10, Math.ceil(this.residues.length / 12 / 10) * 10);
    for (let index = 1; index <= this.residues.length; index += index === 1 ? step - 1 : step) {
      ctx.fillText(String(index), this.#x(index, width), HEIGHT - PADDING.bottom + 6);
    }

    // Curve, one segment per residue coloured by band
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    for (let k = 1; k < this.residues.length; k++) {
      const a = this.residues[k - 1];
      const b = this.residues[k];
      ctx.strokeStyle = plddtColor((a.plddt + b.plddt) / 2);
      ctx.beginPath();
      ctx.moveTo(this.#x(a.index, width), this.#y(a.plddt));
      ctx.lineTo(this.#x(b.index, width), this.#y(b.plddt));
      ctx.stroke();
    }

    if (this.selected !== null) {
      const residue = this.residues.find((r) => r.index === this.selected);
      if (residue) {
        const x = this.#x(residue.index, width);
        ctx.strokeStyle = COLORS.ink;
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, PADDING.top);
        ctx.lineTo(x, HEIGHT - PADDING.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = plddtColor(residue.plddt);
        ctx.beginPath();
        ctx.arc(x, this.#y(residue.plddt), 4.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  #residueAt(event) {
    if (!this.residues.length) return null;
    const rect = this.canvas.getBoundingClientRect();
    const plotWidth = rect.width - PADDING.left - PADDING.right;
    const ratio = (event.clientX - rect.left - PADDING.left) / plotWidth;
    if (ratio < -0.02 || ratio > 1.02) return null;
    const position = Math.round(Math.min(Math.max(ratio, 0), 1) * (this.residues.length - 1));
    return this.residues[position];
  }

  #bindEvents() {
    this.canvas.style.cursor = "crosshair";
    this.canvas.addEventListener("mousemove", (event) => {
      const residue = this.#residueAt(event);
      if (!residue) return this.tooltip.hide();
      const rect = this.canvas.getBoundingClientRect();
      this.tooltip.show(
        this.#x(residue.index, rect.width),
        this.#y(residue.plddt),
        `${residue.index} · ${residue.name} · pLDDT <b>${residue.plddt.toFixed(1)}</b>`,
      );
    });
    this.canvas.addEventListener("mouseleave", () => this.tooltip.hide());
    this.canvas.addEventListener("click", (event) => {
      const residue = this.#residueAt(event);
      if (residue && this.onSelect) this.onSelect(residue.index);
    });
  }
}
