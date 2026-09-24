import { COLORS, strengthColor } from "../lib/palette.js";

const DRAG_SENSITIVITY = 0.01; // radians per pixel

/**
 * Point cloud of a triangle-attention cube [I, J, K], drawn on a 2D canvas
 * with a rotation (around Y then X) and an orthographic projection. Points
 * are painted back to front. Drag with the mouse to rotate.
 */
export class CubeView {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.data = null;
    this.rotation = { x: -0.5, y: 0.6 };
    this.#bindDrag();
  }

  setData(data) {
    this.data = data;
    this.draw();
  }

  draw() {
    const { width, height } = this.canvas;
    const ctx = this.context;
    ctx.fillStyle = COLORS.sunken;
    ctx.fillRect(0, 0, width, height);
    if (!this.data) return;

    const [I, J, K] = this.data.shape;
    const center = { x: width / 2, y: height / 2 };
    const scale = (Math.min(width, height) * 0.38) / ((Math.max(I, J, K) || 1) / 2);

    const projected = this.data.points
      .map(([i, j, k, value]) => ({ ...this.#project(i - I / 2, j - J / 2, k - K / 2, center, scale), value }))
      .sort((a, b) => a.depth - b.depth);

    for (const point of projected) {
      ctx.fillStyle = strengthColor(point.value, { strong: 0.85, medium: 0.7 });
      ctx.beginPath();
      ctx.arc(point.x, point.y, 1.5 + point.value * 1.5, 0, Math.PI * 2);
      ctx.fill();
    }

    if (projected.length === 0) {
      ctx.fillStyle = COLORS.neutral;
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No voxel above the threshold: lower it", center.x, center.y);
    }
  }

  #project(x, y, z, center, scale) {
    const cosY = Math.cos(this.rotation.y), sinY = Math.sin(this.rotation.y);
    const cosX = Math.cos(this.rotation.x), sinX = Math.sin(this.rotation.x);
    const x1 = x * cosY - z * sinY;
    const z1 = x * sinY + z * cosY;
    const y1 = y * cosX - z1 * sinX;
    const depth = y * sinX + z1 * cosX;
    return { x: center.x + x1 * scale, y: center.y + y1 * scale, depth };
  }

  #bindDrag() {
    let last = null;
    this.canvas.addEventListener("mousedown", (event) => {
      last = { x: event.clientX, y: event.clientY };
      this.canvas.classList.add("grabbing");
    });
    window.addEventListener("mouseup", () => {
      last = null;
      this.canvas.classList.remove("grabbing");
    });
    window.addEventListener("mousemove", (event) => {
      if (!last) return;
      this.rotation.y += (event.clientX - last.x) * DRAG_SENSITIVITY;
      this.rotation.x += (event.clientY - last.y) * DRAG_SENSITIVITY;
      last = { x: event.clientX, y: event.clientY };
      this.draw();
    });
  }
}
