/**
 * Canvas helpers shared by the charts: crisp rendering on Retina screens
 * and a floating tooltip positioned over the chart.
 */

/**
 * Sizes `canvas` to its CSS box times the device pixel ratio and returns a
 * 2D context scaled so drawing code can use CSS pixels.
 */
export function setupCanvas(canvas, cssHeight) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.parentElement.clientWidth;
  const height = cssHeight ?? canvas.clientHeight;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

/** Tooltip element appended to a `.chart` container. */
export class Tooltip {
  constructor(container) {
    this.element = document.createElement("div");
    this.element.className = "chart-tooltip";
    this.element.hidden = true;
    container.appendChild(this.element);
  }

  show(x, y, html) {
    this.element.innerHTML = html;
    this.element.style.left = `${x}px`;
    this.element.style.top = `${y}px`;
    this.element.hidden = false;
  }

  hide() {
    this.element.hidden = true;
  }
}

/** Redraws `draw` whenever the element is resized (debounced to one frame). */
export function onResize(element, draw) {
  let frame = null;
  new ResizeObserver(() => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(draw);
  }).observe(element);
}
