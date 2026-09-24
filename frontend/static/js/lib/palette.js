/**
 * Colours for canvases and the 3D viewer, read from the CSS design tokens
 * (css/tokens.css) so charts follow the light/dark theme. Values are cached
 * and refreshed when the theme changes.
 */

function readVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function load() {
  return {
    veryHigh: readVar("--c-vhigh"),
    high: readVar("--c-high"),
    low: readVar("--c-low"),
    veryLow: readVar("--c-vlow"),
    accent: readVar("--accent"),
    accent2: readVar("--accent-2"),
    ink: readVar("--ink"),
    inkSoft: readVar("--ink-soft"),
    inkFaint: readVar("--ink-faint"),
    line: readVar("--line"),
    panel: readVar("--bg-panel"),
    sunken: readVar("--bg-sunken"),
    elevated: readVar("--bg-elevated"),
    heatLow: readVar("--heat-low"),
    heatMid: readVar("--heat-mid"),
    heatHigh: readVar("--heat-high"),
    success: readVar("--success"),
    neutral: "#8a8f9c",
  };
}

export let COLORS = load();
const listeners = new Set();

/** Registers a redraw callback run after the theme (and thus the palette) changes. */
export function onPaletteChange(callback) {
  listeners.add(callback);
}

function refresh() {
  COLORS = load();
  listeners.forEach((callback) => callback());
}
document.addEventListener("themechange", refresh);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", refresh);

/** Confidence band of a pLDDT value, as used by the CSS `[data-band]` selectors. */
export function plddtBand(plddt) {
  if (plddt >= 90) return "very_high";
  if (plddt >= 70) return "confident";
  if (plddt >= 50) return "low";
  return "very_low";
}

/** pLDDT to colour, AlphaFold database bands. */
export function plddtColor(plddt) {
  return {
    very_high: COLORS.veryHigh,
    confident: COLORS.high,
    low: COLORS.low,
    very_low: COLORS.veryLow,
  }[plddtBand(plddt)];
}

/** Normalised weight (0-1) to colour for attention links and cube points. */
export function strengthColor(value, { strong = 0.7, medium = 0.4 } = {}) {
  if (value > strong) return COLORS.veryLow;
  if (value > medium) return COLORS.low;
  return COLORS.high;
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  const full = value.length === 3 ? [...value].map((c) => c + c).join("") : value;
  return [0, 2, 4].map((offset) => parseInt(full.slice(offset, offset + 2), 16));
}

function mix(a, b, t) {
  return a.map((channel, i) => Math.round(channel + (b[i] - channel) * t));
}

/** Two-segment colour ramp: `value` in [0, 1] mapped low → mid → high. */
export function rampRgb(value, low = COLORS.heatLow, mid = COLORS.heatMid, high = COLORS.heatHigh) {
  const v = Math.min(Math.max(value, 0), 1);
  return v < 0.5 ? mix(hexToRgb(low), hexToRgb(mid), v * 2) : mix(hexToRgb(mid), hexToRgb(high), (v - 0.5) * 2);
}

/** Attention heatmap colour: dark blue (weak) to orange (strong). */
export function heatmapRgb(value) {
  return rampRgb(value);
}
