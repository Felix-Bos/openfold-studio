/** Formats a number of seconds as "m:ss". */
export function formatDuration(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

/** 0.4567 -> "45.7%". */
export function formatPercent(fraction, digits = 1) {
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** Escapes text before inserting it into HTML. */
export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
