/**
 * Polls a JSON status endpoint until the job it describes is no longer active.
 *
 * The backend status endpoints all return `{ is_active, ... }`. `onUpdate`
 * receives every payload; once `is_active` is false, `onDone` is called
 * (by default the page reloads to show the final state).
 */
export function pollStatus(url, {
  onUpdate = () => {},
  onDone = () => window.location.reload(),
  intervalMs = 2000,
  retryMs = 3000,
} = {}) {
  async function tick() {
    try {
      const response = await fetch(url);
      const status = await response.json();
      onUpdate(status);
      if (!status.is_active) {
        onDone(status);
        return;
      }
      setTimeout(tick, intervalMs);
    } catch {
      setTimeout(tick, retryMs);
    }
  }
  tick();
}
