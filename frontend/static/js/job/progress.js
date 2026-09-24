import { formatDuration } from "../lib/format.js";
import { pollStatus } from "../lib/polling.js";

// Order of the pipeline stages shown as steps.
const STEP_ORDER = ["running_msa", "running_inference", "completed"];

/**
 * Live progress panel of a running prediction. Polls the job status endpoint
 * and reloads the page once the job is finished.
 *
 * When the backend cannot measure progress (`progress_percent` is null, e.g.
 * during inference) the bar switches to an indeterminate animation and shows
 * the elapsed time instead of an ETA.
 */
export function initJobProgress(panel, statusPill) {
  const step = panel.querySelector("[data-progress-step]");
  const eta = panel.querySelector("[data-progress-eta]");
  const fill = panel.querySelector("[data-progress-fill]");
  const steps = [...panel.querySelectorAll("[data-step]")];

  function renderSteps(status) {
    const current = STEP_ORDER.indexOf(status);
    steps.forEach((item) => {
      const position = STEP_ORDER.indexOf(item.dataset.step);
      item.classList.toggle("done", current > position);
      item.classList.toggle("current", current === position);
    });
  }

  function render(status) {
    step.textContent = status.current_step || "Waiting to start…";
    statusPill.textContent = status.status_display;
    statusPill.className = `status-pill ${status.status}`;
    renderSteps(status.status);

    const indeterminate = status.progress_percent === null || status.progress_percent === undefined;
    fill.classList.toggle("indeterminate", indeterminate);
    if (indeterminate) {
      fill.style.width = "";
      eta.textContent = status.elapsed_seconds ? `elapsed ${formatDuration(status.elapsed_seconds)}` : "";
    } else {
      fill.style.width = `${status.progress_percent}%`;
      eta.textContent = status.eta_seconds != null ? `~${formatDuration(status.eta_seconds)} left` : "";
    }
  }

  pollStatus(panel.dataset.statusUrl, { onUpdate: render });
}
