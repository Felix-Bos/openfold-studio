/**
 * Job page.
 *
 * While the prediction runs: live progress. Once completed: the selected
 * sample is shown in 3D and its confidence analysis (fetched from the API)
 * drives the metric cards, band breakdown, low-confidence regions, the
 * per-residue chart, the sequence strip and the PDE heatmap. Selecting a
 * residue or region anywhere highlights it everywhere.
 */
import { bindRowLinks, setActiveButton } from "../lib/dom.js";
import { MatrixCanvas } from "../lib/matrix-canvas.js";
import { onPaletteChange, rampRgb, COLORS } from "../lib/palette.js";
import { ConfidenceView } from "../job/confidence-view.js";
import { PlddtChart } from "../job/plddt-chart.js";
import { initJobProgress } from "../job/progress.js";
import { SampleViewer } from "../job/sample-viewer.js";

function initResults(root) {
  const viewer = new SampleViewer(root.querySelector("[data-viewer]"));
  const rows = [...root.querySelectorAll("[data-sample-row]")];
  const styleButtons = [...root.querySelectorAll("[data-style]")];
  const field = (name) => root.querySelector(`[data-field="${name}"]`);
  const sequence = root.dataset.sequence;
  let residues = [];

  function selectResidue(index) {
    const residue = residues.find((r) => r.index === index);
    viewer.highlight(index, index, `${index} ${residue?.name ?? ""}`);
    chart.select(index);
    confidenceView.highlightResidue(index);
    pde.setHighlightedRow(index - 1);
    field("selection").textContent = residue
      ? `Residue ${index} (${sequence[index - 1]}) · pLDDT ${residue.plddt.toFixed(1)}`
      : `Residue ${index}`;
  }

  function selectRegion(start, end) {
    viewer.highlight(start, end, `${start}–${end}`);
    field("selection").textContent = `Residues ${start}–${end}`;
  }

  const chart = new PlddtChart(root.querySelector("[data-plddt-chart]"), { onSelect: selectResidue });
  const confidenceView = new ConfidenceView(root, {
    onSelectRegion: selectRegion,
    onSelectResidue: selectResidue,
  });
  // PDE: low error = strong blue, high error = pale. A square-root scale
  // keeps contrast when most errors are small and a few are large.
  const pde = new MatrixCanvas(root.querySelector("[data-pde-chart]"), {
    color: (value, max) => rampRgb(Math.sqrt(value / max), COLORS.veryHigh, COLORS.high, "#f4f5f7"),
    tooltip: (i, j, value) => `${i + 1} ↔ ${j + 1} · error <b>${value.toFixed(2)} Å</b>`,
    onClick: (i) => selectResidue(i + 1),
  });
  onPaletteChange(() => {
    pde.refresh();
    chart.draw();
  });

  async function selectRow(row) {
    rows.forEach((other) => other.classList.toggle("selected", other === row));
    field("title").textContent = `Sample ${row.dataset.sample} · rank ${row.dataset.rank}`;
    field("subtitle").textContent =
      row.dataset.rank === "01" ? `best ranked of ${rows.length} diffusion samples` : `ranked ${Number(row.dataset.rank)} of ${rows.length}`;
    field("selection").textContent = "Click the confidence chart or a region to locate residues.";
    viewer.show(row.dataset.structureUrl);

    const response = await fetch(row.dataset.confidenceUrl);
    if (!response.ok) return;
    const confidence = await response.json();
    residues = confidence.residues;
    confidenceView.render(confidence);
    chart.select(null);
    chart.setResidues(residues);
    if (confidence.pde) {
      pde.setMatrix(confidence.pde.values, confidence.pde.max);
      pde.setHighlightedRow(null);
      field("pde-max").textContent = `${confidence.pde.max.toFixed(1)} Å · uncertain`;
    }
  }

  rows.forEach((row) => {
    row.addEventListener("click", () => selectRow(row));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectRow(row);
      }
    });
  });

  // Ranking-score bars, scaled to the best score.
  const scoreBars = [...root.querySelectorAll("[data-score]")];
  const bestScore = Math.max(...scoreBars.map((bar) => Number(bar.dataset.score)), 1e-9);
  scoreBars.forEach((bar) => {
    bar.style.width = `${(Number(bar.dataset.score) / bestScore) * 100}%`;
  });

  styleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      setActiveButton(styleButtons, "style", button.dataset.style);
      viewer.setStyle(button.dataset.style);
    });
  });

  const spinButton = root.querySelector("[data-spin-toggle]");
  spinButton.addEventListener("click", () => spinButton.classList.toggle("active", viewer.toggleSpin()));

  selectRow(rows[0]);
}

const progressPanel = document.querySelector("[data-job-progress]");
if (progressPanel) initJobProgress(progressPanel, document.getElementById("status-pill"));

const results = document.querySelector("[data-job-results]");
if (results) initResults(results);

bindRowLinks();
