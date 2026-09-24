/**
 * Attention explorer page.
 *
 * State: the selected layer family, block and residue, plus the two
 * thresholds. Every control updates the state then calls `refresh()` (new
 * block data) or `renderLinks()` (same data, new selection). The family
 * cards, map, 3D links, block insights and entropy chart all read the same
 * state, so they always stay in sync.
 *
 * While the extraction is still running, the page only polls its status and
 * reloads when it is done.
 */
import { readJsonScript } from "../lib/dom.js";
import { formatPercent } from "../lib/format.js";
import { MatrixCanvas } from "../lib/matrix-canvas.js";
import { heatmapRgb, onPaletteChange } from "../lib/palette.js";
import { pollStatus } from "../lib/polling.js";
import { AttentionApi } from "../attention/api.js";
import { connectedResidues, strongestResidue } from "../attention/analysis.js";
import { CubeView } from "../attention/cube-view.js";
import { EntropyChart } from "../attention/entropy-chart.js";
import { InsightsView } from "../attention/insights-view.js";
import { AttentionStructureView } from "../attention/structure-view.js";

const THRESHOLD_DEBOUNCE_MS = 200;

function initExplorer(root) {
  const sequence = readJsonScript("job-sequence");
  const cubeFamilies = new Set(readJsonScript("cube-families"));
  const api = new AttentionApi(root.dataset.familiesUrl);

  const refs = {};
  root.querySelectorAll("[data-ref]").forEach((element) => {
    refs[element.dataset.ref] = element;
  });
  const familyCards = [...root.querySelectorAll("[data-family]")];

  const state = {
    family: null,
    block: 0,
    view: "heatmap", // "heatmap" | "cube"
    data: null, // current block: { weights, stats, links, contact_precision }
    selected: null, // selected residue index (0-based)
    linkThreshold: parseFloat(refs["link-threshold"].value),
    cubeThreshold: parseFloat(refs["cube-threshold"].value),
  };
  let summaries = {};

  const structure = new AttentionStructureView(refs.viewer, readJsonScript("best-structure"), sequence);
  const residueLabel = (resi) => structure.residueLabel(resi);

  const map = new MatrixCanvas(refs.heatmap, {
    color: (value, max) => heatmapRgb(Math.sqrt(value / max)),
    tooltip: (row, col, value) => `${residueLabel(row + 1)} → ${residueLabel(col + 1)} · <b>${value.toFixed(4)}</b>`,
    onClick: (row) => selectResidue(row),
  });
  const cube = new CubeView(refs.cube);
  const entropy = new EntropyChart(refs["entropy-chart"], { onSelect: (block) => setBlock(block) });
  const insights = new InsightsView(refs, { residueLabel, onSelectLink: (i) => selectResidue(i) });
  onPaletteChange(() => {
    map.refresh();
    entropy.draw();
    cube.draw();
  });

  // --- Rendering -------------------------------------------------------------

  function renderLinks() {
    if (!state.data || state.selected === null) {
      structure.clearLinks();
      return;
    }
    const row = state.data.weights[state.selected];
    const partners = connectedResidues(row, state.selected, state.linkThreshold);
    structure.showLinks(state.selected, partners);
    map.setHighlightedRow(state.selected);
    refs["viewer-title"].textContent = `Attention of residue ${residueLabel(state.selected + 1)}`;
    refs["viewer-sub"].textContent = `${partners.length} strongest partners · block ${state.block}`;
  }

  function renderFamilyCards() {
    familyCards.forEach((card) => {
      const summary = summaries[card.dataset.family]?.summary;
      if (!summary) return;
      const focus = 1 - summary.mean_entropy / summary.max_entropy;
      card.querySelector("[data-family-stats]").innerHTML = `
        <span class="badge">focus ${formatPercent(focus, 0)}</span>
        <span class="badge">long range ${formatPercent(summary.mean_long_range_mass, 0)}</span>
        <span class="badge">sharpest block ${summary.most_focused_block}</span>`;
    });
  }

  async function refresh() {
    entropy.setCurrent(state.block);
    if (state.view === "cube") {
      cube.setData(await api.cubePoints(state.family, state.block, state.cubeThreshold));
      return;
    }
    const data = await api.block(state.family, state.block);
    if (!data) return;
    state.data = data;
    if (state.selected === null) state.selected = strongestResidue(data.weights);
    map.setMatrix(data.weights);
    insights.render(data, state.block);
    renderLinks();
  }

  // --- Actions ---------------------------------------------------------------

  function selectResidue(index) {
    state.selected = index;
    renderLinks();
  }

  function setBlock(block) {
    state.block = block;
    refs["block-slider"].value = block;
    refs["block-value"].textContent = block;
    state.selected = null;
    refresh();
  }

  function setView(view) {
    state.view = view;
    refs["view-toggle"].querySelectorAll("[data-view]").forEach((button) => {
      button.classList.toggle("active", button.dataset.view === view);
    });
    refs.heatmap.hidden = view !== "heatmap";
    refs.cube.hidden = view !== "cube";
    refs["cube-controls"].hidden = view !== "cube";
  }

  function selectFamily(family) {
    state.family = family;
    familyCards.forEach((card) => card.classList.toggle("active", card.dataset.family === family));

    const hasCube = cubeFamilies.has(family);
    refs["view-toggle"].hidden = !hasCube;
    if (!hasCube) setView("heatmap");

    const summary = summaries[family];
    if (!summary) return;
    const blockCount = summary.shape[0];
    refs["block-slider"].max = Math.max(blockCount - 1, 0);
    entropy.setData(summary.summary.entropy_per_block, summary.summary.max_entropy, state.block);
    setBlock(Math.min(state.block, blockCount - 1));
  }

  // --- Controls --------------------------------------------------------------

  familyCards.forEach((card) => {
    if (!card.disabled) card.addEventListener("click", () => selectFamily(card.dataset.family));
  });

  refs["view-toggle"].querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => {
      setView(button.dataset.view);
      refresh();
    });
  });

  refs["block-slider"].addEventListener("input", (event) => setBlock(parseInt(event.target.value, 10)));

  refs["link-threshold"].addEventListener("input", (event) => {
    state.linkThreshold = parseFloat(event.target.value);
    refs["link-threshold-value"].textContent = state.linkThreshold.toFixed(2);
    renderLinks();
  });

  let debounce = null;
  refs["cube-threshold"].addEventListener("input", (event) => {
    state.cubeThreshold = parseFloat(event.target.value);
    refs["cube-threshold-value"].textContent = state.cubeThreshold.toFixed(2);
    clearTimeout(debounce);
    debounce = setTimeout(refresh, THRESHOLD_DEBOUNCE_MS);
  });

  // --- Start -------------------------------------------------------------------

  api.families().then((families) => {
    summaries = families;
    renderFamilyCards();
    const first = familyCards.find((card) => !card.disabled && summaries[card.dataset.family]);
    if (first) selectFamily(first.dataset.family);
  });
}

const explorer = document.querySelector("[data-attention-explorer]");
if (explorer) initExplorer(explorer);

const pending = document.querySelector("[data-attention-pending]");
if (pending) pollStatus(pending.dataset.statusUrl, { intervalMs: 3000, retryMs: 5000 });
