import { escapeHtml, formatPercent } from "../lib/format.js";
import { plddtBand } from "../lib/palette.js";

const BAND_CLASS = (key) => `band-${key}`;

/**
 * Renders the confidence analysis returned by
 * /api/jobs/<id>/samples/<n>/confidence/ into the job page: verdict,
 * metric cards, band breakdown, low-confidence regions and sequence strip.
 */
export class ConfidenceView {
  constructor(root, { onSelectRegion, onSelectResidue }) {
    this.root = root;
    this.onSelectRegion = onSelectRegion;
    this.onSelectResidue = onSelectResidue;
    this.field = (name) => root.querySelector(`[data-field="${name}"]`);
  }

  render(confidence) {
    const { scores } = confidence;
    this.field("verdict").textContent = confidence.verdict;
    this.field("plddt").textContent = scores.avg_plddt?.toFixed(1) ?? "—";
    this.field("plddt-hint").textContent = this.#bandLabel(scores.avg_plddt);
    this.field("ptm").textContent = scores.ptm?.toFixed(3) ?? "—";
    this.field("gpde").innerHTML = scores.gpde != null ? `${scores.gpde.toFixed(2)}<small> Å</small>` : "—";
    this.field("disorder").textContent = scores.disorder != null ? formatPercent(scores.disorder) : "—";
    this.field("clash").textContent = scores.has_clash ? "Yes" : "None";
    this.field("clash").style.color = scores.has_clash ? "var(--danger)" : "var(--success)";
    this.#renderBands(confidence.bands);
    this.#renderRegions(confidence.low_confidence_regions, confidence.residues.length);
    this.#renderSequence(confidence.residues);
  }

  highlightResidue(index) {
    this.root.querySelectorAll("[data-sequence-strip] span").forEach((span) => {
      span.classList.toggle("active", Number(span.dataset.index) === index);
    });
  }

  #bandLabel(plddt) {
    if (plddt == null) return "local confidence";
    return {
      very_high: "very high confidence",
      confident: "confident",
      low: "low confidence",
      very_low: "very low confidence",
    }[plddtBand(plddt)];
  }

  #renderBands(bands) {
    this.root.querySelector("[data-bands-bar]").innerHTML = bands
      .map((b) => `<span class="${BAND_CLASS(b.key)}" style="width:${b.fraction * 100}%" title="${b.label}"></span>`)
      .join("");
    this.root.querySelector("[data-bands-legend]").innerHTML = bands
      .map(
        (b) =>
          `<li><span class="swatch ${BAND_CLASS(b.key)}"></span>${escapeHtml(b.label)}<span class="value">${formatPercent(b.fraction)}</span></li>`,
      )
      .join("");
  }

  #renderRegions(regions, length) {
    const list = this.root.querySelector("[data-regions]");
    if (!regions.length) {
      list.innerHTML = `<li class="empty">✓ No low-confidence region: the whole chain is predicted with pLDDT ≥ 70 (or only isolated residues are below).</li>`;
      return;
    }
    list.innerHTML = regions
      .map(
        (r) => `<li><button type="button" data-start="${r.start}" data-end="${r.end}">
          <span class="range">${r.start}–${r.end}</span>
          <span class="region-bar"><span style="left:${((r.start - 1) / length) * 100}%;width:${Math.max(((r.end - r.start + 1) / length) * 100, 1)}%"></span></span>
          <span class="plddt" data-band="${plddtBand(r.mean_plddt)}">${r.mean_plddt.toFixed(1)}</span>
        </button></li>`,
      )
      .join("");
    list.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => this.onSelectRegion(Number(button.dataset.start), Number(button.dataset.end)));
    });
  }

  #renderSequence(residues) {
    const strip = this.root.querySelector("[data-sequence-strip]");
    const letters = this.root.dataset.sequence;
    strip.innerHTML = residues
      .map(
        (r) =>
          `<span class="${BAND_CLASS(plddtBand(r.plddt))}" data-index="${r.index}" title="${r.index} ${r.name} · pLDDT ${r.plddt.toFixed(1)}">${letters[r.index - 1] ?? "?"}</span>`,
      )
      .join("");
    strip.querySelectorAll("span").forEach((span) => {
      span.addEventListener("click", () => this.onSelectResidue(Number(span.dataset.index)));
    });
  }
}
