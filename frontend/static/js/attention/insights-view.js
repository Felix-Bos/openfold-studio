import { formatPercent } from "../lib/format.js";

/**
 * Renders the statistics of one attention block: where its attention goes
 * (self / local / long range), how focused it is, the attention mass per
 * sequence-separation bin, and the strongest long-range links with their
 * distance in the predicted structure.
 */
export class InsightsView {
  constructor(refs, { residueLabel, onSelectLink }) {
    this.refs = refs;
    this.residueLabel = residueLabel;
    this.onSelectLink = onSelectLink;
  }

  render(data, block) {
    this.#renderMetrics(data.stats);
    this.#renderSinks(data.stats.sinks);
    this.#renderSeparation(data.stats.separation_profile);
    this.#renderLinks(data.links, data.contact_precision, block);
    this.refs["insights-sub"].textContent = `block ${block} · share of the total attention weight`;
  }

  #renderMetrics(stats) {
    const focus = 1 - stats.entropy / stats.max_entropy;
    const metric = (label, value, hint) =>
      `<div class="mini-metric"><span class="label">${label}</span><span class="value">${value}</span><span class="hint">${hint}</span></div>`;
    this.refs["block-metrics"].innerHTML = [
      metric("Focus", formatPercent(focus, 0), focus > 0.3 ? "concentrated" : focus > 0.12 ? "moderate" : "diffuse"),
      metric("Self", formatPercent(stats.diagonal_mass), "i = j"),
      metric("Local", formatPercent(stats.local_mass), "|i − j| ≤ 3"),
      metric("Long range", formatPercent(stats.long_range_mass), "|i − j| ≥ 24"),
    ].join("");
  }

  #renderSinks(sinks = []) {
    const box = this.refs.sinks;
    box.hidden = !sinks.length;
    if (!sinks.length) return;
    const list = sinks
      .map((sink) => `<strong>${this.residueLabel(sink.index + 1)}</strong> (${formatPercent(sink.share)} of all attention, ${sink.fold_over_uniform.toFixed(0)}× uniform)`)
      .join(", ");
    box.innerHTML = `<div><strong>Attention sink${sinks.length > 1 ? "s" : ""}:</strong> ${list}.
      Most residues park attention there regardless of structure, so links to ${sinks.length > 1 ? "these residues" : "it"} are rarely meaningful contacts.</div>`;
  }

  #renderSeparation(profile) {
    const max = Math.max(...profile.map((bin) => bin.mass), 1e-9);
    this.refs.separation.innerHTML = profile
      .map(
        (bin) => `<div class="separation-row">
          <span class="label">${bin.label}</span>
          <span class="track"><span style="width:${(bin.mass / max) * 100}%"></span></span>
          <span class="value">${formatPercent(bin.mass)}</span>
        </div>`,
      )
      .join("");
  }

  #renderLinks(links, precision, block) {
    const callout = this.refs["contact-callout"];
    if (precision === null || precision === undefined) {
      callout.hidden = true;
    } else {
      callout.hidden = false;
      const contacts = links.filter((link) => link.is_contact).length;
      callout.innerHTML = `<div><strong>${contacts} of ${links.length}</strong> strongest long-range links in block ${block} are
        real 3D contacts (alpha carbons &lt; 8 Å apart): <strong>${formatPercent(precision, 0)}</strong> contact precision.</div>`;
    }

    this.refs.links.innerHTML = links
      .map(
        (link) => `<tr data-residue="${link.i}">
          <td class="pair">${this.residueLabel(link.i + 1)} ↔ ${this.residueLabel(link.j + 1)}</td>
          <td class="num">${link.separation}</td>
          <td class="num">${link.weight.toFixed(3)}</td>
          <td class="num">${link.distance !== undefined ? `${link.distance.toFixed(1)} Å` : "—"}</td>
          <td>${link.is_contact === undefined ? "" : link.is_contact ? '<span class="badge badge--success">contact</span>' : '<span class="badge">no</span>'}</td>
        </tr>`,
      )
      .join("");
    this.refs.links.querySelectorAll("tr").forEach((row) => {
      row.addEventListener("click", () => this.onSelectLink(Number(row.dataset.residue)));
    });
  }
}
