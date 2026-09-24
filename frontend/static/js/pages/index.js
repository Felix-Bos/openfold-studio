/**
 * Home page: example buttons fill the sequence field, and a live analysis
 * (length, mass, charge, composition) is shown while typing.
 */
import { bindRowLinks } from "../lib/dom.js";
import { formatPercent } from "../lib/format.js";
import { analyzeSequence, cleanSequence } from "../home/sequence-analysis.js";

const form = document.querySelector("[data-sequence-form]");

if (form) {
  const textarea = form.querySelector("textarea");
  const panel = form.querySelector("[data-sequence-analysis]");
  const stat = (name) => panel.querySelector(`[data-stat="${name}"]`);
  const bar = panel.querySelector("[data-composition-bar]");
  const legend = panel.querySelector("[data-composition-legend]");

  function render() {
    const sequence = cleanSequence(textarea.value);
    panel.hidden = sequence.length === 0;
    if (!sequence.length) return;

    const analysis = analyzeSequence(sequence);
    stat("length").textContent = `${analysis.length} aa`;
    stat("mass").textContent = `${analysis.massKda.toFixed(1)} kDa`;
    stat("charge").textContent = analysis.charge > 0 ? `+${analysis.charge}` : `${analysis.charge}`;
    stat("invalid").textContent = analysis.invalid.length ? analysis.invalid.join(" ") : "none";
    stat("invalid").classList.toggle("bad", analysis.invalid.length > 0);

    bar.innerHTML = analysis.groups
      .map((g) => `<span style="width:${g.fraction * 100}%;background:${g.color}" title="${g.label}"></span>`)
      .join("");
    legend.innerHTML = analysis.groups
      .map((g) => `<li><i style="background:${g.color}"></i>${g.label} <b>${formatPercent(g.fraction, 0)}</b></li>`)
      .join("");
  }

  form.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      textarea.value = button.dataset.example;
      textarea.removeAttribute("aria-invalid");
      render();
      textarea.focus();
    });
  });

  textarea.addEventListener("input", render);
  render();
}

bindRowLinks();
