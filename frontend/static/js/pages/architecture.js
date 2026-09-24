/* global katex */
/**
 * "Understand the model" page: clicking an element of the diagrams opens a
 * drawer explaining it (content in architecture/details.js).
 */
import { DETAILS } from "../architecture/details.js";

const overlay = document.getElementById("detail-overlay");
const titleEl = document.getElementById("detail-title");
const sourceEl = document.getElementById("detail-source");
const bodyEl = document.getElementById("detail-body");

function openDetail(key) {
  const detail = DETAILS[key];
  if (!detail) return;
  titleEl.textContent = detail.title;
  sourceEl.innerHTML = `<span class="source-ref">${detail.source}</span>`;
  bodyEl.innerHTML = detail.html;
  overlay.classList.add("open");
  document.body.style.overflow = "hidden";

  bodyEl.querySelectorAll(".formula-block").forEach((el) => {
    const tex = el.textContent.trim().replace(/^\$\$|\$\$$/g, "");
    try {
      katex.render(tex, el, { displayMode: true, throwOnError: false });
    } catch (e) {
      el.textContent = tex;
    }
  });
}

function closeDetail() {
  overlay.classList.remove("open");
  document.body.style.overflow = "";
}

document.querySelectorAll("[data-detail]").forEach((el) => {
  el.addEventListener("click", () => openDetail(el.dataset.detail));
});

document.getElementById("detail-close").addEventListener("click", closeDetail);
overlay.addEventListener("click", (e) => {
  if (e.target === overlay) closeDetail();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && overlay.classList.contains("open")) closeDetail();
});
