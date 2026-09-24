/** Light / dark theme toggle, remembered in localStorage. */
const toggle = document.querySelector("[data-theme-toggle]");

function currentTheme() {
  const explicit = document.documentElement.dataset.theme;
  if (explicit) return explicit;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

toggle?.addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("theme", next);
  } catch {
    /* storage unavailable: the choice only lasts for this page */
  }
  document.dispatchEvent(new CustomEvent("themechange", { detail: next }));
});
