/** Small DOM helpers shared by the page scripts. */

/** Parses a `{{ value|json_script:"id" }}` block rendered by Django. */
export function readJsonScript(id) {
  const element = document.getElementById(id);
  return element ? JSON.parse(element.textContent) : null;
}

/** Marks the button whose `data-<key>` equals `value` as active within `buttons`. */
export function setActiveButton(buttons, key, value) {
  buttons.forEach((button) => button.classList.toggle("active", button.dataset[key] === value));
}

/** Makes every `[data-href]` element (e.g. a table row) navigate on click. */
export function bindRowLinks(root = document) {
  root.querySelectorAll("[data-href]").forEach((row) => {
    row.addEventListener("click", (event) => {
      if (event.target.closest("a, button")) return;
      window.location.href = row.dataset.href;
    });
  });
}
