/* global $3Dmol */
/**
 * Creates a 3Dmol.js viewer (the library is loaded globally from static/vendor)
 * with a transparent background, so the page's theme shows through.
 */
export function createStructureViewer(container) {
  return $3Dmol.createViewer(container, { backgroundColor: "white", backgroundAlpha: 0 });
}
