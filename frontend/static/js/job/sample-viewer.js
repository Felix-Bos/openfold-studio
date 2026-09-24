import { createStructureViewer } from "../lib/structure-viewer.js";
import { COLORS, plddtColor } from "../lib/palette.js";

const colorByPlddt = (atom) => plddtColor(atom.b);

// 3Dmol style definition for each display mode offered by the page.
const STYLES = {
  cartoon: () => ({ cartoon: { colorfunc: colorByPlddt } }),
  stick: () => ({ stick: { colorfunc: colorByPlddt, radius: 0.15 } }),
  sphere: () => ({ sphere: { colorfunc: colorByPlddt, scale: 0.35 } }),
};

/**
 * 3D view of one predicted sample, coloured by per-residue confidence
 * (OpenFold writes pLDDT into the B-factor column of the mmCIF file).
 * Structures are downloaded once per sample and cached.
 */
export class SampleViewer {
  constructor(container) {
    this.viewer = createStructureViewer(container);
    this.style = "cartoon";
    this.spinning = false;
    this.cache = new Map();
    this.label = null;
  }

  async show(structureUrl) {
    if (!this.cache.has(structureUrl)) {
      const response = await fetch(structureUrl);
      this.cache.set(structureUrl, await response.text());
    }
    this.viewer.clear();
    this.label = null;
    this.viewer.addModel(this.cache.get(structureUrl), "cif");
    this.setStyle(this.style);
    this.viewer.zoomTo();
    this.viewer.render();
    this.viewer.spin(this.spinning ? "y" : false);
  }

  setStyle(style) {
    this.style = style;
    this.viewer.setStyle({}, STYLES[style]());
    this.viewer.render();
  }

  toggleSpin() {
    this.spinning = !this.spinning;
    this.viewer.spin(this.spinning ? "y" : false);
    return this.spinning;
  }

  /** Outlines residues `start`..`end` (1-based, inclusive), labels them and zooms in. */
  highlight(start, end = start, text = null) {
    this.setStyle(this.style);
    this.viewer.removeAllLabels();
    const selection = { resi: `${start}-${end}` };
    this.viewer.addStyle(selection, { stick: { colorscheme: "whiteCarbon", radius: 0.22 } });
    const atoms = this.viewer.selectedAtoms({ ...selection, atom: "CA" });
    const anchor = atoms[Math.floor(atoms.length / 2)];
    if (anchor) {
      this.viewer.addLabel(text ?? `${start}${end !== start ? `–${end}` : ""}`, {
        position: { x: anchor.x, y: anchor.y, z: anchor.z },
        backgroundColor: COLORS.elevated,
        backgroundOpacity: 0.92,
        fontColor: COLORS.ink,
        borderColor: COLORS.accent,
        borderThickness: 1,
        fontSize: 13,
        inFront: true,
      });
    }
    this.viewer.zoomTo(selection, 600);
    this.viewer.render();
  }

  resetView() {
    this.viewer.removeAllLabels();
    this.setStyle(this.style);
    this.viewer.zoomTo({}, 600);
  }
}
