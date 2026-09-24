import { createStructureViewer } from "../lib/structure-viewer.js";
import { COLORS, strengthColor } from "../lib/palette.js";

const baseStyle = () => ({ cartoon: { color: COLORS.neutral, opacity: 0.9 } });

/**
 * Predicted structure on which the attention of one residue is drawn:
 * the selected residue is highlighted, its strongest partners are coloured
 * by weight and linked to it by cylinders between alpha carbons.
 *
 * Residue numbers in 3Dmol (`resi`) start at 1; matrix indices start at 0.
 */
export class AttentionStructureView {
  constructor(container, cifText, sequence, { onHover = () => {} } = {}) {
    this.sequence = sequence;
    this.viewer = createStructureViewer(container);
    this.viewer.addModel(cifText, "cif");
    this.viewer.setStyle({}, baseStyle());
    this.viewer.zoomTo();
    this.viewer.render();
    this.alphaCarbons = new Map();
    this.hoverLabel = null;
    this.viewer.setHoverable(
      {},
      true,
      (atom) => this.#showHover(atom, onHover),
      () => this.#hideHover(),
    );
  }

  /** "119 (R)": residue number and one-letter amino acid. */
  residueLabel(resi) {
    return `${resi} (${this.sequence[resi - 1] || "?"})`;
  }

  /** Draws the links from `selectedIndex` to each `{ index, normalized }` partner. */
  showLinks(selectedIndex, partners) {
    this.viewer.removeAllShapes();
    this.viewer.setStyle({}, baseStyle());

    const source = this.#alphaCarbon(selectedIndex + 1);
    for (const partner of partners) {
      const color = strengthColor(partner.normalized);
      this.viewer.setStyle({ resi: partner.index + 1 }, { cartoon: { color } });
      const target = this.#alphaCarbon(partner.index + 1);
      if (!source || !target) continue;
      this.viewer.addCylinder({
        start: source,
        end: target,
        radius: 0.08 + partner.normalized * 0.08,
        color,
        alpha: 0.75,
        fromCap: 1,
        toCap: 1,
      });
    }

    this.viewer.setStyle({ resi: selectedIndex + 1 }, { cartoon: { color: COLORS.veryHigh } });
    if (source) this.viewer.addSphere({ center: source, radius: 0.75, color: COLORS.veryHigh, alpha: 0.85 });
    this.viewer.render();
  }

  clearLinks() {
    this.viewer.removeAllShapes();
    this.viewer.setStyle({}, baseStyle());
    this.viewer.render();
  }

  #alphaCarbon(resi) {
    if (!this.alphaCarbons.has(resi)) {
      const atoms = this.viewer.selectedAtoms({ resi });
      const atom = atoms.find((a) => a.atom === "CA") || atoms[0];
      this.alphaCarbons.set(resi, atom ? { x: atom.x, y: atom.y, z: atom.z } : null);
    }
    return this.alphaCarbons.get(resi);
  }

  #showHover(atom, onHover) {
    if (!atom.resi) return;
    if (this.hoverLabel) this.viewer.removeLabel(this.hoverLabel);
    this.hoverLabel = this.viewer.addLabel(this.residueLabel(atom.resi), {
      position: { x: atom.x, y: atom.y, z: atom.z },
      backgroundColor: COLORS.elevated,
      backgroundOpacity: 0.88,
      fontColor: COLORS.ink,
      fontSize: 14,
      borderColor: COLORS.accent,
      borderThickness: 1,
      inFront: true,
    });
    onHover(atom.resi);
    this.viewer.render();
  }

  #hideHover() {
    if (!this.hoverLabel) return;
    this.viewer.removeLabel(this.hoverLabel);
    this.hoverLabel = null;
    this.viewer.render();
  }
}
