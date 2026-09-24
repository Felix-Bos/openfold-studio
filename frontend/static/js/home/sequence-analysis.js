/**
 * Pure computations on a raw protein sequence, used for the live preview
 * under the sequence field. Values are approximate and only for guidance.
 */

// Average residue masses (Da) in a peptide chain (amino acid minus water).
const RESIDUE_MASS = {
  A: 71.08, R: 156.19, N: 114.1, D: 115.09, C: 103.14, E: 129.12, Q: 128.13, G: 57.05, H: 137.14,
  I: 113.16, L: 113.16, K: 128.17, M: 131.19, F: 147.18, P: 97.12, S: 87.08, T: 101.1, W: 186.21,
  Y: 163.18, V: 99.13, U: 150.04, O: 237.3,
};
const WATER_MASS = 18.02;

/** Physico-chemical groups shown in the composition bar. */
export const GROUPS = [
  { key: "hydrophobic", label: "Hydrophobic", residues: "AVILMFWC", color: "var(--c-high)" },
  { key: "polar", label: "Polar", residues: "STNQYGPH", color: "var(--accent-2)" },
  { key: "positive", label: "Positive", residues: "KR", color: "var(--c-vhigh)" },
  { key: "negative", label: "Negative", residues: "DE", color: "var(--c-vlow)" },
];

const VALID = new Set("ACDEFGHIKLMNPQRSTVWYXBZJUO");

export function cleanSequence(raw) {
  return raw.replace(/\s+/g, "").toUpperCase();
}

export function analyzeSequence(sequence) {
  const counts = {};
  for (const residue of sequence) counts[residue] = (counts[residue] || 0) + 1;

  const invalid = Object.keys(counts).filter((residue) => !VALID.has(residue));
  const mass = sequence.length
    ? [...sequence].reduce((sum, residue) => sum + (RESIDUE_MASS[residue] ?? 110), WATER_MASS)
    : 0;
  const charge = (counts.K || 0) + (counts.R || 0) - (counts.D || 0) - (counts.E || 0);
  const groups = GROUPS.map((group) => ({
    ...group,
    fraction: sequence.length
      ? [...group.residues].reduce((sum, residue) => sum + (counts[residue] || 0), 0) / sequence.length
      : 0,
  }));

  return { length: sequence.length, massKda: mass / 1000, charge, invalid, groups };
}
