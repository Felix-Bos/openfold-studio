/** Pure functions over an N x N attention matrix (rows = query residue). */

/** Minimum / maximum number of links drawn from the selected residue. */
export const MIN_VISIBLE_LINKS = 12;
export const MAX_VISIBLE_LINKS = 24;

/** Index of the residue that sends the most attention to other residues. */
export function strongestResidue(weights) {
  if (!weights || weights.length === 0) return null;
  let bestIndex = 0;
  let bestScore = -Infinity;
  weights.forEach((row, i) => {
    const score = row.reduce((sum, value, j) => (i === j ? sum : sum + value), 0);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  });
  return bestIndex;
}

/**
 * Partners of `selected`, strongest first, as `{ index, value, normalized }`
 * (normalized = value / max of the row). Keeps partners whose normalized
 * weight reaches `threshold`, but always at least MIN_VISIBLE_LINKS and at
 * most MAX_VISIBLE_LINKS so the 3D view is never empty nor cluttered.
 */
export function connectedResidues(row, selected, threshold) {
  if (!row) return [];
  const max = Math.max(...row);
  if (max <= 0) return [];
  return row
    .map((value, index) => ({ index, value, normalized: value / max }))
    .filter((item) => item.index !== selected && item.value > 0)
    .sort((a, b) => b.value - a.value)
    .filter((item, rank) => item.normalized >= threshold || rank < MIN_VISIBLE_LINKS)
    .slice(0, MAX_VISIBLE_LINKS);
}
