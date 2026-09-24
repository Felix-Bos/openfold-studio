"""Per-residue view of a predicted structure, read from its mmCIF text.

OpenFold writes the per-atom pLDDT in the B-factor column. Following the
AlphaFold convention, a residue's confidence is the pLDDT of its alpha
carbon (CA), and its position is the CA coordinate.
"""

from dataclasses import dataclass

import numpy as np

# Distance under which two residues' alpha carbons are considered in contact.
CONTACT_DISTANCE = 8.0


@dataclass(frozen=True)
class Residue:
    index: int  # 1-based position in the sequence
    name: str  # three-letter code, e.g. "MET"
    plddt: float
    ca: tuple[float, float, float]


def parse_residues(cif_text: str) -> list[Residue]:
    """Extracts one Residue per alpha carbon from the `_atom_site` loop."""
    columns: list[str] = []
    residues: list[Residue] = []
    in_atom_site = False

    for raw in cif_text.splitlines():
        line = raw.strip()
        if line.startswith("_atom_site."):
            columns.append(line.split(".", 1)[1].strip())
            in_atom_site = True
            continue
        if not in_atom_site or not line.startswith(("ATOM", "HETATM")):
            if in_atom_site and residues and (line == "#" or line.startswith("loop_")):
                break
            continue

        values = dict(zip(columns, line.split(), strict=False))
        if values.get("label_atom_id") != "CA":
            continue
        residues.append(
            Residue(
                index=int(values["label_seq_id"]),
                name=values["label_comp_id"],
                plddt=float(values["B_iso_or_equiv"]),
                ca=(float(values["Cartn_x"]), float(values["Cartn_y"]), float(values["Cartn_z"])),
            )
        )
    return residues


def ca_distance_matrix(residues: list[Residue]) -> np.ndarray:
    """[N, N] matrix of alpha-carbon distances in ångströms."""
    coords = np.array([r.ca for r in residues], dtype=np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))
