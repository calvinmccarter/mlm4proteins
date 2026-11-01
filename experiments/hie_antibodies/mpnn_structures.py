"""Structure config for the ProteinMPNN/AbMPNN one-edit PLL eval + a trimmer.

Four of the seven Hie antibodies have a deposited structure whose heavy/light
chain contains the exact Hie VH/VL as a substring (verified against RCSB). The
two UCAs (no crystal structure) and c143 (near-homolog only, no exact match) are
omitted.

``trim_structure`` downloads the PDB and writes a minimal complex: the VH and VL
chains plus any other chain in atomic contact with the Fab (one copy of the
antigen), dropping the redundant Fab/antigen copies in these large assemblies.
This keeps the interface context while shrinking the complex ProteinMPNN must
encode.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
STRUCT_DIR = EXPERIMENT_DIR / "structures"

# antibody -> (PDB id, VH auth chain, VL auth chain). VH/VL chains verified to
# contain the exact Hie VH/VL sequence.
STRUCTURES: dict[str, dict[str, str]] = {
    "medi8852":  {"pdb": "5JW4", "vh_chain": "Q", "vl_chain": "R"},
    "mab114":    {"pdb": "5FHC", "vh_chain": "H", "vl_chain": "L"},
    "s309":      {"pdb": "6WPS", "vh_chain": "C", "vl_chain": "D"},
    "regn10987": {"pdb": "6XDG", "vh_chain": "C", "vl_chain": "A"},
}

_CONTACT_ANGSTROM = 5.0


def _atoms_by_chain(pdb_path: Path):
    """chain_id -> (N,3) float array of ATOM coordinates (first altloc only)."""
    coords: dict[str, list] = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        ch = line[21]
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        coords.setdefault(ch, []).append(xyz)
    return {c: np.asarray(v, dtype=np.float64) for c, v in coords.items()}


def trim_structure(pdb_id: str, vh_chain: str, vl_chain: str,
                   raw_path: Path, out_path: Path,
                   contact_ang: float = _CONTACT_ANGSTROM) -> list[str]:
    """Write a trimmed PDB with {VH, VL, chains contacting the Fab}. Returns the
    list of kept chain IDs."""
    from scipy.spatial import cKDTree

    atoms = _atoms_by_chain(raw_path)
    fab = np.vstack([atoms[vh_chain], atoms[vl_chain]])
    tree = cKDTree(fab)
    keep = {vh_chain, vl_chain}
    for ch, xyz in atoms.items():
        if ch in keep or xyz.size == 0:
            continue
        if tree.query_ball_point(xyz, contact_ang, return_length=True).sum() > 0:
            keep.add(ch)

    out_lines = []
    for line in raw_path.read_text().splitlines():
        if line.startswith(("ATOM", "TER", "HETATM")) and line[21] in keep:
            if line.startswith("HETATM"):  # drop waters/ligands, keep polymer only
                continue
            out_lines.append(line)
        elif line.startswith(("HEADER", "CRYST1")):
            out_lines.append(line)
    out_lines.append("END")
    out_path.write_text("\n".join(out_lines) + "\n")
    return sorted(keep)
