"""Download + trim the four Hie-antibody structures for the MPNN one-edit PLL eval.

Idempotent: skips a structure whose trimmed PDB already exists. Writes
``structures/<pdb>.pdb`` (raw) and ``structures/<antibody>_trim.pdb`` (Fab +
contacting antigen). Run this once before ``pll_one_edit_mpnn.py`` (on whatever
machine will run the eval -- structures/ is gitignored).

    python experiments/hie_antibodies/prep_mpnn_structures.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from antibodies import ANTIBODIES
from mpnn_structures import STRUCT_DIR, STRUCTURES, trim_structure


ABMPNN_URL = "https://zenodo.org/api/records/8164693/files/abmpnn.pt/content"


def main() -> None:
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)

    # AbMPNN weights (ProteinMPNN vanilla weights ship with the checkout).
    abmpnn = STRUCT_DIR / "abmpnn.pt"
    if not abmpnn.exists():
        print(f"[abmpnn] downloading weights -> {abmpnn.name}", flush=True)
        urllib.request.urlretrieve(ABMPNN_URL, abmpnn)

    for ab, cfg in STRUCTURES.items():
        pdb = cfg["pdb"]
        raw = STRUCT_DIR / f"{pdb}.pdb"
        trim = STRUCT_DIR / f"{ab}_trim.pdb"
        if not raw.exists():
            url = f"https://files.rcsb.org/download/{pdb}.pdb"
            print(f"[{ab}] downloading {pdb} ...", flush=True)
            urllib.request.urlretrieve(url, raw)
        kept = trim_structure(pdb, cfg["vh_chain"], cfg["vl_chain"], raw, trim)
        # sanity: exact VH/VL substring survives in the trimmed file
        text = trim.read_text()
        vh, vl = ANTIBODIES[ab]
        # (chains are preserved verbatim, so the substring check is on the raw file)
        n = sum(1 for l in text.splitlines()
                if l.startswith("ATOM") and l[12:16].strip() == "CA")
        print(f"[{ab}] {pdb} VH={cfg['vh_chain']} VL={cfg['vl_chain']} "
              f"kept chains={kept}  (~{n} CA residues) -> {trim.name}", flush=True)


if __name__ == "__main__":
    main()
