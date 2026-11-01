"""Canonical amino acid alphabet used throughout mmmlm."""

# Alphabetical 20-letter standard amino acid order. All `Tensor[L, 20]` outputs
# from ProteinMLM implementations follow this column ordering.
AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"
NUM_AMINO_ACIDS: int = len(AMINO_ACIDS)
AA_TO_IDX: dict[str, int] = {a: i for i, a in enumerate(AMINO_ACIDS)}
IDX_TO_AA: dict[int, str] = {i: a for i, a in enumerate(AMINO_ACIDS)}
