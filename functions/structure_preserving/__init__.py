"""Structure-preserving BDI/KAK decomposition of TFIM time evolution in so(2n)."""

from .find_DLA import tfim_pauliwords_gen, dla_pauli_words
from .build_isomorphism import map_to_majarana, build_so_matrix
from .BDI_decomp import from_generator, recursive_bdi, bdi, build_kak
from .BDI_verification import verify_bdi_decomposition
from .map_back import build_majorana_dla_map, map_ops_to_pauli

__all__ = [
    "tfim_pauliwords_gen", "dla_pauli_words",
    "map_to_majarana", "build_so_matrix",
    "from_generator", "recursive_bdi", "bdi", "build_kak",
    "verify_bdi_decomposition",
    "build_majorana_dla_map", "map_ops_to_pauli",
]
