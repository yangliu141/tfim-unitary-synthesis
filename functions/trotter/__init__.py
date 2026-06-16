"""First-order Trotter product-formula decomposition for the TFIM."""

from .trotterization import (
    trotter_pauli_decomp,
    unitary_from_pauli_decomp,
    phase_aligned_error,
)

__all__ = ["trotter_pauli_decomp", "unitary_from_pauli_decomp", "phase_aligned_error"]
