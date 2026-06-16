"""Shared verification helpers used across the three methods: build a unitary
from a Pauli decomposition and measure how close two unitaries are up to a
global phase."""

import numpy as np
from scipy.linalg import expm

# Pauli matrices
_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_word_to_matrix(word):
    """Return the 2^n x 2^n matrix of a Pauli word (tuple/list/str of I, X, Y, Z)."""
    mat = _PAULI[word[0]]
    for p in word[1:]:
        mat = np.kron(mat, _PAULI[p])
    return mat


def phase_aligned_error(U_ref, U_test):
    """Frobenius distance between U_ref and U_test after removing the global phase."""
    d = U_ref.shape[0]
    phase = np.angle(np.trace(U_test @ U_ref.conj().T) / d)
    return float(np.linalg.norm(U_ref - np.exp(-1j * phase) * U_test))


def reconstruct_pauli_decomp(pauli_decomp, n, t=1.0):
    """Reconstruct U = prod_k exp(-i alpha_k P_k) from a Pauli decomposition.

    Each entry is (word, coeff) or (word, coeff, op_type). For 'a0' ops the
    coefficient was stored divided by t in map_ops_to_pauli, so it is multiplied
    back by t here; all other coefficients are used directly. Operators are
    applied in list order: U = G_0 @ G_1 @ ... @ G_K.
    """
    U = np.eye(2 ** n, dtype=complex)
    for entry in pauli_decomp:
        word, coeff = entry[0], entry[1]
        op_type = entry[2] if len(entry) > 2 else None
        alpha = coeff * t if op_type == "a0" else coeff
        U = U @ expm(-1j * alpha * pauli_word_to_matrix(word))
    return U
