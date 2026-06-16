import numpy as np
from scipy.linalg import expm

from functions.common.verification import pauli_word_to_matrix, phase_aligned_error


def trotter_pauli_decomp(n, J, h, t, r, rotated=True):
    """
    First-order Trotter decomposition as Pauli exponentials.

    Each tuple is (pauli_word, coeff, stage), representing exp(-i coeff P).

    For rotated=True:
        H = -J sum_i X_i X_{i+1} - h sum_i Z_i.

    Therefore one Trotter step applies
    exp(+i J dt X_i X_{i+1}) and exp(+i h dt Z_i),
    which corresponds to coeff = -J dt and coeff = -h dt in exp(-i coeff P).
    """

    dt = t / r
    decomp = []

    for _ in range(r):
        for i in range(n - 1):
            word = ["I"] * n
            if rotated:
                word[i], word[i + 1] = "X", "X"
            else:
                word[i], word[i + 1] = "Z", "Z"
            decomp.append((tuple(word), -J * dt))
        for i in range(n):
            word = ["I"] * n
            word[i] = "Z" if rotated else "X"
            decomp.append((tuple(word), -h * dt))

    return decomp

def unitary_from_pauli_decomp(pauli_decomp, n):
    """Build U = prod_k exp(-i coeff_k P_k)."""
    U = np.eye(2**n, dtype=complex)
    for word, coeff in pauli_decomp:
        U = U @ expm(-1j * coeff * pauli_word_to_matrix(word))
    return U