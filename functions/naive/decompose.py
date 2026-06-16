import numpy as np
from scipy.linalg import cossin, schur

_ATOL = 1e-10


def decompose(U):
    """
    Decompose unitary U via alternating AIII and Type-A steps into a flat
    list of primitive ops in circuit order.

    Op format: (data, size, op_type)
      "sq" : 2x2 unitary leaf, for single-qubit decomposition; size=2
      "ry" : UCR-Y angle array theta, length=size//2
             (from AIII A-block: RY(2*theta[j]) on target when controls=|j>)
      "rz" : UCR-Z angle array phi, length=size//2
             (from Type-A D-block: Rz(-2*phi[j]) on target when controls=|j>)

    For an n-qubit system, size encodes which qubit is the UCR target:
      target qubit = n - log2(size),  controls = qubits below the target.
    """
    # Verify unitary input
    if not np.allclose(U @ U.conj().T, np.eye(len(U)), atol=_ATOL):
        raise ValueError(
            f"Input is not unitary "
            f"(max dev: {np.max(np.abs(U @ U.conj().T - np.eye(len(U)))):.2e})"
        )
    return _aiii(U, U.shape[0])


def _aiii(U, size):
    """AIII (cosine-sine) decomposition: U = K1 * A(theta) * K2."""
    if size == 2:
        return [(U, 2, "sq")]
    if np.allclose(U, np.eye(size), atol=_ATOL):
        return []

    p = size // 2
    (k11, k12), theta, (k21, k22) = cossin(U, p=p, q=p, separate=True)

    return (
        _type_a(k21, k22, size)      # K2
        + [(theta, size, "ry")]      # A-block UCR-Y
        + _type_a(k11, k12, size)    # K1
    )


def _type_a(U_top, V_bot, size):
    """Type-A demultiplexing: block-diag(U_top, V_bot) = (I x W) * D * (I x X).

    Uses Schur: U_top @ V_bot.H = W * diag(e^{2i*phi}) * W.H
    """

    p = size // 2

    delta = U_top @ V_bot.conj().T
    T, W = schur(delta, output="complex")
    phi = np.angle(np.diag(T)) / 2
    X = np.diag(np.exp(1j * phi)) @ W.conj().T @ V_bot

    return (
        _aiii(X, p)              # X on lower register
        + [(phi, size, "rz")]    # D-block UCR-Z
        + _aiii(W, p)            # W on lower register
    )
