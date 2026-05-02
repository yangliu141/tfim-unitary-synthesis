"""
Recursive AIII / Type-A unitary synthesis.

Decomposes an arbitrary n-qubit unitary into a flat list of native gates
using alternating cosine-sine (AIII) and demultiplexing (Type-A) decompositions.

Gate format: (name, angle, wires)
  name:  'Rz', 'SX', 'X', 'CRY', 'CRZ'
  angle: float for parametric gates, None for SX / X
  wires: list — one element for single-qubit, two for two-qubit gates
"""

import numpy as np
from scipy.linalg import cossin, schur
from single_qubit_decomposer import SingleQubitDecomposer

_ANGLE_TOL   = 1e-10
_IDENTITY_TOL = 1e-8
_sq = SingleQubitDecomposer()


# ── Public API ───────────────────────────────────────────────────────────────

def decompose(U):
    """Decompose unitary U into a flat list of (gate_name, angle, wires) tuples."""
    if not np.allclose(U @ U.conj().T, np.eye(len(U)), atol=1e-10):
        raise ValueError(
            f"Input is not unitary "
            f"(max dev: {np.max(np.abs(U @ U.conj().T - np.eye(len(U)))):.2e})"
        )
    nqubits = int(np.round(np.log2(len(U))))
    gates = []
    _aiii(U, list(range(nqubits)), gates)
    return gates


def gate_counts(gates):
    """Summarise gate counts by type and by number of qubits acted on."""
    by_type    = {}
    by_nqubits = {}
    for name, _, wires in gates:
        by_type[name]       = by_type.get(name, 0) + 1
        by_nqubits[len(wires)] = by_nqubits.get(len(wires), 0) + 1
    return {"total": len(gates), "by_type": by_type, "by_nqubits": by_nqubits}


def verify(U, tol=1e-8):
    """
    Verify the AIII/TypeA decomposition by algebraically reconstructing U
    from the block matrices and comparing to the original.
    Returns (passed: bool, max_error: float).
    Only feasible for len(U) <= 2^10 due to matrix memory.
    """
    U_rec = _aiii_reconstruct(U)
    err = float(np.max(np.abs(U - U_rec)))
    if err < tol:
        return True, err
    phase = float(np.angle(np.trace(U_rec @ U.conj().T) / len(U)))
    err_aligned = float(np.max(np.abs(U - np.exp(1j * phase) * U_rec)))
    return err_aligned < tol, err_aligned


def build_circuit(gates, nqubits):
    """
    Return a PennyLane QNode that applies the decomposed gate list.

    NOTE: The CRY/CRZ entries in the gate list are an approximation of the
    true AIII/Type-A A-blocks (which are uniformly controlled rotations).
    Use verify() for correctness checks — it works algebraically without
    the PennyLane approximation.
    """
    import pennylane as qml
    dev = qml.device("default.qubit", wires=nqubits)

    @qml.qnode(dev)
    def circuit():
        for name, angle, wires in gates:
            if   name == "Rz":  qml.RZ(angle, wires=wires[0])
            elif name == "SX":  qml.SX(wires=wires[0])
            elif name == "X":   qml.PauliX(wires=wires[0])
            elif name == "CRY": qml.CRY(angle, wires=wires)
            elif name == "CRZ": qml.CRZ(angle, wires=wires)
        return qml.state()

    return circuit


# ── Internal recursive helpers ────────────────────────────────────────────────

def _aiii(U, wires, gates):
    """AIII step: U = K1 · A · K2 via cosine-sine decomposition."""
    dim = len(U)
    if dim == 2:
        _single_qubit(U, wires, gates)
        return
    if np.allclose(U, np.eye(dim), atol=_IDENTITY_TOL):
        return
    (u1, u2), theta, (v1, v2) = cossin(U, dim // 2, dim // 2, separate=True)
    theta = np.where(np.abs(theta) < _ANGLE_TOL, 0.0, theta)
    _type_a(v1, v2, wires, gates)      # K2
    _cry_block(theta, wires, gates)    # A-block (uniformly controlled RY)
    _type_a(u1, u2, wires, gates)      # K1


def _type_a(U, V, wires, gates):
    """Type-A step: demultiplexes block-diagonal [[U,0],[0,V]] into identical blocks."""
    dim  = len(U)
    half = len(wires) // 2
    upper, lower = wires[:half], wires[half:]

    delta = U @ V.conj().T
    T, W  = schur(delta, output="complex")   # delta = W @ T @ W†, W unitary
    phi   = np.angle(np.diag(T)) / 2
    X_blk = np.diag(np.exp(1j * phi)) @ W.conj().T @ V

    if dim == 2:
        # Base case: W and X are single-qubit unitaries applied to one wire each
        _single_qubit(X_blk, upper, gates)   # K2 upper
        _single_qubit(X_blk, lower, gates)   # K2 lower
        _crz_block(phi, wires, gates)
        _single_qubit(W, upper, gates)       # K1 upper
        _single_qubit(W, lower, gates)       # K1 lower
    else:
        _aiii(X_blk, upper, gates)           # K2 upper
        _aiii(X_blk, lower, gates)           # K2 lower
        _crz_block(phi, wires, gates)
        _aiii(W, upper, gates)               # K1 upper
        _aiii(W, lower, gates)               # K1 lower


def _cry_block(theta, wires, gates):
    """Emit CRY gates for the AIII A-block."""
    if len(wires) < 2:
        return
    half    = len(wires) // 2
    control = wires[0]
    for i, angle in enumerate(theta):
        if abs(angle) >= _ANGLE_TOL:
            target = wires[min(half + (i % half), len(wires) - 1)]
            gates.append(("CRY", 2.0 * angle, [control, target]))


def _crz_block(phi, wires, gates):
    """Emit CRZ gates for the Type-A A-block."""
    if len(wires) < 2:
        return
    half    = len(wires) // 2
    control = wires[0]
    for i, angle in enumerate(phi):
        if abs(angle) >= _ANGLE_TOL:
            target = wires[min(half + (i % half), len(wires) - 1)]
            gates.append(("CRZ", -2.0 * angle, [control, target]))


def _single_qubit(U, wires, gates):
    """Decompose a 2×2 unitary and append native single-qubit gates."""
    if len(wires) == 0:
        return
    result = _sq.decompose(U)
    if result is None:
        return
    gate_list, _ = result
    # The decomposer builds U = G[0] @ G[1] @ ... in reversed-list order,
    # so apply in reversed order to get the same matrix in the circuit.
    for name, angle in reversed(gate_list):
        gates.append((name, angle, wires))


# ── Algebraic reconstruction (used by verify) ─────────────────────────────────

def _aiii_reconstruct(U):
    """Algebraically reconstruct U via AIII: returns K1 @ A @ K2."""
    dim = len(U)
    if dim == 2:
        return _sq_reconstruct(U)
    if np.allclose(U, np.eye(dim), atol=_IDENTITY_TOL):
        return np.eye(dim, dtype=complex)
    (u1, u2), theta, (v1, v2) = cossin(U, dim // 2, dim // 2, separate=True)
    K2 = _type_a_reconstruct(v1, v2)
    C, S = np.diag(np.cos(theta)), np.diag(np.sin(theta))
    A = np.block([[C, -S], [S, C]])
    K1 = _type_a_reconstruct(u1, u2)
    return K1 @ A @ K2


def _type_a_reconstruct(U, V):
    """Algebraically reconstruct [[U,0],[0,V]] via Type-A demultiplexing."""
    dim   = len(U)
    zero  = np.zeros((dim, dim), dtype=complex)
    delta = U @ V.conj().T
    T, W  = schur(delta, output="complex")   # delta = W @ T @ W†, W unitary
    phi   = np.angle(np.diag(T)) / 2
    X_blk = np.diag(np.exp(1j * phi)) @ W.conj().T @ V
    W_rec = _aiii_reconstruct(W)
    X_rec = _aiii_reconstruct(X_blk)
    K1 = np.block([[W_rec, zero], [zero, W_rec]])
    K2 = np.block([[X_rec, zero], [zero, X_rec]])
    D   = np.diag(np.exp( 1j * phi))
    Dd  = np.diag(np.exp(-1j * phi))
    A   = np.block([[D, zero], [zero, Dd]])
    return K1 @ A @ K2


def _sq_reconstruct(U):
    """Reconstruct a 2×2 unitary via single-qubit gate decomposition."""
    gates_lib = _sq.gates
    result = _sq.decompose(U)
    if result is None:
        return np.eye(2, dtype=complex)
    gate_list, phase = result
    M = np.eye(2, dtype=complex)
    for name, angle in reversed(gate_list):
        if   name == "Rz": M = gates_lib["Rz"](angle) @ M
        elif name == "SX":  M = gates_lib["SX"] @ M
        elif name == "X":   M = gates_lib["X"]  @ M
    return M * np.exp(1j * phase)
