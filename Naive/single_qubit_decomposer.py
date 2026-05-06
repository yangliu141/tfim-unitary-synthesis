import numpy as np

# Gate format: (gate_name, angle, wires)
#   gate_name : "Rz" | "Ry" | "CNOT"
#   angle     : float for Rz/Ry, None for CNOT
#   wires     : [qubit] for single-qubit, [control, target] for CNOT


def to_gates(ops, n):
    """Convert ops from decompose() into a flat list of elementary gates.

    ops : list of (data, size, op_type) from decompose()
    n   : total number of qubits (U is 2^n x 2^n)
    """
    gates = []
    for data, size, op_type in ops:
        target = n - int(np.log2(size))
        controls = list(range(target + 1, n))

        if op_type == "sq":
            gates += _sq(data, target)

        elif op_type == "ry":
            gates += _ucr(data, target, controls, "Ry",  2.0)
            
        elif op_type == "rz":
            gates += _ucr(data, target, controls, "Rz", -2.0)

    return gates


# ── Single-qubit ZYZ decomposition ───────────────────────────────────────────

def _sq(U, qubit):
    """Decompose a 2x2 unitary into Rz Ry Rz via ZYZ."""
    alpha, beta, gamma, delta = _zyz_angles(U)
    gates = []
    if abs(delta) > 1e-10:
        gates.append(("Rz", delta, [qubit]))
    if abs(gamma) > 1e-10:
        gates.append(("Ry", gamma, [qubit]))
    if abs(beta) > 1e-10:
        gates.append(("Rz", beta, [qubit]))
    return gates


def _zyz_angles(U):
    """Extract ZYZ Euler angles: U = e^{i*alpha} Rz(beta) Ry(gamma) Rz(delta).

    Rz(t) = [[e^{-it/2}, 0], [0, e^{it/2}]]
    Ry(t) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
    """
    phase = np.angle(np.linalg.det(U)) / 2
    V = U * np.exp(-1j * phase)

    gamma = 2 * np.arctan2(abs(V[1, 0]), abs(V[0, 0]))
    sum_angles = 2 * np.angle(V[1, 1])
    dif_angles = 2 * np.angle(V[1, 0])
    beta  = (sum_angles + dif_angles) / 2
    delta = (sum_angles - dif_angles) / 2

    return phase, beta, gamma, delta


# ── Uniformly controlled rotation (Gray-code recursion) ──────────────────────

def _ucr(angles, target, controls, gate, scale):
    """UCR: apply gate(scale * angles[j]) to target when controls encode |j>.

    gate  : "Ry" or "Rz"
    scale : +2.0 for Ry (from AIII A-block), -2.0 for Rz (from Type-A D-block)
    """
    if len(angles) == 0 or np.allclose(angles, 0):
        return []
    if not controls:
        return [(gate, scale * angles[0], [target])]

    half = len(angles) // 2
    avg  = (angles[:half] + angles[half:]) / 2
    diff = (angles[:half] - angles[half:]) / 2

    ctrl = controls[0]
    rest = controls[1:]
    return (
        _ucr(avg,  target, rest, gate, scale)
        + [("CNOT", None, [ctrl, target])]
        + _ucr(diff, target, rest, gate, scale)
        + [("CNOT", None, [ctrl, target])]
    )
