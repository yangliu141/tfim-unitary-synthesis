"""
Compile a list of Pauli rotations to an elementary {CNOT, single-qubit} gate set
and count them, so that TFIM / Trotter (which natively output multi-qubit
PauliRot's) can be compared on the same footing as the Naive pipeline (which
already outputs elementary {Ry, Rz, CNOT}).

A weight-w Pauli rotation exp(-i theta P) compiles (standard CNOT-ladder
construction) to:
    - 2*(w - 1)            CNOTs              (MultiRZ ladder)
    - 1                    Rz                 (central rotation)
    - 2 per X or Y factor  basis-change gates (H for X, RX for Y, applied twice)

The closed form below was validated to match qml.transforms.decompose exactly
across 200 random Pauli words (see verification in commit history).
"""


def pauli_rot_elementary_counts(pauli_decomp):
    """Return {'CNOT', 'single_qubit', 'total'} for a list of Pauli rotations.

    pauli_decomp : iterable whose entries start with a Pauli word
                   (tuple or string of 'I'/'X'/'Y'/'Z'), e.g. (word, coeff, ...).
    """
    cnot = 0
    single = 0
    for entry in pauli_decomp:
        word = entry[0] if isinstance(entry, (tuple, list)) else entry
        active = [p for p in word if p != "I"]
        w = len(active)
        if w == 0:
            continue
        n_xy = sum(1 for p in active if p in ("X", "Y"))
        cnot += 2 * (w - 1)
        single += 1 + 2 * n_xy
    return {"CNOT": cnot, "single_qubit": single, "total": cnot + single}
