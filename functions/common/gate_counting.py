"""
Count the number of elementary gates {CNOT, single-qubit}
needed to implement a list of Pauli rotations.
"""
def pauli_rot_elementary_counts(pauli_decomp):
    """Return {'CNOT', 'single_qubit', 'total'} for a list of Pauli rotations.

    pauli_decomp : iterable whose entries start with a Pauli word
                   (tuple or string of 'I'/'X'/'Y'/'Z'), e.g. (word, coeff, ...).
    """
    # Counters
    cnot = 0
    single = 0

    for entry in pauli_decomp:
        word = entry[0] if isinstance(entry, (tuple, list)) else entry
        active = [p for p in word if p != "I"]
        w = len(active) # weight of the Pauli word (number of non-identity Paulis)
        if w == 0:
            continue
        n_xy = sum(1 for p in active if p in ("X", "Y")) # number of X/Y Paulis (which require basis change)
        cnot += 2 * (w - 1) # Ladder of CNOTs to entangle w qubits and disentangle after rotation
        single += 1 + 2 * n_xy # 1 single-qubit rotation + 2 basis changes per X/Y Pauli
    return {"CNOT": cnot, "single_qubit": single, "total": cnot + single}
