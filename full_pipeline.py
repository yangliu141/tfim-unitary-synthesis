#!/usr/bin/env python3
"""
Full TFIM decomposition pipeline.

    H_TFIM --DLA--> g --rho--> so(2n) --exp--> U(t)
    --recursive BDI/KAK--> K1 A K2 --map back--> {exp(i theta_k P_k)}

Usage:  python full_pipeline.py [n] [t]
        n: number of qubits (default 4)
        t: evolution time   (default 1.0)
"""

import sys
import numpy as np
from scipy.linalg import expm
import pennylane as qml

from build_TFIM import TFIM_Ham
from find_DLA import tfim_pauliwords_gen, dla_pauli_words
from build_isomorphism import map_to_majarana, build_so_matrix
from BDI_decomp import from_generator, bdi, build_kak, recursive_bdi
from BDI_verification import verify_bdi_decomposition
from map_back import build_majorana_dla_map, map_ops_to_pauli


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_word_to_matrix(pw):
    mat = _PAULI[pw[0]]
    for p in pw[1:]:
        mat = np.kron(mat, _PAULI[p])
    return mat


def phase_aligned_error(U_ref, U_test):
    d = U_ref.shape[0]
    phase = np.angle(np.trace(U_test @ U_ref.conj().T) / d)
    return np.linalg.norm(U_ref - np.exp(-1j * phase) * U_test)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    t = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    J, h = 1.0, 1.0
    rotated, periodic = True, False

    print(f"n = {n}, J = {J}, h = {h}, t = {t}, rotated = {rotated}, periodic = {periodic}")
    print("=" * 60)

    # [1] DLA generators (Pauli words) and DLA closure
    generators = tfim_pauliwords_gen(n, rotated=rotated, periodic=periodic)
    dla_words = dla_pauli_words(generators)
    print(f"\n[1] Initial generators: {len(generators)} | DLA size: {len(dla_words)} "
          f"(expected n(2n-1) = {n*(2*n-1)})")

    # [2] Map to so(2n) via Majorana isomorphism
    maj_mapping = map_to_majarana(generators, J=J, h=h)
    rho_iH = build_so_matrix(maj_mapping, n)
    assert np.allclose(rho_iH.T, -rho_iH, atol=1e-10)
    print(f"\n[2] rho(iH) shape: {rho_iH.shape}, skew-symmetric: True")

    # [3] Exponentiate to U(t) in SO(2n)
    U_t = from_generator(rho_iH, t=t)
    assert np.allclose(U_t @ U_t.T, np.eye(U_t.shape[0]), atol=1e-10)
    print(f"\n[3] U(t) in SO({2*n}), orthogonal: True")

    # [4] Recursive BDI/KAK decomposition
    ops = recursive_bdi(U_t, num_iter=None, return_all=False)
    op_counts = {}
    for _, _, _, op_type in ops:
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
    print(f"\n[4] Recursive BDI: {len(ops)} ops, breakdown: {op_counts}")

    # [5] Map back to Pauli rotations
    mapping = build_majorana_dla_map(dla_words)
    pauli_decomp = map_ops_to_pauli(ops, mapping, time=t)
    print(f"\n[5] Pauli decomposition: {len(pauli_decomp)} gates "
          f"(expected n(2n-1) = {n*(2*n-1)})")

    # [6] Verify by reconstructing exp(-i H t)
    H = TFIM_Ham(n, J=J, h=h, rotated=rotated, periodic=periodic)
    U_ref = expm(-1.0j * H * t)
    U_rec = np.eye(2 ** n, dtype=complex)
    for word, coeff, _ in pauli_decomp:
        U_rec = U_rec @ expm(-1.0j * coeff * pauli_word_to_matrix(word))

    err = phase_aligned_error(U_ref, U_rec)
    print(f"\n[6] Phase-aligned reconstruction error: {err:.3e}")
    print("    " + ("PASS" if err < 1e-8 else "FAIL"))

    return pauli_decomp, err


def kak_time_evolution(pauli_decomp, time):
    """PennyLane circuit fragment that applies the decomposition.

    The 'a0' angles were divided by t at decomposition time, so they must be
    re-multiplied by ``time`` here. Pauli words are applied in reverse so that
    the matrix product matches U_rec = G_0 @ G_1 @ ... @ G_K.
    """
    for word, coeff, op_type in pauli_decomp[::-1]:
        if op_type == "a0":
            coeff = coeff * time
        pauli_str = "".join(p for p in word if p != "I")
        wires = [i for i, p in enumerate(word) if p != "I"]
        qml.PauliRot(2 * coeff, pauli_str, wires=wires)


if __name__ == "__main__":
    main()
