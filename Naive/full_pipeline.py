#!/usr/bin/env python3
"""
Naive unitary synthesis pipeline.

Decomposes exp(-i H t) for the TFIM using alternating AIII / Type-A
(cosine-sine + demultiplexing) decomposition.  Intended to be compared
against the TFIM-specific pipeline in the parent directory.

NOTE: The full 2^n unitary must be built and decomposed, so this method
is limited to small n (realistically n <= 10 on a standard machine).

Usage:  python full_pipeline.py [n] [t]
        n: number of qubits (default 4)
        t: evolution time   (default 1.0)
"""

import sys
import os
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TFIM"))
from build_TFIM import TFIM_Ham
from decompose import decompose, gate_counts, verify


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    t = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    J, h = 1.0, 1.0
    rotated, periodic = True, False

    print(f"n = {n}, J = {J}, h = {h}, t = {t}, rotated = {rotated}, periodic = {periodic}")
    print("=" * 60)

    # [1] Build U(t) = exp(-i H t) in the full 2^n Hilbert space
    H  = TFIM_Ham(n, J=J, h=h, rotated=rotated, periodic=periodic)
    U_t = expm(-1j * H * t)
    print(f"\n[1] U(t) shape: {U_t.shape}")

    # [2] Decompose using alternating AIII / Type-A
    print("\n[2] Running recursive AIII / Type-A decomposition...")
    gates  = decompose(U_t)
    counts = gate_counts(gates)
    print(f"    Total gates: {counts['total']}")
    print(f"    By type:     {counts['by_type']}")
    print(f"    By n_qubits: {counts['by_nqubits']}")

    # [3] Verify the algebraic decomposition (reconstructs U from block matrices directly)
    if n <= 10:
        passed, err = verify(U_t)
        print(f"\n[3] Algebraic reconstruction error: {err:.3e}")
        print("    " + ("PASS" if passed else "FAIL"))
    else:
        print(f"\n[3] Skipped (n={n} > 10): full matrix verification infeasible.")

    return gates, counts


if __name__ == "__main__":
    main()
