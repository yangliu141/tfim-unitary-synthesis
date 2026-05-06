#!/usr/bin/env python3
"""
Naive TFIM decomposition pipeline.

    H_TFIM --expm--> U(t)
    --recursive AIII/Type-A--> ops --ZYZ/UCR--> {Ry, Rz, CNOT}

Timing note: setup (building H and computing expm) is excluded from
"decomposition time" to match the TFIM pipeline convention, where steps
[1]-[3] (DLA, isomorphism, SO(2n) exponentiation) are also excluded.

Usage:  python full_pipeline.py [n] [t]
        n: number of qubits (default 4)
        t: evolution time   (default 1.0)
"""

import sys
import os
import time
import numpy as np
from scipy.linalg import expm
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TFIM"))

from decompose import decompose
from single_qubit_decomposer import to_gates
from build_TFIM import TFIM_Ham


def phase_aligned_error(U_ref, U_test):
    phase = np.angle(np.trace(U_test @ U_ref.conj().T))
    return float(np.max(np.abs(U_test - np.exp(1j * phase) * U_ref)))


def main(n_qubits, t=1.0, verbose=True):

    pipeline_time_start = time.time()
    results = {}

    n = n_qubits

    J, h = 1.0, 1.0
    rotated, periodic = True, False

    if verbose:
        print(f"n = {n}, J = {J}, h = {h}, t = {t}, rotated = {rotated}, periodic = {periodic}")
        print("=" * 60)

    # [1] Build H and compute U(t) = exp(-i H t) in the full 2^n Hilbert space
    H = TFIM_Ham(n, J=J, h=h, rotated=rotated, periodic=periodic)
    U_t = expm(-1j * H * t)
    
    if verbose:
        print(f"\n[1] U(t) shape: {U_t.shape}, "
              f"unitary: {np.allclose(U_t @ U_t.conj().T, np.eye(2**n), atol=1e-10)}")

    # [2] AIII / Type-A recursive decomposition  ← timer starts here
    time_start = time.time()

    ops = decompose(U_t)
    op_counts = Counter(op_type for _, _, op_type in ops)
    if verbose:
        print(f"\n[2] AIII/Type-A ops: {len(ops)}, breakdown: {dict(op_counts)}")

    # [3] Expand ops to elementary gates (ZYZ single-qubit + UCR)
    gates = to_gates(ops, n)
    gate_counts = Counter(name for name, _, _ in gates)

    time_end = time.time()           # ← timer ends here

    if verbose:
        print(f"\n[3] Elementary gates: {len(gates)}, breakdown: {dict(gate_counts)}")
        print(f"\nDecomposition time: {time_end - time_start:.8f} seconds")

    # [4] Verify by reconstructing U(t) from the circuit — feasible for small n
    err = None
    if n <= 10:
        import pennylane as qml
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev)
        def circuit():
            for name, angle, wires in gates:
                if   name == "Ry":   qml.RY(angle, wires=wires[0])
                elif name == "Rz":   qml.RZ(angle, wires=wires[0])
                elif name == "CNOT": qml.CNOT(wires=wires)
            return qml.state()

        err = phase_aligned_error(U_t, qml.matrix(circuit)())
        if verbose:
            print(f"\n[4] Phase-aligned reconstruction error: {err:.3e}")
            print("    " + ("PASS" if err < 1e-6 else "FAIL"))
    else:
        if verbose:
            print(f"\n[4] Skipped (n={n} > 10): full Hilbert space verification infeasible.")

    pipeline_time_end = time.time()

    results["Gate counts"] = {"total": len(gates), "by_type": dict(gate_counts)}
    results["Op counts"] = {"total": len(ops),   "by_type": dict(op_counts)}
    results["Error"] = err
    results["Decomposition time"] = time_end - time_start
    results["Total pipeline time"] = pipeline_time_end - pipeline_time_start

    # Export results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
    os.makedirs(results_dir, exist_ok=True)

    times_file = os.path.join(results_dir, "Naive_decomposition_times.txt")

    with open(times_file, "a") as f:
        if f.tell() == 0:
            f.write("n_qubits, total_pipeline_time, decomposition_time\n")
    
        f.write(f"{n}, {results['Total pipeline time']:.8f}, {results['Decomposition time']:.8f}\n")

    counts_file = os.path.join(results_dir, "Naive_decomposition_errors_and_counts.txt")

    with open(counts_file, "a") as f:
        if f.tell() == 0:
            f.write("n_qubits, error, total_gates, Ry, Rz, CNOT\n")
        err_str = f"{err:.3e}" if err is not None else "N/A"

        f.write(f"{n}, {err_str}, {results['Gate counts']['total']},"
                f"{gate_counts.get('Ry', 0)},"
                f"{gate_counts.get('Rz', 0)},"
                f"{gate_counts.get('CNOT', 0)}\n")

    return results


if __name__ == "__main__":

    if len(sys.argv) > 1:
        # Single run: python full_pipeline.py [n] [t]
        main(int(sys.argv[1]), t=float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
        sys.exit()

    n_values = np.arange(1, 13)   # naive is classically infeasible past ~n=12

    all_results = {}
    for n in n_values:
        results = main(n, verbose=True)
        all_results[n] = results
        print(f"Total pipeline time for n={n}: {results['Total pipeline time']:.8f} seconds")


    import matplotlib.pyplot as plt

    times = [all_results[n]["Decomposition time"] for n in n_values]
    gate_totals = [all_results[n]["Gate counts"]["total"] for n in n_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(n_values, times, color="red", marker="o")
    ax1.set_xlabel("Number of qubits (n)")
    ax1.set_ylabel("Decomposition time (seconds)")
    ax1.set_title("Decomposition time vs n")

    ax2.plot(n_values, gate_totals, color="blue", marker="o")
    ax2.set_xlabel("Number of qubits (n)")
    ax2.set_ylabel("Total gate count")
    ax2.set_title("Gate count vs n")

    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
    plt.savefig(os.path.join(results_dir, "Naive_scaling.png"), dpi=150)
    plt.show()
