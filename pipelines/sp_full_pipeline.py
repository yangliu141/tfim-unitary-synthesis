"""
Full structure-preserving decomposition pipeline.

    H_TFIM --DLA--> g --rho--> so(2n) --exp--> U(t)
    --recursive BDI/KAK--> K1 A K2 --map back--> {exp(i theta_k P_k)}

Usage:  python pipelines/sp_full_pipeline.py [n] [t]
        n: number of qubits (default 4)
        t: evolution time   (default 1.0)
"""

import sys
import os
import numpy as np
from scipy.linalg import expm

# Make the repo root importable so the `functions` package is found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.common.build_TFIM import TFIM_Ham
from functions.structure_preserving.find_DLA import tfim_pauliwords_gen, dla_pauli_words
from functions.structure_preserving.build_isomorphism import map_to_majarana, build_so_matrix
from functions.structure_preserving.BDI_decomp import from_generator, recursive_bdi
from functions.structure_preserving.map_back import build_majorana_dla_map, map_ops_to_pauli
from functions.common.gate_counting import pauli_rot_elementary_counts
from functions.common.verification import phase_aligned_error, reconstruct_pauli_decomp

import time


def main(n_qubits, t=1.0, verbose=True):

    pipeline_time_start = time.time()
    results = {}

    n = n_qubits

    J, h = 1.0, 1.0
    rotated, periodic = True, False

    if verbose:
        print(f"n = {n}, J = {J}, h = {h}, t = {t}, rotated = {rotated}, periodic = {periodic}")
        print("=" * 60)

    # [1] DLA generators (Pauli words) and DLA closure
    generators = tfim_pauliwords_gen(n, rotated=rotated, periodic=periodic)
    dla_words = dla_pauli_words(generators)
    if verbose:
        print(f"\n[1] Initial generators: {len(generators)} | DLA size: {len(dla_words)} "
              f"(expected n(2n-1) = {n*(2*n-1)})")

    # [2] Map to so(2n) via Majorana isomorphism
    maj_mapping = map_to_majarana(generators, J=J, h=h)
    rho_iH = build_so_matrix(maj_mapping, n)
    assert np.allclose(rho_iH.T, -rho_iH, atol=1e-10)
    if verbose:
        print(f"\n[2] rho(iH) shape: {rho_iH.shape}, skew-symmetric: True")

    # [3] Exponentiate to U(t) in SO(2n)
    U_t = from_generator(rho_iH, t=t)
    assert np.allclose(U_t @ U_t.T, np.eye(U_t.shape[0]), atol=1e-10)
    if verbose:
        print(f"\n[3] U(t) in SO({2*n}), orthogonal: True")

    # [4] Recursive BDI/KAK decomposition
    # Start time measurement
    time_start = time.time()
    ops = recursive_bdi(U_t, num_iter=None, return_all=False)
    op_counts = {}
    for _, _, _, op_type in ops:
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
    if verbose:
        print(f"\n[4] Recursive BDI: {len(ops)} ops, breakdown: {op_counts}")

    # [5] Map back to Pauli rotations
    mapping = build_majorana_dla_map(dla_words)
    pauli_decomp = map_ops_to_pauli(ops, mapping, time=t)
    if verbose:
        print(f"\n[5] Pauli decomposition: {len(pauli_decomp)} gates "
              f"(expected n(2n-1) = {n*(2*n-1)})")
    # End time measurement
    time_end = time.time()
    
    by_stage = {}
    by_weight = {}
    for word, _, op_type in pauli_decomp:
        by_stage[op_type] = by_stage.get(op_type, 0) + 1
        w = sum(1 for p in word if p != "I")
        by_weight[w] = by_weight.get(w, 0) + 1
    if verbose:
        print(f"By stage:  {by_stage}")
        print(f"By weight: { {k: by_weight[k] for k in sorted(by_weight)} }")
    if verbose:
        print(f"\nTotal decomposition time: {time_end - time_start:.8f} seconds")
    
    # [6] Verify by reconstructing exp(-i H t) — only feasible for small n
    if n <= 7:
        H = TFIM_Ham(n, J=J, h=h, rotated=rotated, periodic=periodic)
        U_ref = expm(-1.0j * H * t)
        U_rec = reconstruct_pauli_decomp(pauli_decomp, n, t)
        err = phase_aligned_error(U_ref, U_rec)
        if verbose:
            print(f"\n[6] Phase-aligned reconstruction error: {err:.3e}")
            print("    " + ("PASS" if err < 1e-8 else "FAIL"))
    else:
        if verbose:
            print(f"\n[6] Skipped (n={n} > 7): full Hilbert space verification infeasible.")


    # Compile Pauli rotations to elementary {CNOT, single-qubit} gates for a fair
    # comparison with the Naive pipeline.
    elem = pauli_rot_elementary_counts(pauli_decomp)

    results["Pauli decomposition"] = pauli_decomp
    results["Error"] = err if n <= 7 else None
    results["Gate counts"] = {"total": len(pauli_decomp), "by_stage": by_stage, "by_weight": by_weight,
                              "cnot": elem["CNOT"], "single_qubit": elem["single_qubit"],
                              "elementary_total": elem["total"]}
    results["Decomposition time"] = time_end - time_start
    pipeline_time_end = time.time()
    results["Total pipeline time"] = pipeline_time_end - pipeline_time_start


    # Export results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "SPD_decomposition_results.txt")
    with open(results_file, "a") as f:
        # If file is empty, write header
        if f.tell() == 0:
            f.write("n_qubits, total_time, decomposition_time, error, total_gates, cnot, single_qubit, elementary_total\n")
        err_str = f"{results['Error']:.3e}" if results['Error'] is not None else "N/A"
        gc = results["Gate counts"]
        f.write(f"{n}, {results['Total pipeline time']:.8f}, {results['Decomposition time']:.8f}, {err_str}, "
                f"{gc['total']}, {gc['cnot']}, {gc['single_qubit']}, {gc['elementary_total']}\n")

    return results


if __name__ == "__main__":

    if len(sys.argv) > 1:
        # Single run: python full_pipeline.py [n] [t]
        main(int(sys.argv[1]), t=float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
        sys.exit()

    
    n_values = [1, 2, 3, 5, 7, 10, 13, 19, 26, 37, 51, 71, 100, 138, 193]
    all_results = {}
    for n in n_values:
        results = main(n)
        all_results[n] = results
        print(f"Total pipeline time for n={n}: {results['Total pipeline time']:.8f} seconds")

    import matplotlib.pyplot as plt

    # Plot the time taken for the decomposition as a function of n
    times = [all_results[n]["Decomposition time"] for n in n_values]
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, times, color='red', marker='o')
    plt.xlabel('Number of Qubits (n)')
    plt.ylabel('Decomposition Time (seconds)')
    plt.title('Decomposition Time vs Number of Qubits')
    plt.show()

    # Plot the total number of gates in the decomposition as a function of n
    gate_counts = [all_results[n]["Gate counts"]["total"] for n in n_values]
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, gate_counts, color='blue', marker='o')
    plt.xlabel('Number of Qubits (n)')
    plt.ylabel('Total Number of Gates')
    plt.title('Total Number of Gates vs Number of Qubits')
    plt.show()