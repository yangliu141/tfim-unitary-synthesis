"""
First-order Trotter pipeline for the Transverse Field Ising Model.

    H_TFIM --split--> H_coupling + H_field
    --first-order Trotter--> {exp(-i theta_k P_k)}
    --optional verification--> compare with exp(-i H t)

Usage:  python trotter_pipeline.py [n] [t] [r]
        n: number of qubits       
        t: evolution time         (default 1.0)
        r: Trotter steps          
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TFIM"))
from trotterization import trotter_pauli_decomp, unitary_from_pauli_decomp,phase_aligned_error
from build_TFIM import TFIM_Ham
from gate_counting import pauli_rot_elementary_counts
from scipy.linalg import expm



def main(n_qubits, trotter_steps, t=1.0, verbose=True):

    # Start time
    pipeline_time_start = time.time()
    results = {}

    n = n_qubits
    r = trotter_steps

    J, h = 1.0, 1.0
    rotated, periodic = True, False

    if verbose:
        print(f"n = {n}, J = {J}, h = {h}, t = {t}, r = {r}, rotated = {rotated}, periodic = {periodic}")
        print("=" * 60)

    # [1] Split Hamiltonian into Trotter layers
    n_coupling_terms = n if periodic and n > 2 else n - 1
    n_field_terms = n
    gates_per_step = n_coupling_terms + n_field_terms
    expected_total_gates = r * gates_per_step

    if verbose:
        print(f"\n[1] Hamiltonian split")
        print(f"    Coupling terms per step: {n_coupling_terms}")
        print(f"    Field terms per step:    {n_field_terms}")
        print(f"    Gates per step:          {gates_per_step}")
        print(f"    Expected total gates:    {expected_total_gates}")

    # [2] Build first-order Trotter decomposition
    time_start = time.time()
    pauli_decomp = trotter_pauli_decomp(n, J=J, h=h, t=t, r=r, rotated=rotated)
    time_end = time.time()


    if verbose:
        print(f"\n[2] First-order Trotter decomposition: {len(pauli_decomp)} gates")
        print(f"\nTotal decomposition time: {time_end - time_start:.8f} seconds")

    # [3] Verify by reconstructing exp(-i H t) — only feasible for small n
    err = None
    verification_time = None
    if n <= 8:
        verify_start = time.time()
        H = TFIM_Ham(n, J=J, h=h, rotated=rotated, periodic=periodic)
        U_ref = expm(-1.0j * H * t)
        U_rec = unitary_from_pauli_decomp(pauli_decomp, n)
        err = phase_aligned_error(U_ref, U_rec)
        verification_time = time.time() - verify_start

        if verbose:
            print(f"\n[3] Phase-aligned reconstruction error: {err:.3e}")
            print(f"    Verification time: {verification_time:.8f} seconds")
    else:
        if verbose:
            print(f"\n[3] Skipped verification (n={n} > 8): full Hilbert space reconstruction infeasible.")

    # Compile Pauli rotations to elementary {CNOT, single-qubit} gates for a
    # fair comparison with the Naive pipeline (analysis step, outside the timer).
    elem = pauli_rot_elementary_counts(pauli_decomp)

    results["Pauli decomposition"] = pauli_decomp
    results["Error"] = err
    results["Gate counts"] = {
    "total": len(pauli_decomp),
    "coupling": r * (n - 1),
    "field": r * n,
    "cnot": elem["CNOT"],
    "single_qubit": elem["single_qubit"],
    "elementary_total": elem["total"],
    }
    results["Decomposition time"] = time_end - time_start
    results["Verification time"] = verification_time

    pipeline_time_end = time.time()
    results["Total pipeline time"] = pipeline_time_end - pipeline_time_start

    # Export results to a text file for later analysis
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "TFIM_trotter_results.txt")
    with open(results_file, "a") as f:
        if f.tell() == 0:
            f.write("n_qubits, trotter_steps, total_time, decomposition_time, verification_time, error, total_gates, cnot, single_qubit, elementary_total\n")
        err_str = f"{results['Error']:.3e}" if results["Error"] is not None else "N/A"
        ver_str = f"{results['Verification time']:.8f}" if results["Verification time"] is not None else "N/A"
        gc = results["Gate counts"]
        f.write(
            f"{n}, {r}, {results['Total pipeline time']:.8f}, "
            f"{results['Decomposition time']:.8f}, {ver_str}, {err_str}, "
            f"{gc['total']}, {gc['cnot']}, {gc['single_qubit']}, {gc['elementary_total']}\n"
        )

    return results


if __name__ == "__main__":

    if len(sys.argv) > 1:
        # Single run: python trot_full_pipeline.py [n] [t] [r]
        n_arg = int(sys.argv[1])
        t_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        r_arg = int(sys.argv[3]) if len(sys.argv) > 3 else n_arg
        main(n_arg, trotter_steps=r_arg, t=t_arg)
        sys.exit()

    n_values = [4, 6, 8]

    all_results = {}

    for n in n_values:
        r_values = [1, 2, 4, 8, n, 2*n, 4*n]

        for r in r_values:
            results = main(n, trotter_steps=r)
            all_results[(n, r)] = results

            print(
                f"n={n}, r={r}: "
                f"total_time={results['Total pipeline time']:.8f} s, "
                f"decomp_time={results['Decomposition time']:.8f} s, "
                f"gates={results['Gate counts']['total']}, "
                f"error={results['Error']}"
            )

    import matplotlib.pyplot as plt

    # ------------------------------------------------------------
    # Plot 1: Gate count vs r, one curve per n
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    for n in n_values:
        r_values = [1, 2, 4, 8, n, 2*n, 4*n]
        gate_counts = [
            all_results[(n, r)]["Gate counts"]["total"]
            for r in r_values
        ]

        plt.plot(r_values, gate_counts, marker="o", label=f"n={n}")

    plt.xlabel("Trotter steps r")
    plt.ylabel("Total Number of PauliRot Gates")
    plt.title("Trotter Gate Count vs Trotter Steps")
    plt.legend()
    plt.show()

    # ------------------------------------------------------------
    # Plot 2: Error vs r, one curve per n
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    for n in n_values:
        r_values = [1, 2, 4, 8, n, 2*n, 4*n]
        errors = [
            all_results[(n, r)]["Error"]
            for r in r_values
        ]

        plt.plot(r_values, errors, marker="o", label=f"n={n}")

    plt.xlabel("Trotter steps r")
    plt.ylabel("Phase-aligned reconstruction error")
    plt.title("Trotter Error vs Trotter Steps")
    plt.yscale("log")
    plt.legend()
    plt.show()

    # ------------------------------------------------------------
    # Plot 3: Decomposition time vs r, one curve per n
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    for n in n_values:
        r_values = [1, 2, 4, 8, n, 2*n, 4*n]
        times = [
            all_results[(n, r)]["Decomposition time"]
            for r in r_values
        ]

        plt.plot(r_values, times, marker="o", label=f"n={n}")

    plt.xlabel("Trotter steps r")
    plt.ylabel("Decomposition Time (seconds)")
    plt.title("Trotter Decomposition Time vs Trotter Steps")
    plt.legend()
    plt.show()
