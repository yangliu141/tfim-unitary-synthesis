# Unitary Synthesis for the Transverse Field Ising Model

A pipeline that decomposes the time-evolution operator of the
Transverse Field Ising Model (TFIM) into a sequence of single- and
two-qubit Pauli rotations, by exploiting the model's free-fermion
structure.

The route through the pipeline is:

```
H_TFIM  --DLA-->  g  --rho-->  so(2n)  --exp-->  U(t)
        --recursive BDI/KAK-->  K1 A K2  --map back-->  { exp(-i theta_k P_k) }
```

Concretely, instead of synthesising `U(t) = exp(-i H t)` as a generic
2^n x 2^n unitary, we work in the `so(2n)` representation provided by
the Majorana / Jordan–Wigner isomorphism, perform a recursive
Cosine–Sine (BDI / KAK) decomposition there, and then map every
rotation back to a Pauli rotation acting on the qubit register. The
final circuit contains exactly `n(2n - 1)` Pauli rotations — the
dimension of the dynamical Lie algebra of the TFIM.

## Repository layout

| File | Purpose |
| --- | --- |
| `build_TFIM.py` | Builds the TFIM Hamiltonian matrix and its generators. |
| `find_DLA.py` | Generates Pauli words and computes the dynamical Lie algebra (DLA) by nested commutators. |
| `build_isomorphism.py` | Maps Pauli generators to the `so(2n)` Majorana representation and assembles the skew-symmetric matrix `rho(iH)`. |
| `BDI_decomp.py` | Cosine–Sine / BDI(p,q) decomposition and the recursive KAK splitter. |
| `BDI_verification.py` | Reconstructs `U(t)` from the recursive ops and checks it against the original. |
| `map_back.py` | Converts the `so(2n)` rotations and angles back into Pauli rotations on the qubit register. |
| `full_pipeline.py` | End-to-end script: TFIM -> DLA -> so(2n) -> U(t) -> recursive BDI -> Pauli rotations -> reconstruction check. |
| `full_pipeline.ipynb` | Notebook version of the pipeline with intermediate output and plots. |

## Usage

Run the full pipeline from the command line:

```bash
python full_pipeline.py [n] [t]
```

* `n` — number of qubits (default `4`)
* `t` — evolution time (default `1.0`)

Example output:

```
n = 4, J = 1.0, h = 1.0, t = 1.0, rotated = True, periodic = False
[1] Initial generators: 7 | DLA size: 28 (expected n(2n-1) = 28)
[2] rho(iH) shape: (8, 8), skew-symmetric: True
[3] U(t) in SO(8), orthogonal: True
[4] Recursive BDI: ... ops, breakdown: {...}
[5] Pauli decomposition: 28 gates (expected n(2n-1) = 28)
[6] Phase-aligned reconstruction error: ...e-..
    PASS
```

`full_pipeline.py` also exposes `kak_time_evolution(pauli_decomp, time)`,
a PennyLane circuit fragment that applies the decomposition as
`qml.PauliRot` gates — useful for plugging the synthesised circuit into
a larger PennyLane workflow.

## Dependencies

* Python 3.x
* `numpy`
* `scipy` (`scipy.linalg.cossin`, `expm`)
* `pennylane` (only required for `kak_time_evolution`)

## Notes on the decomposition

* The `cossin` call in `BDI_decomp.bdi` deliberately uses `q = p`
  (a symmetric column split) so that the canonical BDI(p,q) layout is
  produced even for non-power-of-2 dimensions; the asymmetric split
  introduces a permutation component that breaks the Pauli mapping.
* The first level of CS angles is tagged `"a0"` to mark them as the
  ones tied to the original evolution time `t`. They are divided by
  `t` at decomposition time and re-multiplied by the desired evolution
  time when the circuit is applied. Angles produced by deeper
  recursive levels are tagged `"a"` and are *not* rescaled.
