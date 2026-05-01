"""
Unitary Synthesis V2 — Recursive AIII / TypeA Decomposition

Decomposes arbitrary n-qubit unitaries into a decomposition tree using
alternating cosine-sine (AIII) and demultiplexing (TypeA) decompositions.

Leaf-level single-qubit translation, circuit generation, and verification live in separate modules:

  - single_qubit_decomposer.py   — 2×2 → gate sequence translation
  - circuit_builder.py           — decomposition tree → PennyLane circuit
  - verification.py              — reconstruction-based verification & metrics
  - decomposition_nodes.py       — node data classes (shared across all modules)
"""

import numpy as np
from scipy.linalg import cossin, eig
from typing import Optional

from decomposition_nodes import (
    DecompositionNode, SingleQubitNode, IdentityNode, AIIINode, TypeANode
)
from single_qubit_decomposer import SingleQubitDecomposer
from circuit_builder import CircuitBuilder
from Synthesiser.V_1.verification import DecompositionVerifier

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Main Synthesiser Class


class UnitarySynthesiser:
    """
    Recursive unitary synthesis using alternating cosine-sine (AIII) and demultiplexing (A) decompositions 
    Decomposes arbitrary n-qubit unitaries into 2x2 unitaries (leaves) plus uniformly controlled rotations (A-blocks)
    """
    
    UNITARITY_TOL = 1e-10
    ANGLE_TOL = 1e-10
    GATE_SIMPLIFY_TOL = 1e-8  # Used for identity pruning in the structural decomposition

    def __init__(self, unitary: np.ndarray, gates: dict = None):
        self.unitary = unitary
        
        if not self._is_unitary(unitary):
            raise ValueError(
                "Input matrix is not unitary! U @ U† must equal I.\n"
                f"Max deviation: {np.max(np.abs(unitary @ unitary.conj().T - np.eye(len(unitary)))):.6e}"
            )
        
        self.single_qubit_decomposer = SingleQubitDecomposer(gates)
        self.gates = self.single_qubit_decomposer.gates
        self.dimension = len(unitary)
        self.nqubits = int(np.round(np.log2(self.dimension)))
        self.decomposition_tree: Optional[DecompositionNode] = None
    
    def _is_unitary(self, U: np.ndarray, tol: float = None) -> bool:
        tol = tol or self.UNITARITY_TOL
        return np.allclose(U @ U.conj().T, np.eye(len(U)), atol=tol) # Check if U @ U† ≈ I within tolerance
    
    # Decomposition: AIII ↔ A Alternating Recursion
    
    def decompose(self, show: bool = False, show_leaves: bool = False) -> DecompositionNode:
        """Recursively decompose the unitary into native gates."""
        print("Starting recursive KAK-decomposition, alternating AIII/A...")
        self.decomposition_tree = self._aiii_decompose(self.unitary)
        print("Decomposition complete.")
        
        # Build helpers now that the tree exists
        self._verifier = DecompositionVerifier(self.unitary, self.gates, self.decomposition_tree)
        self._circuit_builder = CircuitBuilder(self.nqubits, self.decomposition_tree)
        
        if show:
            print("\nDecomposition Tree:")
            self._verifier.print_tree()
        
        print("\nDecomposition Summary:")
        for key, value in self.get_info().items():
            print(f"  {key}: {value}")
        
        if show_leaves:
            self._verifier.print_leaves()
        
        return self.decomposition_tree
    
    def _aiii_decompose(self, U: np.ndarray) -> DecompositionNode:
        """
        Type AIII decomposition: U = K1 * A * K2
        
        Uses cosine-sine decomposition to factor U into:
        - K1 = [[u1, 0], [0, u2]]: block-diagonal with DIFFERENT blocks
        - A: uniformly controlled RY rotations (entangling)
        - K2 = [[v1, 0], [0, v2]]: block-diagonal with DIFFERENT blocks
        
        Then applies Type A decomposition to K1 and K2 to demultiplex them.
        """
        dim = len(U)
        
        # Base case: single qubit
        if dim == 2:
            return self._decompose_single_qubit(U)
        
        # Identity pruning
        if np.allclose(U, np.eye(dim), atol=self.GATE_SIMPLIFY_TOL):
            return IdentityNode(dimension=dim)
        
        # CS decomposition: U = [[u1, 0], [0, u2]] @ A @ [[v1, 0], [0, v2]]
        (u1, u2), theta, (v1, v2) = cossin(U, dim//2, dim//2, separate=True)
        theta = np.where(np.abs(theta) < self.ANGLE_TOL, 0.0, theta)
        
        # Apply Type A decomposition (demultiplexing) to the block-diagonal matrices
        k1_node = self._a_decompose(u1, u2)
        k2_node = self._a_decompose(v1, v2)
        
        return AIIINode(
            dimension=dim,
            k1=k1_node,
            k2=k2_node,
            theta=theta
        )
    
    def _a_decompose(self, U: np.ndarray, V: np.ndarray) -> DecompositionNode:
        """
        Type A decomposition (demultiplexing): [[U, 0], [0, V]] = K1 * A * K2
        
        Takes two DIFFERENT blocks U, V from a block-diagonal matrix.
        Produces:
        - K1 = [[W, 0], [0, W]]: block-diagonal with IDENTICAL blocks
        - A: uniformly controlled RZ rotations
        - K2 = [[X, 0], [0, X]]: block-diagonal with IDENTICAL blocks
        
        Then recursively applies AIII decomposition to W and X.
        
        Algorithm from PennyLane tutorial
        """
        dim = len(U)
        full_dim = 2 * dim
        
        # Demultiplexing algorithm
        delta = U @ V.conj().T
        
        # Eigenvalue decomposition: δ = W @ D² @ W†
        D_squared, W = eig(delta)
        
        # Compute square root by halving phases: D = diag(e^{i*φ/2})
        phi = np.angle(D_squared) / 2
        D = np.diag(np.exp(1j * phi))
        
        # Compute X = D @ W† @ V
        X = D @ W.conj().T @ V
        
        
        # Base case: single qubit blocks
        if dim == 2:
            # W and X are now 2x2 single-qubit unitaries
            w_node = self._decompose_single_qubit(W)
            x_node = self._decompose_single_qubit(X)
        else:
            # Recursively decompose W and X using AIII decomposition
            w_node = self._aiii_decompose(W)
            x_node = self._aiii_decompose(X)
        
        return TypeANode(
            dimension=full_dim,
            w=w_node,
            x=x_node,
            phi=phi
        )
    
    def _decompose_single_qubit(self, U: np.ndarray) -> DecompositionNode:
        """Delegate to the SingleQubitDecomposer and wrap the result in a tree node."""
        result = self.single_qubit_decomposer.decompose(U)

        if result is None:
            return IdentityNode(dimension=2, raw_unitary=U.copy())
        gates, global_phase = result
        
        return SingleQubitNode(gates=gates, global_phase=global_phase, raw_unitary=U.copy())
    
    # Delegated Methods
    
    def verify_decomposition(self, tolerance: float = 1e-8) -> bool:
        """Verify reconstructed unitary matches original (up to global phase)."""
        self._require_decomposition()
        return self._verifier.verify_decomposition(tolerance)
    
    def count_leaves(self) -> int:
        self._require_decomposition()
        return self._verifier.count_leaves()
    
    def get_depth(self) -> int:
        self._require_decomposition()
        return self._verifier.get_depth()
    
    def count_gates(self) -> dict:
        self._require_decomposition()
        gates_dict = self._verifier.count_gates()
        return gates_dict
    
    def get_info(self) -> dict:
        self._require_decomposition()
        return self._verifier.get_info()
    
    def to_pennylane_circuit(self):
        """Convert decomposition to a PennyLane circuit."""
        self._require_decomposition()
        return self._circuit_builder.to_pennylane_circuit()
    
    def draw_circuit(self):
        """Draw the decomposed circuit."""
        self._require_decomposition()
        return self._circuit_builder.draw_circuit()
    
    def print_multi_qubit_structure(self):
        """Print which qubits each operation acts on."""
        self._require_decomposition()
        self._verifier.print_multi_qubit_structure()
    
    def print_raw_unitaries(self):
        """Print the raw 2×2 unitaries at the leaves before gate translation."""
        self._require_decomposition()
        self._verifier.print_raw_unitaries()
    
    def get_raw_unitaries(self) -> list:
        """Return a list of raw 2×2 unitary matrices from the leaf nodes."""
        self._require_decomposition()
        return self._verifier.get_raw_unitaries()
    
    def _require_decomposition(self):
        if self.decomposition_tree is None:
            raise ValueError("Must call decompose() first!")