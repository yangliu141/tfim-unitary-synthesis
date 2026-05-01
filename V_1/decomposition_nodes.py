"""
Decomposition tree node data classes.

These nodes form the abstract representation of the recursive AIII / TypeA
decomposition.  They are pure data — no PennyLane or hardware-specific logic.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

# Node Classes for Decomposition Tree

@dataclass
class DecompositionNode(ABC):
    """Abstract base class for decomposition tree nodes."""
    
    @abstractmethod
    def count_gates(self) -> dict:
        """Count gates in this subtree."""
        pass
    
    @abstractmethod
    def get_depth(self) -> int:
        """Get depth of this subtree."""
        pass
    
    @abstractmethod
    def reconstruct(self, gates: dict) -> np.ndarray:
        """Reconstruct the unitary from this node."""
        pass


@dataclass
class SingleQubitNode(DecompositionNode):
    """Represents a single-qubit gate sequence."""
    gates: List[Tuple[str, Optional[float]]]  # Gates in matrix multiplication order (leftmost first)
    global_phase: float = 0.0
    raw_unitary: Optional[np.ndarray] = None   # The original 2×2 unitary before gate translation
    
    def count_gates(self) -> dict:
        counts = {'SX': 0, 'Rz': 0, 'X': 0, 'CRY': 0, 'CRZ': 0}
        for gate_name, _ in self.gates:
            if gate_name in counts:
                counts[gate_name] += 1
        return counts
    
    def get_depth(self) -> int:
        return 0
    
    def reconstruct(self, gates: dict) -> np.ndarray:
        """Reconstruct U = G_0 @ G_1 @ ... @ G_n (matrix order, leftmost applied last to state)."""
        U = np.eye(2, dtype=complex)
        for gate_name, angle in reversed(self.gates):  # Reversed to get G_0 @ G_1 @ ... order
            if gate_name == 'Rz' and angle is not None:
                U = gates['Rz'](angle) @ U
            elif gate_name == 'SX':
                U = gates['SX'] @ U
            elif gate_name == 'X':
                U = gates['X'] @ U
        return U * np.exp(1j * self.global_phase)


@dataclass
class IdentityNode(DecompositionNode):
    """Represents a trivial identity operation (pruned)."""
    dimension: int = 2
    raw_unitary: Optional[np.ndarray] = None  # The original unitary before pruning (only for 2×2 leaves)
    
    def count_gates(self) -> dict:
        return {'SX': 0, 'Rz': 0, 'X': 0, 'CRY': 0, 'CRZ': 0}
    
    def get_depth(self) -> int:
        return 0
    
    def reconstruct(self, gates: dict) -> np.ndarray:
        return np.eye(self.dimension, dtype=complex)


@dataclass 
class AIIINode(DecompositionNode):
    """
    Type AIII decomposition: U = K₁ · A · K₂
    
    K₁ = [[u1, 0], [0, u2]] and K₂ = [[v1, 0], [0, v2]] are block-diagonal.
    A contains uniformly controlled RY rotations (multiplexed).
    """
    dimension: int
    k1: DecompositionNode  # Decomposition of K₁ (block-diagonal with different blocks)
    k2: DecompositionNode  # Decomposition of K₂ (block-diagonal with different blocks)
    theta: np.ndarray      # Angles for the A matrix (uniformly controlled RY)
    
    def count_gates(self) -> dict:
        counts = {'SX': 0, 'Rz': 0, 'X': 0, 'CRY': 0, 'CRZ': 0}
        for node in [self.k1, self.k2]:
            sub_counts = node.count_gates()
            for gate in counts:
                counts[gate] += sub_counts.get(gate, 0)

        # Count controlled rotations from A matrix (uniformly controlled RY)
        counts['CRY'] += sum(1 for t in self.theta if abs(t) > 1e-10)
        return counts
    
    def get_depth(self) -> int:
        return 1 + max(self.k1.get_depth(), self.k2.get_depth())
    
    def reconstruct(self, gates: dict) -> np.ndarray:
        K1 = self.k1.reconstruct(gates)
        K2 = self.k2.reconstruct(gates)
        
        cos_theta = np.diag(np.cos(self.theta))
        sin_theta = np.diag(np.sin(self.theta))
        A = np.block([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        return K1 @ A @ K2


@dataclass
class TypeANode(DecompositionNode):
    """
    Type A decomposition (demultiplexing): [[U, 0], [0, V]] = K1 * A * K2
    
    Takes block-diagonal with DIFFERENT blocks U, V.
    Produces K1 = [[W, 0], [0, W]], K2 = [[X, 0], [0, X]] with IDENTICAL blocks.
    A contains uniformly controlled RZ rotations.
    """
    dimension: int
    w: DecompositionNode   # Decomposition of W (appears in both diagonal blocks of K1)
    x: DecompositionNode   # Decomposition of X (appears in both diagonal blocks of K2)
    phi: np.ndarray        # Angles for the A matrix (uniformly controlled RZ)
    
    def count_gates(self) -> dict:
        counts = {'SX': 0, 'Rz': 0, 'X': 0, 'CRY': 0, 'CRZ': 0}
        # W and X each appear in BOTH blocks, so we count them twice
        for node in [self.w, self.x]:
            sub_counts = node.count_gates()
            for gate in counts:
                counts[gate] += 2 * sub_counts.get(gate, 0)
        # Count controlled rotations from A matrix (uniformly controlled RZ)
        counts['CRZ'] += sum(1 for p in self.phi if abs(p) > 1e-10)
        return counts
    
    def get_depth(self) -> int:
        return 1 + max(self.w.get_depth(), self.x.get_depth())
    
    def reconstruct(self, gates: dict) -> np.ndarray:
        W = self.w.reconstruct(gates)
        X = self.x.reconstruct(gates)
        
        half = self.dimension // 2
        zero = np.zeros((half, half), dtype=complex)
        K1 = np.block([[W, zero], [zero, W]])
        K2 = np.block([[X, zero], [zero, X]])
        
        # A = [[D, 0], [0, D†]] where D = diag(e^{i*φ})
        D = np.diag(np.exp(1j * self.phi))
        D_dag = np.diag(np.exp(-1j * self.phi))
        A = np.block([[D, zero], [zero, D_dag]])
        
        return K1 @ A @ K2
