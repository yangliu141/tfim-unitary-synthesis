"""
Verification, metrics, and visualization for the decomposition tree.

Provides tools to verify that the decomposition faithfully reproduces
the original unitary, count gates, and print/visualise the tree structure.
"""

import numpy as np
from typing import List

from decomposition_nodes import (
    DecompositionNode, SingleQubitNode, IdentityNode, AIIINode, TypeANode
)


class DecompositionVerifier:
    """
    Verifies and inspects a decomposition tree.
    
    Handles reconstruction-based verification (comparing the reconstructed
    unitary against the original), gate counting, leaf counting, and
    tree printing / structure visualization.
    """
    
    ANGLE_TOL = 1e-10
    
    def __init__(self, unitary: np.ndarray, gates: dict, decomposition_tree: DecompositionNode):
        self.unitary = unitary
        self.gates = gates
        self.decomposition_tree = decomposition_tree
        self.nqubits = int(np.round(np.log2(len(unitary))))
    
    # Verification
    
    def verify_decomposition(self, tolerance: float = 1e-8) -> bool:
        """Verify reconstructed unitary matches original (up to global phase)."""
        reconstructed = self.decomposition_tree.reconstruct(self.gates)
        
        if np.allclose(self.unitary, reconstructed, atol=tolerance):
            print("✓ Decomposition verified: exact match.")
            return True
        
        product = self.unitary @ reconstructed.conj().T
        phase = np.angle(product.flat[0])
        if np.allclose(product * np.exp(-1j * phase), np.eye(len(self.unitary)), atol=tolerance):
            print(f"✓ Decomposition verified: match up to global phase (φ = {phase:.4f} rad).")
            return True
        
        print("✗ Decomposition verification FAILED.")
        print(f"  Max error: {np.max(np.abs(self.unitary - reconstructed)):.2e}")
        return False
    
    # Metrics
    
    def count_leaves(self) -> int:
        return self._count_leaves_recursive(self.decomposition_tree)
    
    def _count_leaves_recursive(self, node: DecompositionNode) -> int:
        if isinstance(node, (SingleQubitNode, IdentityNode)):
            return 1
        if isinstance(node, AIIINode):
            return self._count_leaves_recursive(node.k1) + self._count_leaves_recursive(node.k2)
        if isinstance(node, TypeANode):
            # W and X each appear twice (in both diagonal blocks)
            return 2 * (self._count_leaves_recursive(node.w) + self._count_leaves_recursive(node.x))
        return 0
    
    def get_depth(self) -> int:
        return self.decomposition_tree.get_depth()
    
    def count_gates(self) -> dict:
        return self.decomposition_tree.count_gates()
    
    def get_info(self) -> dict:
        return {
            'num_leaves': self.count_leaves(),
            'depth': self.get_depth(),
            'gate_counts': self.count_gates()
        }
    
    # Visualization
    
    def print_tree(self, node: DecompositionNode = None, level: int = 0):
        """Print the decomposition tree structure."""
        if node is None:
            node = self.decomposition_tree
        
        indent = "  " * level
        
        if isinstance(node, IdentityNode):
            print(f"{indent}Identity({node.dimension}x{node.dimension})")
        elif isinstance(node, SingleQubitNode):
            gates_str = " → ".join(f"{g}({a:.3f})" if a else g for g, a in node.gates) or "I"
            print(f"{indent}SingleQubit: {gates_str}")
        elif isinstance(node, AIIINode):
            print(f"{indent}AIII({node.dimension}x{node.dimension}):")
            print(f"{indent}  θ: {node.theta}")
            print(f"{indent}  K₁:"); self.print_tree(node.k1, level + 2)
            print(f"{indent}  K₂:"); self.print_tree(node.k2, level + 2)
        elif isinstance(node, TypeANode):
            print(f"{indent}TypeA({node.dimension}x{node.dimension}):")
            print(f"{indent}  φ: {node.phi}")
            print(f"{indent}  W:"); self.print_tree(node.w, level + 2)
            print(f"{indent}  X:"); self.print_tree(node.x, level + 2)
    
    def print_leaves(self):
        """Print details of all leaf nodes."""
        leaves = []
        self._collect_leaves(self.decomposition_tree, leaves)
        print("\nLeaf Details:")
        for i, leaf in enumerate(leaves, 1):
            if isinstance(leaf, IdentityNode):
                print(f"  Leaf {i}: Identity")
            elif isinstance(leaf, SingleQubitNode):
                gates_str = ", ".join(f"{g}({a:.3f})" if a else g for g, a in leaf.gates)
                print(f"  Leaf {i}: {gates_str}")
    
    def _collect_leaves(self, node: DecompositionNode, leaves: list):
        if isinstance(node, (SingleQubitNode, IdentityNode)):
            leaves.append(node)
        elif isinstance(node, AIIINode):
            self._collect_leaves(node.k1, leaves)
            self._collect_leaves(node.k2, leaves)
        elif isinstance(node, TypeANode):
            self._collect_leaves(node.w, leaves)
            self._collect_leaves(node.x, leaves)
    
    def get_raw_unitaries(self) -> list:
        """Collect the raw 2×2 unitaries from all leaf nodes."""
        leaves = []
        self._collect_leaves(self.decomposition_tree, leaves)
        return [leaf.raw_unitary for leaf in leaves if leaf.raw_unitary is not None]
    
    def print_raw_unitaries(self):
        """Print the original 2×2 unitaries that arrive at the leaves before gate translation."""
        leaves = []
        self._collect_leaves(self.decomposition_tree, leaves)
        print("\nRaw 2×2 Unitaries (before gate translation):")
        for i, leaf in enumerate(leaves, 1):
            if leaf.raw_unitary is not None:
                label = "Identity" if isinstance(leaf, IdentityNode) else "SingleQubit"
                print(f"  Leaf {i} ({label}):")
                U = leaf.raw_unitary
                print(f"    [{U[0,0]:+.4f}  {U[0,1]:+.4f}]")
                print(f"    [{U[1,0]:+.4f}  {U[1,1]:+.4f}]")
            else:
                print(f"  Leaf {i}: (no raw unitary stored)")
    
    def print_multi_qubit_structure(self):
        """Print which qubits each operation acts on."""
        print("\n" + "="*70)
        print("MULTI-QUBIT STRUCTURE")
        print("="*70)
        self._print_structure(self.decomposition_tree, list(range(self.nqubits)))
        print("="*70)
    
    def _print_structure(self, node: DecompositionNode, wires: List[int], indent: int = 0):
        prefix = "  " * indent
        wire_str = f"qubit {wires[0]}" if len(wires) == 1 else f"qubits {wires}"
        
        if isinstance(node, IdentityNode):
            print(f"{prefix}➤ I on {wire_str}")
        elif isinstance(node, SingleQubitNode):
            gates = " → ".join(f"{g}({a:.2f})" if a else g for g, a in node.gates) or "I"
            print(f"{prefix}➤ {gates} on {wire_str}")
        elif isinstance(node, AIIINode):
            print(f"{prefix}AIII on {wire_str}:")
            print(f"{prefix}  K₂:")
            self._print_structure(node.k2, wires, indent + 2)
            active = sum(1 for t in node.theta if abs(t) > self.ANGLE_TOL)
            if active:
                print(f"{prefix}  A-block: {active} CRY gates")
            print(f"{prefix}  K₁:")
            self._print_structure(node.k1, wires, indent + 2)
        elif isinstance(node, TypeANode):
            n = len(wires)
            upper, lower = wires[:n//2], wires[n//2:]
            print(f"{prefix}TypeA on {wire_str}:")
            print(f"{prefix}  X on qubits {upper} and {lower}:")
            self._print_structure(node.x, upper, indent + 2)
            active = sum(1 for p in node.phi if abs(p) > self.ANGLE_TOL)
            if active:
                print(f"{prefix}  A-block: {active} CRZ gates")
            print(f"{prefix}  W on qubits {upper} and {lower}:")
            self._print_structure(node.w, upper, indent + 2)
