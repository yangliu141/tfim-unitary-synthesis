"""
PennyLane circuit generation from the decomposition tree.

Converts the abstract decomposition tree (AIIINode, TypeANode, SingleQubitNode, 
IdentityNode) into a concrete PennyLane quantum circuit with IBM-native gates.
"""

import pennylane as qml
from typing import List

from decomposition_nodes import (
    DecompositionNode, SingleQubitNode, IdentityNode, AIIINode, TypeANode
)


class CircuitBuilder:
    """
    Converts a decomposition tree into a PennyLane circuit.
    
    Handles the mapping from abstract controlled rotations (CRY, CRZ)
    in the AIII/TypeA A-blocks to concrete PennyLane gate calls.
    """
    
    ANGLE_TOL = 1e-10
    
    def __init__(self, nqubits: int, decomposition_tree: DecompositionNode):
        self.nqubits = nqubits
        self.decomposition_tree = decomposition_tree
    
    def to_pennylane_circuit(self):
        """Convert decomposition to a PennyLane circuit."""
        dev = qml.device('default.qubit', wires=self.nqubits)
        
        tree = self.decomposition_tree
        builder = self
        
        @qml.qnode(dev)
        def circuit():
            builder._apply_node(tree, list(range(builder.nqubits)))
            return qml.state()
        
        return circuit
    
    def draw_circuit(self):
        """Draw the decomposed circuit."""
        circuit = self.to_pennylane_circuit()
        return qml.draw_mpl(circuit)()
    
    def _apply_node(self, node: DecompositionNode, wires: List[int]):
        """Recursively apply gates from decomposition tree."""
        if isinstance(node, IdentityNode):
            return
        
        if isinstance(node, SingleQubitNode):
            for gate_name, angle in node.gates:
                if gate_name == 'Rz' and angle is not None:
                    qml.RZ(angle, wires=wires[0])
                elif gate_name == 'SX':
                    qml.SX(wires=wires[0])
                elif gate_name == 'X':
                    qml.PauliX(wires=wires[0])
            return
        
        if isinstance(node, AIIINode):
            # Apply K₂ 
            self._apply_node(node.k2, wires)
            
            # Apply A block (uniformly controlled RY rotations)
            self._apply_aiii_a_block(node.theta, wires)
            
            # Apply K₁
            self._apply_node(node.k1, wires)
            return
        
        if isinstance(node, TypeANode):
            n = len(wires)
            upper_wires, lower_wires = wires[:n//2], wires[n//2:]
            
            # Apply K₂ = [[X, 0], [0, X]] - same X on both halves
            self._apply_node(node.x, upper_wires)
            self._apply_node(node.x, lower_wires)
            
            # Apply A block (uniformly controlled RZ rotations)
            self._apply_type_a_block(node.phi, wires)
            
            # Apply K₁ = [[W, 0], [0, W]] - same W on both halves
            self._apply_node(node.w, upper_wires)
            self._apply_node(node.w, lower_wires)
            return
    
    def _apply_aiii_a_block(self, theta, wires: List[int]):
        """
        Apply the AIII A matrix as uniformly controlled RY rotations.
        A = [[C, -S], [S, C]] where C=diag(cos θ), S=diag(sin θ)
        """
        if len(wires) < 2:
            return
        
        control = wires[0]
        half = len(wires) // 2
        
        for i, angle in enumerate(theta):
            if abs(angle) < self.ANGLE_TOL:
                continue
            target = wires[half + (i % half)] if half + (i % half) < len(wires) else wires[-1]
            qml.CRY(2 * angle, wires=[control, target])
    
    def _apply_type_a_block(self, phi, wires: List[int]):
        """
        Apply the Type A A matrix as uniformly controlled RZ rotations.
        A = [[D, 0], [0, D†]] where D=diag(e^{-iφ})
        """
        if len(wires) < 2:
            return
        
        control = wires[0]
        half = len(wires) // 2
        
        for i, angle in enumerate(phi):
            if abs(angle) < self.ANGLE_TOL:
                continue
            target = wires[half + (i % half)] if half + (i % half) < len(wires) else wires[-1]
            qml.CRZ(-2 * angle, wires=[control, target])
