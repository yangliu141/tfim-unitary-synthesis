import numpy as np
from typing import List, Tuple, Optional


class SingleQubitDecomposer:
    """
    Decomposes 2×2 unitary matrices into IBM-native gate sequences (Rz, SX, X).
    
    This is the "last mile" of unitary synthesis: once the recursive 
    AIII/TypeA decomposition has reduced everything to 2×2 unitaries,
    this class translates them into hardware-implementable gates.
    
    Returns:
        decompose(U) → None if identity, or (gates_list, global_phase)
        where gates_list is List[Tuple[str, Optional[float]]]
    """
    
    GATE_SIMPLIFY_TOL = 1e-8
    ANGLE_TOL = 1e-10

    def __init__(self, gates: dict = None):
        self.gates = gates if gates else self._init_gates()

    def _init_gates(self) -> dict:
        return {
            'SX': (1/np.sqrt(2)) * np.array([[1, -1j], [-1j, 1]]),
            'X': np.array([[0, 1], [1, 0]]),
            'Rz': lambda theta: np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)]
            ])
        }

    # Main Entry Point

    def decompose(self, U: np.ndarray) -> Optional[Tuple[List[Tuple[str, Optional[float]]], float]]:
        """
        Decompose a 2×2 unitary into native gates.
        
        Returns:
            None if U is (approximately) identity.
            (gates_list, global_phase) otherwise,
            where gates_list is e.g. [('Rz', 0.5), ('SX', None), ('Rz', 1.2), ...]
        """
        # Check 1: Identity
        if np.allclose(U, np.eye(2), atol=self.GATE_SIMPLIFY_TOL):
            return None
        
        # Check 2: X gate
        if self._is_x_gate(U):
            return [('X', None)], 0.0
        
        # Check 3: Pure Rz rotation
        rz_angle = self._extract_rz_angle(U)
        if rz_angle is not None:
            if abs(rz_angle) < self.ANGLE_TOL:
                return None  # Identity
            return [('Rz', rz_angle)], 0.0
        
        # Check 4: SX gate
        if self._is_sx_gate(U):
            return [('SX', None)], 0.0
        
        # Check 5: General ZXZ decomposition
        return self._zxz_decompose(U)

    # Gate Recognition

    def _is_x_gate(self, U: np.ndarray) -> bool:
        tol = self.GATE_SIMPLIFY_TOL
        return (abs(U[0,0]) < tol and abs(U[1,1]) < tol and
                abs(abs(U[0,1]) - 1) < tol and abs(abs(U[1,0]) - 1) < tol)
    
    def _extract_rz_angle(self, U: np.ndarray) -> Optional[float]:
        tol = self.GATE_SIMPLIFY_TOL
        if abs(U[0,1]) < tol and abs(U[1,0]) < tol and abs(U[0,0]) > tol:
            return float(2 * np.angle(U[1,1] / U[0,0]))
        return None
    
    def _is_sx_gate(self, U: np.ndarray) -> bool:
        SX = self.gates['SX']
        return any(np.allclose(U, p * SX, atol=self.GATE_SIMPLIFY_TOL) for p in [1, -1, 1j, -1j])

    # ZXZ Decomposition (General Case)

    def _zxz_decompose(self, U: np.ndarray) -> Optional[Tuple[List[Tuple[str, Optional[float]]], float]]:
        """
        Decompose U = e^(i*phase) * Rz(a) @ SX @ Rz(b) @ SX @ Rz(c).
        
        Uses numerical optimization for robust angle extraction.
        The gate sequence uses SX = (1/√2)[[1, -i], [-i, 1]] as the "X-like" rotation.
        
        Returns:
            None if U is identity, or (gates_list, global_phase).
        """
        from scipy.optimize import minimize
        
        SX = self.gates['SX']
        Rz = self.gates['Rz']
        
        # Get SU(2) parameters
        det_U = np.linalg.det(U)
        base_phase = float(np.angle(det_U) / 2)
        U_su2 = U * np.exp(-1j * base_phase)
        alpha, beta = U_su2[0, 0], U_su2[1, 0]
        
        cos_half_theta = np.clip(np.abs(alpha), 0, 1)
        theta = float(2 * np.arccos(cos_half_theta))
        
        # Special case: nearly pure Z rotation (no X component needed)
        if theta < self.ANGLE_TOL:
            z_angle = float(2 * np.angle(alpha))
            if abs(z_angle) < self.ANGLE_TOL:
                return None  # Identity
            return [('Rz', z_angle)], base_phase
        
        # Special case: nearly an X gate
        if abs(theta - np.pi) < self.ANGLE_TOL:
            return [('X', None)], base_phase + np.angle(beta)
        
        # General case: use optimization
        def objective(params):
            a, b, c = params
            computed = Rz(a) @ SX @ Rz(b) @ SX @ Rz(c)
            # Use least-squares phase
            phase = np.sum(np.conj(computed) * U) / np.sum(np.abs(computed)**2)
            return np.linalg.norm(U - phase * computed)**2
        
        # Try multiple initial guesses and take the best
        arg_alpha = np.angle(alpha) if np.abs(alpha) > 1e-10 else 0.0
        arg_beta = np.angle(beta) if np.abs(beta) > 1e-10 else 0.0
        
        initial_guesses = [
            [arg_alpha + arg_beta, theta, arg_alpha - arg_beta],
            [arg_alpha - arg_beta, theta, arg_alpha + arg_beta],
            [0, theta, 0],
            [-np.pi, theta, 0],
            [0, theta, -np.pi],
        ]
        
        best_result = None
        best_obj = float('inf')
        
        for x0 in initial_guesses:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 2000})
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
        
        a_opt, b_opt, c_opt = best_result.x
        
        # Build and simplify gates
        raw_gates = [('Rz', float(a_opt)), ('SX', None), ('Rz', float(b_opt)), ('SX', None), ('Rz', float(c_opt))]
        gates = self._simplify_gates(raw_gates)
        
        # Compute final gate product after simplification
        computed = np.eye(2, dtype=complex)
        for gate_name, angle in reversed(gates):
            if gate_name == 'Rz' and angle is not None:
                computed = Rz(angle) @ computed
            elif gate_name == 'SX':
                computed = SX @ computed
        
        # Get final phase
        phase = np.sum(np.conj(computed) * U) / np.sum(np.abs(computed)**2)
        total_phase = float(np.angle(phase))
        
        return gates, total_phase
    
    def _zxz_decompose_analytic(self, U: np.ndarray) -> Optional[Tuple[List[Tuple[str, Optional[float]]], float]]:
        """
        Analytic ZXZ decomposition
        """

    # Gate Simplification

    def _simplify_gates(self, gates: List[Tuple[str, Optional[float]]]) -> List[Tuple[str, Optional[float]]]:
        """Simplify gate sequence: merge adjacent Rz, normalize angles, prune near-zero."""
        tol = 1e-6  # Use a more generous tolerance for simplification
        result = []
        for gate, angle in gates:
            if gate == 'Rz':
                if angle is not None:
                    # Normalize angle to [-π, π]
                    angle = float(np.mod(angle + np.pi, 2 * np.pi) - np.pi)
                    # Skip if near zero
                    if abs(angle) < tol:
                        continue
                # Merge with previous Rz if possible
                if result and result[-1][0] == 'Rz':
                    merged = (result[-1][1] or 0) + (angle or 0)
                    # Normalize merged angle
                    merged = float(np.mod(merged + np.pi, 2 * np.pi) - np.pi)
                    if abs(merged) < tol:
                        result.pop()  # Remove the previous Rz entirely
                    else:
                        result[-1] = ('Rz', merged)
                    continue
            result.append((gate, angle))
        return [(g, a) for g, a in result if not (g == 'Rz' and a is not None and abs(a) < tol)]
