import numpy as np
from scipy.linalg import cossin, expm

"""
The general BDI structure-preserving decomposition 
is based on the work by Wierichs et al. [1].
"""


def block_diag(*blocks):
    """Create a block-diagonal matrix from the given blocks."""
    total = sum(b.shape[0] for b in blocks)
    out = np.zeros((total, total), dtype=blocks[0].dtype)
    i = 0
    for b in blocks:
        n = b.shape[0]
        out[i:i + n, i:i + n] = b
        i += n
    return out


def bdi(U, p=None, q=None):
    """
    CS decomposition for a unitary/orthogonal matrix U of size (p+q) x (p+q).

    Returns k11, k12, theta, k21, k22 for U = KAK
    """
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U must be a square matrix")

    dim = U.shape[0]

    if p is None or q is None:
        p = dim // 2
        q = dim - p

    if p + q != dim:
        raise ValueError("p + q must equal U.shape[0]")

    # Verify unitary
    if not np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-10):
        raise ValueError("U is not unitary")

    # CS decomposition, use q=p for balanced split
    (k11, k12), theta, (k21, k22) = cossin(U, p=p, q=p, separate=True)

    # Extract determinant signs to verify all blocks are in SO group (det=+1)
    def get_det_sign(m, tol=1e-8):
        d = np.linalg.det(m)
        d = np.real_if_close(d, tol=1000)
        if np.iscomplexobj(d):
            raise ValueError(f"Determinant has imaginary part: det={d}")
        d = float(np.real(d))
        return 1.0 if d > 0 else -1.0

    # Get signs of determinants
    s11 = get_det_sign(k11)
    s12 = get_det_sign(k12)
    s21 = get_det_sign(k21)
    s22 = get_det_sign(k22)

    # Apply sign-based determinant normalization
    # Flip the first column/row of each block by its sign to ensure det = +1
    k11[:, 0] *= s11
    k12[:, q - min(p, q)] *= s12
    k21[0, :] *= s21
    
    row22 = q - min(p, q)
    if row22 < k22.shape[0]:
        k22[row22, :] *= s22
    else:
        k22[0, :] *= s22

    # Adjust theta to compensate for the sign changes
    theta[0] *= s11 * s12
    if s11 * s21 < 0:
        theta[0] += np.pi

    # After sign-flipping, all blocks should have det close to +1
    # We only verify that the flipping worked, not that products match
    assert np.isclose(np.linalg.det(k11), 1.0, atol=1e-6), \
        f"k11 det={np.linalg.det(k11)} after flipping"
    assert np.isclose(np.linalg.det(k12), 1.0, atol=1e-6), \
        f"k12 det={np.linalg.det(k12)} after flipping"
    assert np.isclose(np.linalg.det(k21), 1.0, atol=1e-6), \
        f"k21 det={np.linalg.det(k21)} after flipping"
    assert np.isclose(np.linalg.det(k22), 1.0, atol=1e-6), \
        f"k22 det={np.linalg.det(k22)} after flipping"
    
    return k11, k12, theta, k21, k22


def from_generator(generator, t=1.0):
    """Build the group element exp(t * generator) used as BDI input."""
    return expm(t * generator)


def build_kak(k11, k12, theta, k21, k22):
    """
    Build K1, A, K2 factors from one CS split.
    """
    p = k11.shape[0]
    q = k12.shape[0]
    K1 = block_diag(k11, k12)
    K2 = block_diag(k21, k22)
    c = np.diag(np.cos(theta))
    s = np.diag(np.sin(theta))
    A = np.block([
        [c, -s],
        [s, c],
    ])

    return K1, A, K2


def recursive_bdi(U, num_iter=None, return_all=False):
    """
    Recursively apply BDI decomposition to U, returning a list of operations at the end..

    Each op is a tuple: (matrix_or_angles, start, end, type)
    where type is one of: k1, k2, a, a0.
    """
    dim = U.shape[0]
    p = dim // 2
    q = dim - p
    k11, k12, theta, k21, k22 = bdi(U, p, q)

    # Dictionary to hold operations at each level of recursion
    ops_by_level = {
        -1: [(U, 0, dim, None)],
        0: [
            (k11, 0, p, "k1"),
            (k12, p, dim, "k1"),
            (theta, 0, dim, "a0"),
            (k21, 0, p, "k2"),
            (k22, p, dim, "k2"),
        ],
    }
    """
    Note that we store the first angle level as "a0" to distinguish it from later angles
    This is because the "0" level corresponds to the original unitary decomposition,
    such that the angles at this level are the only ones that are tied to "time". Later angles
    are just angles from decomposing the k1 and k2 factors, and are not directly related to the original time evolution.
    In mapping back to Pauli rotations, we rescale the "a0" angles by the original time t, while the lager
    "a" angles are not rescaled.
    """

    current_ops = ops_by_level[0]
    it = 0

    while True:
        decomposed_any = False
        new_ops = []

        for op, start, end, op_type in current_ops:
            size = end - start

            if size <= 1:
                new_ops.append((op, start, end, op_type))
                continue

            if op_type is not None and op_type.startswith("a"):
                new_ops.append((op, start, end, op_type))
                continue

            if size == 2:
                new_ops.append((op, start, end, op_type))
                continue

            if not isinstance(op, np.ndarray) or op.shape[0] != size:
                new_ops.append((op, start, end, op_type))
                continue

            sub_p = size // 2
            sub_q = size - sub_p
            
            d11, d12, dtheta, d21, d22 = bdi(op, sub_p, sub_q)

            new_ops.extend([
                (d11, start, start + sub_p, "k1"),
                (d12, start + sub_p, end, "k1"),
                (dtheta, start, end, "a"),
                (d21, start, start + sub_p, "k2"),
                (d22, start + sub_p, end, "k2"),
            ])
            decomposed_any = True

        it += 1

        if return_all:
            ops_by_level[it] = list(new_ops)

        current_ops = new_ops

        if not decomposed_any:
            # We could not decompose further
            break

        if num_iter is not None and it >= num_iter:
            # We have reached the maximum number of iterations
            break

    if return_all:
        return ops_by_level
    
    return current_ops



