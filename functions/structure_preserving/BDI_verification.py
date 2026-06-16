import numpy as np
from scipy.linalg import cossin, expm

def embed_block(block, start, end, dim):
    """Embed a smaller block into a larger identity matrix at the specified location."""

    M = np.eye(dim)
    M[start:end, start:end] = block
    return M

def angles_to_A_block(angles, size):
    """Convert a vector of angles to the corresponding A-block matrix."""
    p = size // 2
    q = size - p
    r = min(p, q)
    
    C = np.diag(np.cos(angles))
    S = np.diag(np.sin(angles))

    # General CS form for p <= q:
    # [[ C,  0, -S],
    #  [ 0,  I,  0],
    #  [ S,  0,  C]]
    mid = q - r
    Zrm = np.zeros((r, mid))
    Zmr = np.zeros((mid, r))
    Im  = np.eye(mid)

    top = np.hstack([C,   Zrm, -S])
    midb = np.hstack([Zmr, Im,  Zmr])
    bot = np.hstack([S,   Zrm,  C])

    return np.vstack([top, midb, bot])

def reconstruct_from_ops_in_order(ops, dim):
    """Reconstruct the unitary by multiplying the embedded blocks in the order they appear
    in the ops list."""
    U = np.eye(dim)

    for matrix_or_angles, start, end, op_type in ops:
        if op_type in ("k1", "k2"):
            block = matrix_or_angles

        elif op_type in ("a", "a0"):
            block = angles_to_A_block(matrix_or_angles, end - start)
        else:
            continue

        embedded_block = embed_block(block, start, end, dim)
        U = U @ embedded_block
    return U

def verify_bdi_decomposition(U_t, ops, tol=1e-8):
    """Verify that the BDI decomposition ops reconstructs U_t within tolerance."""
    dim = U_t.shape[0]
    U_reconstructed = reconstruct_from_ops_in_order(ops, dim)
    print("Max reconstruction error:", np.max(np.abs(U_t - U_reconstructed)))
    print("Recursive allclose:", np.allclose(U_t, U_reconstructed, atol=1e-8))