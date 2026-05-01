import numpy as np

def pw_to_majorana(pw):
    n = len(pw)

    positions = []
    for j, pauli in enumerate(pw):
        if pauli != "I":
            positions.append(j)
    
    # Single Z - operator
    if len(positions) == 1:
        j = positions[0]
        if pw[j] != "Z":
            raise ValueError(f"Error in PauliWord")
        return -1, (2*j, 2*j+1)
    
    # Two operators with Z or nothing inbetween
    left, right = positions[0], positions[-1]

    L, R = pw[left], pw[right]

    # The four different cases
    if L == "Y" and R == "X":
        return -1, (2 * left + 1, 2* right+1)
    if L == "X" and R =="X":
        return +1, (2 * left, 2* right +1 )
    if L == "Y" and R == "Y":
        return -1, (2 * left+1, 2* right)
    if L == "X" and R =="Y":
        return +1, (2 * left, 2* right )


def build_majorana_dla_map(DLA_generators):
    map = {}

    for gen in DLA_generators:
        sign, maj_indicies = pw_to_majorana(gen)

        # Convert back from numpy indicing 
        #i = maj_indicies[0]
        #j = maj_indicies[1] 
        map[maj_indicies] = (gen,sign)
    return map

def angles_to_reducible(angles, start, end, mapping):

    op = {}
    size = end - start
    # scipy CS layout: rotation rows of the A-block are (k, q+k) where q = size - size//2.
    # For even size, q = size//2 so this matches a symmetric p|p split; for odd size
    # we must use q (the larger half).
    q = size - size // 2

    for k, angle in enumerate(angles):
        i = start + k
        j = start + q + k

        pauli_word, sign = mapping[(i,  j)]
        op[pauli_word] = angle / (2 * sign)

    return op


def group_matrix_to_reducible(matrix, start,mapping, tol=1e-8):

    op = {}
    seen_ids = set()

    rows, cols = np.where(np.abs(matrix) > tol)

    for n, m in zip(rows, cols):
        if n >= m:
            continue

        if n in seen_ids or m in seen_ids:
            raise ValueError(f"indices {n}, {m} reused in commuting block:\n{matrix}")

        # K-block 2x2 rotation [[c,-s],[s,c]] has arctan2(M[0,1], M[0,0]) = -theta;
        # negate so 'angle' is the SO rotation angle theta.
        angle = float(-np.arctan2(matrix[n, m], matrix[n, n]))

        i = start + n
        j = start + m

        pauli_word, sign = mapping[(i, j)]
        #op[pauli_word] = angle / (2 * signs[(i, j)])
        op[pauli_word] = angle / (2 * sign)


        seen_ids.update({n, m})

    return op

def map_ops_to_pauli(recursive_decomp, mapping, time=None, tol=1e-8):
    pauli_decomp = []

    for matrix_or_angles, start, end, op_type in recursive_decomp:
        if op_type.startswith("a"):
            ps = angles_to_reducible(matrix_or_angles, start, end, mapping)
        else:
            ps = group_matrix_to_reducible(matrix_or_angles, start, mapping)
        if op_type == "a0" and time is not None:
            ps = {word: coeff / time for word, coeff in ps.items()}


        pauli_decomp.extend((word, coeff, op_type) for word, coeff in ps.items())

    return pauli_decomp