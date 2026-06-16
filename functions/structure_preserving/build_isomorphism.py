import numpy as np

def map_to_majarana(generators, J = 1.0, h = 1.0):
    """
    Input: rotated generators of a Hamiltonian in terms of Pauli words 
    Output: Corresponding majarana indicies
    """
    mapping = {}
    for gen in generators: 
        positions = []
        paulis = []

        # Loop through all Pauli operators in the Pauli word
        for j, pauli in enumerate(gen):

            # Find the Pauli operators that are not Identity
            if pauli != "I":
                positions.append(j)
                paulis.append(gen[j])
            
        # When we have single Z_js
        if len(positions) == 1:

            # Since we loop from k = 1
            j = positions[0]
            mapping[gen] = (-h, (2*j, 2*j + 1))
        
        # When we have X_jX_j+1
        elif len(positions) ==2:
            j1, j2 = positions[0], positions[1]
            mapping[gen] = (J, (2*j1, 2*j2 + 1))
    return mapping

# Define isomorphism 
def isomorphism(indicies, n):
    i,j = indicies

    # Convert back to numpy indicing 
    i = i
    j = j
    dim = 2*n

    # Define the elementry matricies
    Eij = np.zeros((dim,dim))
    Eji = np.zeros((dim,dim))
    Eij[i,j] = 1
    Eji[j,i] = 1

    # Return F 
    return 2*(Eij - Eji)

# Build the matrix from the isomorphism 
def build_so_matrix(maj_mapping, n):
    dim = 2*n
    matrix = np.zeros((dim,dim))

    for coeff, indicies in maj_mapping.values():
        matrix += coeff * isomorphism(indicies, n)
    return matrix
