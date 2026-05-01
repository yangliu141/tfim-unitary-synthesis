import numpy as np

# Pauli X gate on qubit n
def X(n,qubits):

    #Define Pauli X and identity gate
    X_gate = np.array([[0,1],[1,0]])
    I_gate = np.array([[1,0],[0,1]])

    operations = [I_gate] * qubits
    operations[n] = X_gate

    Xn = operations[0]
    for i in range(1, qubits):
        Xn = np.kron(Xn, operations[i])
    return Xn

# Pauli Z gate on qubit n    
def Z(n,qubits):

    #Define Pauli X and identity gate
    Z_gate = np.array([[1,0],[0,-1]])
    I_gate = np.array([[1,0],[0,1]])

    operations = [I_gate] * qubits
    operations[n] = Z_gate

    Zn = operations[0]
    for i in range(1, qubits):
        Zn = np.kron(Zn, operations[i])
    return Zn

# Find the generators in terms of matrices for the transverse field Ising model
def TFIM_generators(qubits,rotated = True, periodic=False):
    generators = []
    if rotated:
        # XX couplings
        for i in range(qubits - 1):
            generators.append(X(i, qubits) @ X(i + 1, qubits))

        # periodic XX coupling
        if periodic and qubits > 2:
            generators.append(X(qubits - 1, qubits) @ X(0, qubits))

        # Z fields
        for i in range(qubits):
            generators.append(Z(i, qubits))
    
    else:
        # ZZ couplings
        for i in range(qubits - 1):
            generators.append(Z(i, qubits) @ Z(i + 1, qubits))

        # periodic ZZ coupling
        if periodic and qubits > 2:
            generators.append(Z(qubits - 1, qubits) @ Z(0, qubits))

        # X fields
        for i in range(qubits):
            generators.append(X(i, qubits)) 
        


    return generators

# Find the Hamiltonian matrix for the transverse field ising model
def TFIM_Ham(qubits, J=1.0, h=1.0, rotated = True, periodic=False):
    dim = 2**qubits
    H = np.zeros((dim, dim))
    if rotated:
        # XX coupling
        for i in range(qubits - 1):
            H -= J * (X(i, qubits) @ X(i + 1, qubits))
        # periodic XX coupling
        if periodic and qubits > 2:
            H -= J * (X(qubits - 1, qubits) @ X(0, qubits))
        # Z fields
        for i in range(qubits):
            H -= h * Z(i, qubits)
    else:
        # ZZ couplings
        for i in range(qubits - 1):
            H -= J * (Z(i, qubits) @ Z(i + 1, qubits))

        # periodic ZZ coupling
        if periodic and qubits > 2:
            H -= J * (Z(qubits - 1, qubits) @ Z(0, qubits))

        # X fields
        for i in range(qubits):
            H -= h * X(i, qubits) 


    return H