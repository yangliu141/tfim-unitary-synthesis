from itertools import combinations


# Function for writing the Hamiltonian in terms of Pauli Words
def tfim_pauliwords_gen(n, rotated = True, periodic = False):
    gens = []
    if rotated: 
        # XX couplings
        for i in range(n-1):
            pword = ["I"] * n
            pword[i] = "X"
            pword[i + 1] = "X"
            gens.append(tuple(pword))
        
        if periodic and n > 2:
            pword = ["I"] * n
            pword[-1] = "X"
            pword[0] = "X"
            gens.append(tuple(pword))
        
        # Z fields
        for i in range(n):
            pword = ["I"] * n
            pword[i] = "Z"
            gens.append(tuple(pword))
    
    else:
        # ZZ couplings
        for i in range(n-1):
            pword = ["I"] * n
            pword[i] = "Z"
            pword[i + 1] = "Z"
            gens.append(tuple(pword))
        
        if periodic and n > 2:
            pword = ["I"] * n
            pword[-1] = "Z"
            pword[0] = "Z"
            gens.append(tuple(pword))
        
        # Z fields
        for i in range(n):
            pword = ["I"] * n
            pword[i] = "X"
            gens.append(tuple(pword)) 
    return gens


# Define Pauli Word multiplications

_SINGLE_MUL = {
    ("I", "I"): (1, "I"),
    ("I", "X"): (1, "X"),
    ("I", "Y"): (1, "Y"),
    ("I", "Z"): (1, "Z"),

    ("X", "I"): (1, "X"),
    ("Y", "I"): (1, "Y"),
    ("Z", "I"): (1, "Z"),

    ("X", "X"): (1, "I"),
    ("Y", "Y"): (1, "I"),
    ("Z", "Z"): (1, "I"),

    ("X", "Y"): (1j, "Z"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "X"): (1j, "Y"),

    ("Y", "X"): (-1j, "Z"),
    ("Z", "Y"): (-1j, "X"),
    ("X", "Z"): (-1j, "Y"),
}

def multiply_pauli_words(pw1, pw2):
    """
    Multiply two Pauli words
    Input: pw1, pw2 as tuples 
    Returns: (phase, resulting_pauli_word)
    """

    phase = 1
    result = []

    for a,b in zip(pw1, pw2):
        ph, c = _SINGLE_MUL[(a,b)]
        phase *= ph
        result.append(c)
    return phase, tuple(result)

def are_commuting(pw1,pw2):
    # Count for how many operators anticommute
    count = 0

    for a,b in zip(pw1, pw2):

        # a and b commutes
        if a =="I" or b == "I" or a ==b:
            continue

        # else they anticommute
        count += 1
    # If even number anticommutations, then commute
    # If odd number of anticommutations, then anticommute
    return (count % 2) == 0

def commutator_pauli_words(pw1, pw2):

    if (are_commuting(pw1,pw2)):
        return None
    _, results = multiply_pauli_words(pw1,pw2)
    return results


def dla_pauli_words(generators, max_iterations = 1000, full_size = None):
    dla = list(generators)
    dla_set = set(dla)
    epoch = 0 
    changed = True
    while changed and epoch < max_iterations: 
        changed = False 
        current = list(dla)


        # Take all pairs of Pauliwords in the dla list
        for pw1, pw2 in combinations(current, 2):
            c = commutator_pauli_words(pw1,pw2)

            # Continue if the pauli words commute
            if c is None:
                continue
            
            # Add the commutator if it is not already in the dla set
            if c not in dla_set:
                dla.append(c)
                dla_set.add(c)
                changed = True

                # Finish if we reach the expected length of the DLA
                if full_size is not None and len(dla) >= full_size:
                    return dla
        epoch += 1
    
    return dla