"""Unitary synthesis for the Transverse-Field Ising Model.

Three decomposition methods, each in its own subpackage:

    functions.structure_preserving  -- exact BDI/KAK decomposition via so(2n)
    functions.naive                 -- general arbitrary-unitary synthesis (AIII/Type-A)
    functions.trotter               -- first-order Trotter product formula
    functions.common                -- shared helpers (Hamiltonian, gate counting)
"""
