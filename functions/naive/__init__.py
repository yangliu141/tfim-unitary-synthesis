"""Naive arbitrary-unitary synthesis via recursive AIII / Type-A decomposition."""

from .decompose import decompose
from .single_qubit_decomposer import to_gates

__all__ = ["decompose", "to_gates"]
