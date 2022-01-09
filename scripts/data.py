#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np

from typing import Union, List


# CLASSES
class ProblemSize:
    """
    Class to store the problem size
    """

    def __init__(self, m: int, n: int) -> None:
        """
        Constructor
        :param m: Number of warehouses
        :param n: Number of shops
        """

        self.M = m
        self.N = n


class ProblemParameters:
    """
    Class to store the problem parameters
    """

    def __init__(self, f: np.ndarray, S: np.ndarray, c: np.ndarray, b: np.ndarray, d: np.ndarray, V: np.ndarray) -> None:
        """
        Constructor
        :param f: Factory to warehouses transport costs, shape: Mx1
        :param S: Warehouses to shops transport costs, shape: MxN
        :param c: Warehouses capacities, shape: Mx1
        :param b: Warehouses building costs, shape: Mx1
        :param d: Shops demands, shape: 1xN
        :param V: Costs established between warehouses and shops, shape: MxN
        """

        self.f = f
        self.S = S
        self.c = c
        self.b = b
        self.d = d
        self.V = V


class AlgorithmParameters:
    """
    Class to store the algorithm parameters
    """

    def __init__(self, mutation_ratio: float, population_size: int, n_generations: int, crossover_ratio: float,
                 penalty_coefficient: float, methods: List[int], methods_values: List[Union[int, float]]) -> None:
        """
        Constructor
        :param mutation_ratio: Mutation ratio
        :param population_size: Population size
        :param n_generations: Number of generations
        :param crossover_ratio: Crossover chance ratio
        :param penalty_coefficient: Penalty coefficient
        :param methods: Start solution, selection, crossover and mutation methods numbers
        :param methods_values: Start solution, selection, crossover and mutation methods values
        """

        self.mutation_ratio = mutation_ratio
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_ratio = crossover_ratio
        self.penalty_coefficient = penalty_coefficient
        self.methods = methods
        self.methods_values = methods_values
