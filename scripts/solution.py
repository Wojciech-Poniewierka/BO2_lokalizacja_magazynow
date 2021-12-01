#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import random
import numpy as np

from typing import Tuple

# PROJECT MODULES
from config import MUTATION_RATIO, NOISE, CONSTRAINT_ACCURACY


# CLASSES
class Solution:
    """
    Class to represent the solution
    """

    def __init__(self, n_warehouses: int, n_shops: int, f: np.ndarray, s: np.ndarray, c: np.ndarray, b: np.ndarray,
                 d: np.ndarray, v: np.ndarray) -> None:
        """
        Constructor
        :param n_warehouses: Number of warehouses
        :param n_shops: Number of shops
        :param f: Factory to warehouses transport costs, dim: Mx1
        :param s: Warehouses to shops transport costs, dim: MxN
        :param c: Warehouses capacities, dim: Mx1
        :param b: Warehouses building costs, dim: Mx1
        :param d: Demands of the shops, dim: Nx1
        :param v: Sugar values established between warehouses and shops, dim: MxN
        """

        # Number of facilities
        self.M = n_warehouses
        self.N = n_shops

        # Problem parameters
        self.f = f
        self.S = s
        self.c = c
        self.b = b
        self.d = d
        self.V = v

        # Solution matrix
        self.X = np.random.uniform(size=(self.M, self.N))
        self.scale()

        while not self.is_feasible():
            self.X = np.random.uniform(size=(self.M, self.N))
            self.scale()

    def __eq__(self, other: "Solution") -> bool:
        """
        Magic method to compare the Solution instances
        :param other: Other instance
        :return: Flag informing if the Solution instances are equal
        """

        return (self.X == other.X).all()

    def __getitem__(self, coords: Tuple[int, int]) -> float:
        """
        Magic method to get the Solution matrix element
        :param coords: Indexes of the Solution matrix to get to the element
        :return: Solution matrix element
        """

        i, j = coords

        return self.X[i, j]

    def __str__(self) -> str:
        """
        Magic method to represent the class as a string
        :return: String representation of the solution matrix
        """

        return "\n".join([" ".join([str(round(elem, 2)).center(4) for elem in self.X[i, :]]) for i in range(self.M)])

    def calculate_fitness(self) -> float:
        """
        Method to calculate the objective function value
        :return: Objective function value
        """

        income = ((self.V * self.X) @ self.d).sum(axis=0)
        cost = np.dot(self.f + self.b + (np.ceil(self.X) * self.S).sum(axis=1), (self.X.sum(axis=1) > 0).astype(int))

        return income - cost

    def is_feasible(self) -> bool:
        """
        Method to check if the solution is feasible
        :return: Flag informing if the solution is feasible
        """

        decision_variables_constraint = ((0 <= self.X)  & (self.X <= 1)).all()
        shop_demand_constraint = np.allclose(self.X.sum(axis=0), np.ones(self.N), atol=CONSTRAINT_ACCURACY)
        capacity_constraint = (np.dot(self.X, self.d.T) <= self.c).all()

        return decision_variables_constraint and shop_demand_constraint and capacity_constraint

    def scale(self) -> None:
        """
        Method to scale the decision variables so that they meet the constraint
        """

        for j in range(self.N):
            col_sum = self.X[:, j].sum(axis=0)
            self.X[:, j] /= col_sum

    def mutate(self) -> None:
        """
        Method to mutate the solution
        """

        if random.uniform(0, 1) < MUTATION_RATIO:
            X = self.X
            self.X = np.random.normal(loc=X, scale=NOISE)
            self.scale()

            while not self.is_feasible():
                self.X = np.random.normal(loc=X, scale=NOISE)
                self.scale()