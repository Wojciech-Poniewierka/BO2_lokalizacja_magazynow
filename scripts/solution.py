#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np

from typing import Tuple, Optional


# CLASSES
class Solution:
    """
    Class to represent the solution
    """

    def __init__(self, shape: Tuple[int, int],
                 problem_parameters: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                 algorithm_parameters: Tuple[float, float, float, int, float, int, float, float, float,
                                             Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 mat: Optional[np.ndarray] = None) -> None:
        """
        Constructor
        :param shape: Problem shape
        :param problem_parameters: Tuple: (Factory to warehouses transport costs, shape: Mx1,
        Warehouses to shops transport costs, shape: MxN, Warehouses capacities, shape: Mx1,
        Warehouses building costs, shape: Mx1, Shops demands, shape: Nx1,
        Sugar values established between warehouses and shops, shape: MxN)
        :param algorithm_parameters: Tuple: (Mutation_ratio, Noise, Constraint_accuracy, Population_size, Min_fitness,
        Max_generations, Crossover_ratio, Transport_cost_amplifier, Building_cost_amplifier, Capacity_range,
        Demand_range, Cost_range)
        :param mat: Decision variables matrix
        """

        # Number of facilities
        self.M, self.N = shape

        # Problem parameters
        self.f, self.S, self.c, self.b, self.d, self.V = problem_parameters

        # Algorithm parameters
        self.mutation_ratio = algorithm_parameters[0]
        self.noise = algorithm_parameters[1]
        self.constraint_accuracy = algorithm_parameters[2]

        # Solution matrix
        if mat is None:
            self.X = np.random.normal(size=(self.M, self.N))
            self.scale()

            while not self.is_feasible():
                self.X = np.random.normal(size=(self.M, self.N))
                self.scale()

        else:
            self.X = mat

        self.fitness = self.calculate_fitness()

    def __eq__(self, other: "Solution") -> bool:
        """
        Magic method to compare the Solution instances
        :param other: Other Solution instance
        :return: Flag informing if the Solution instances are equal
        """

        return (self.X == other.X).all()
    
    def __add__(self, other: "Solution") -> "Solution":
        """
        Magic method to add 2 Solution instances
        :param other: Other Solution instance
        :return: Sum of Solution instances matrices
        """

        self.X += other.X
        self.fitness = self.calculate_fitness()

        return self

    def __sub__(self, other: "Solution") -> "Solution":
        """
        Magic method to subtract 2 Solution instances
        :param other: Other Solution instance
        :return: Subtraction of Solution instances matrices
        """

        self.X -= other.X
        self.fitness = self.calculate_fitness()

        return self

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

        return "\n".join([" ".join([str(round(elem, 2)).center(6) for elem in self.X[i, :]]) for i in range(self.M)])

    def mul(self, number: float) -> "Solution":
        """
        Method to multiply the Solution instance by a constant value
        :param number: Constant value
        :return: Multiplied Solution instance
        """

        self.X *= number
        self.fitness = self.calculate_fitness()

        return self

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
        shop_demand_constraint = np.allclose(self.X.sum(axis=0), np.ones(self.N), atol=self.constraint_accuracy)
        capacity_constraint = (np.dot(self.X, self.d.T) <= self.c).all()

        return decision_variables_constraint and shop_demand_constraint and capacity_constraint

    def scale(self) -> None:
        """
        Method to scale the decision variables so that they meet the constraint
        """

        for j in range(self.N):
            self.X[:, j] = np.abs(self.X[:, j])

            col_sum = self.X[:, j].sum(axis=0)
            self.X[:, j] /= col_sum

    def mutate(self) -> None:
        """
        Method to mutate the solution
        """

        if np.random.normal() < self.mutation_ratio:
            X = self.X
            self.X = np.random.normal(loc=X, scale=self.noise)
            self.scale()

            while not self.is_feasible():
                self.X = np.random.normal(loc=X, scale=self.noise)
                self.scale()
