#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np

from typing import Tuple, Optional, Union, List
from copy import deepcopy
from random import shuffle

# PROJECT MODULES
from config import CONSTRAINT_ACCURACY
from data import ProblemSize, ProblemParameters, AlgorithmParameters


# CLASSES
class Solution:
    """
    Class to represent the solution
    """

    def __init__(self, problem_size: ProblemSize, problem_parameters: ProblemParameters,
                 algorithm_parameters: AlgorithmParameters, mat: Optional[np.ndarray] = None) -> None:
        """
        Constructor
        :param problem_size: Problem size
        :param problem_parameters: Tuple: (Factory to warehouses transport costs, shape: Mx1,
        Warehouses to shops transport costs, shape: MxN, Warehouses capacities, shape: Mx1,
        Warehouses building costs, shape: Mx1, Shops demands, shape: Nx1,
        Sugar values established between warehouses and shops, shape: MxN)
        :param algorithm_parameters: Algorithm parameters
        :param mat: Decision variables matrix
        """

        # Problem size
        self.problem_size = problem_size
        self.M = problem_size.M
        self.N = problem_size.N

        # Problem parameters
        self.problem_parameters = problem_parameters
        self.f, self.S, self.c = problem_parameters.f, problem_parameters.S, problem_parameters.c
        self.b, self.d, self.V = problem_parameters.b, problem_parameters.d, problem_parameters.V

        # Algorithm parameters
        self.algorithm_parameters = algorithm_parameters
        self.mutation_ratio = algorithm_parameters.mutation_ratio
        self.n_generations = algorithm_parameters.n_generations
        self.equality_penalty_coefficient = algorithm_parameters.equality_penalty_coefficient
        self.inequality_penalty_coefficient = algorithm_parameters.inequality_penalty_coefficient

        start_method = algorithm_parameters.methods[0]
        start_method_value = algorithm_parameters.methods_values[0]
        self.mutation_method = algorithm_parameters.methods[3]
        self.mutation_method_value = algorithm_parameters.methods_values[3]

        # Fitness
        self.fitness, self.penalty_equality, self.penalty_inequality = 0, 0, 0

        # Decision variables matrix
        if mat is None:
            if start_method == 0:
                while True:
                    if np.random.uniform() < start_method_value:
                        self.X = np.random.uniform(low=0, high=2 / self.M, size=(self.M, self.N))

                    else:
                        self.X = np.random.uniform(size=(self.M, self.N))

                        for j in range(self.N):
                            col_sum = self.X[:, j].sum()
                            self.X[:, j] /= col_sum

                    if (self.X @ self.d.T <= self.c).all():
                        break

            else:
                while True:
                    if np.random.uniform() < start_method_value:
                        self.X = np.random.uniform(low=0, high=2 / self.M, size=(self.M, self.N))

                    else:
                        self.X = np.zeros((self.M, self.N))

                        for j in range(self.N):
                            idx_to_omit = np.random.randint(self.M)
                            indexes: List[int] = [i for i in range(self.M) if i != idx_to_omit]
                            shuffle(indexes)

                            while indexes:
                                i = indexes.pop()
                                self.X[i, j] = np.random.uniform(low=0, high=1 - self.X[:, j].sum())

                            self.X[idx_to_omit, j] = 1 - self.X[:, j].sum()

                    if (self.X @ self.d.T <= self.c).all():
                        break

        else:
            self.X = mat

        self.calculate()

    def __eq__(self, other: "Solution") -> bool:
        """
        Magic method to compare the Solution instances
        :param other: Other Solution instance
        :return: Flag informing if the Solution instances are equal
        """

        return (self.X == other.X).all()

    def __gt__(self, other: "Solution") -> bool:
        """
        Magic method to compare the Solution instances
        :param other: Other Solution instance
        :return: Flag informing if the Solution 1 is greater than Solution 2
        """

        return (self.X > other.X).astype(int).sum() > 0.5 * self.M * self.N
    
    def __add__(self, other: "Solution") -> "Solution":
        """
        Magic method to add 2 Solution instances
        :param other: Other Solution instance
        :return: Sum of Solution instances matrices
        """

        return Solution(self.problem_size, self.problem_parameters, self.algorithm_parameters, mat=self.X + other.X)

    def __sub__(self, other: "Solution") -> "Solution":
        """
        Magic method to subtract 2 Solution instances
        :param other: Other Solution instance
        :return: Subtraction of Solution instances matrices
        """

        return Solution(self.problem_size, self.problem_parameters, self.algorithm_parameters, mat=self.X - other.X)

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

        X = np.zeros((self.M + 2, self.N)) - 1
        X[:self.M, :] = self.X
        X[self.M + 1, :] = self.X.sum(axis=0)

        return "\n".join(["".join(["||".center(9) if elem < 0 else
                                   str(round(elem, 2)).ljust(10 - len(str(round(elem, 2))))
                                   for elem in X[i, :]]) for i in range(self.M + 2)])

    def mul(self, factor: Union[int, float, np.ndarray]) -> "Solution":
        """
        Method to multiply the Solution instance by a factor
        :param factor: Scalar value or matrix
        :return: Multiplied Solution instance
        """

        return Solution(self.problem_size, self.problem_parameters, self.algorithm_parameters, mat=self.X * factor)

    def scale(self) -> "Solution":
        """
        Method to scale the Solution matrix
        :return: Scaled solution
        """

        mat = deepcopy(self.X)

        for j in range(self.N):
            col_sum = mat[:, j].sum()
            mat[:, j] /= col_sum

        return Solution(self.problem_size, self.problem_parameters, self.algorithm_parameters, mat=mat)

    def calculate(self) -> None:
        """
        Method to calculate the objective function and the penalty values
        """

        income = ((self.V * self.X) @ self.d.T).sum()
        are_located = (self.X.sum(axis=1) > 0).astype(int)
        cost = (are_located * (self.b + self.f + (np.ceil(self.X) * self.S).sum(axis=1).reshape(self.M, 1))).sum()
        # equality_constraint_diff = np.abs(self.X.sum(axis=0) - 1)
        # inequality_constraint_diff = np.maximum(np.zeros((self.M, 1)), self.X @ self.d.T - self.c)
        # self.penalty_equality = self.equality_penalty_coefficient * equality_constraint_diff.sum()**2
        # self.penalty_inequality = self.inequality_penalty_coefficient * inequality_constraint_diff.sum()**2
        equality_constraint_diff = np.power(self.X.sum(axis=0) - 1, 2)
        inequality_constraint_diff = np.power(np.maximum(np.zeros((self.M, 1)), self.X @ self.d.T - self.c), 2)
        self.penalty_equality = self.equality_penalty_coefficient * equality_constraint_diff.sum()
        self.penalty_inequality = self.inequality_penalty_coefficient * inequality_constraint_diff.sum()
        self.fitness = income - cost - self.penalty_equality - self.penalty_inequality

    def is_correct(self) -> bool:
        """
        Method to check if the solution is feasible
        :return: Flag informing if the solution is feasible
        """

        return ((0 <= self.X) & (self.X <= 1)).all()

    def is_feasible(self) -> bool:
        """
        Method to check if the solution is feasible
        :return: Flag informing if the solution is feasible
        """

        return self.is_correct() and (self.X @ self.d.T <= self.c).all() and (np.abs(self.X.sum(axis=0) - 1) < CONSTRAINT_ACCURACY * np.ones((1, self.problem_size.N))).all()

    def mutate(self, n_generation: int) -> "Solution":
        """
        Method to mutate the solution
        :param n_generation: Current generation number
        :return: Mutated solution
        """

        # Swap
        if self.mutation_method == 0:
            mat = deepcopy(self.X)

            while True:
                if np.random.uniform() >= self.mutation_ratio:
                    break

                idx = np.random.choice(self.N, size=2, replace=False)
                col1_idx = idx[0]
                col2_idx = idx[1]
                mat[:, [col1_idx, col2_idx]] = mat[:, [col2_idx, col1_idx]]

            return Solution(self.problem_size, self.problem_parameters, self.algorithm_parameters, mat=mat)

        # Borrow
        elif self.mutation_method == 1:
            mat = deepcopy(self.X)

            while np.random.uniform() < self.mutation_ratio:
                idx = np.random.choice(self.M, size=2, replace=False)
                row1_idx = idx[0]
                row2_idx = idx[1]
                dx = np.random.uniform(low=0, high=min(min(self.X[row2_idx, :]), 1 - max(self.X[row1_idx, :])))
                mat[row1_idx, :] = self.X[row1_idx, :] + dx
                mat[row2_idx, :] = self.X[row2_idx, :] - dx

            return Solution(self.problem_size, self.problem_parameters, self.algorithm_parameters, mat=mat)

        # # Non-uniform
        elif self.mutation_method == 2:
            b = self.mutation_method_value
            T = self.n_generations
            mat = deepcopy(self.X)

            for i in range(self.M):
                for j in range(self.N):
                    if np.random.uniform() < self.mutation_ratio:
                        if np.random.uniform() < 0.5:
                            alpha = 1 - mat[i, j]

                        else:
                            alpha = -mat[i, j]

                        r = np.random.uniform()
                        mat[i, j] = mat[i, j] + alpha * (1 - r**((1 - n_generation / T)**b))

            return Solution(self.problem_size, self.problem_parameters, self.algorithm_parameters, mat=mat)

        # Polynomial
        if self.mutation_method == 3:
            eta = self.mutation_method_value
            mat = deepcopy(self.X)

            for i in range(self.M):
                for j in range(self.N):
                    if np.random.uniform() < self.mutation_ratio:
                        u = np.random.uniform()

                        if u <= 0.5:
                            mat[i, j] = mat[i, j] + mat[i, j] * ((2 * u)**(1 / (1 + eta)) - 1)

                        else:
                            mat[i, j] = mat[i, j] + (1 - mat[i, j]) * (1 - (2 - 2 * u)**(1 / (1 + eta)))

            return Solution(self.problem_size, self.problem_parameters, self.algorithm_parameters, mat=mat)
