#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np

from typing import Tuple, Optional, Union, List

# PROJECT MODULES
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
        self.f = problem_parameters.f
        self.S = problem_parameters.S
        self.c = problem_parameters.c
        self.b = problem_parameters.b
        self.d = problem_parameters.d
        self.V = problem_parameters.V

        # Algorithm parameters
        self.algorithm_parameters = algorithm_parameters
        self.mutation_ratio = algorithm_parameters.mutation_ratio
        self.n_generations = algorithm_parameters.n_generations
        self.equality_penalty_coefficient = algorithm_parameters.equality_penalty_coefficient
        self.inequality_penalty_coefficient = algorithm_parameters.inequality_penalty_coefficient
        self.mutation_method = algorithm_parameters.methods[3]
        self.mutation_method_value = algorithm_parameters.methods_values[3]

        # Fitness
        self.fitness, self.penalty_equality, self.penalty_inequality = 0, 0, 0

        # Decision variables matrix
        if mat is None:
            while True:
                self.X = np.random.uniform(size=(self.M, self.N))

                for j in range(self.N):
                    col_sum = self.X[:, j].sum()
                    self.X[:, j] /= col_sum

                if (self.X @ self.d.T <= self.c).all():
                    break

            # self.X = np.zeros((self.M, self.N))
            # indexes_to_omit: List[Tuple[int, int]] = [(np.random.randint(self.M), j) for j in range(self.N)]
            #
            # while True:
            #     indexes_to_draw = [(i, j) for j in range(self.N) for i in range(self.M) if (i, j) not in indexes_to_omit]
            #
            #     while indexes_to_draw:
            #         i, j = indexes_to_draw.pop(np.random.randint(len(indexes_to_draw)))
            #         self.X[i, j] = np.random.uniform(low=0, high=1 - self.X[:, j].sum())
            #
            #     for i, j in indexes_to_omit:
            #         self.X[i, j] = 1 - self.X[:, j].sum()
            #
            #     if (self.X @ self.d.T <= self.c).all():
            #         break

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

    def calculate(self) -> None:
        """
        Method to calculate the objective function and the penalty values
        """

        income = (self.V * self.X @ self.d.T).sum()
        are_located = (self.X.sum(axis=1) > 0).astype(int)
        cost = (are_located * (self.b + self.f + (np.ceil(self.X) * self.S).sum(axis=1).reshape(self.M, 1))).sum()
        equality_constraint_diff = self.X.sum(axis=0) - 1
        inequality_constraint_diff = np.maximum(np.zeros((self.M, self.N)), self.X @ self.d.T - self.c)
        self.penalty_equality = self.equality_penalty_coefficient**3 * abs(equality_constraint_diff.sum())
        self.penalty_inequality = self.inequality_penalty_coefficient**3 * abs(inequality_constraint_diff.sum())
        self.fitness = income - cost - self.penalty_equality - self.penalty_inequality

    def is_feasible(self) -> bool:
        """
        Method to check if the solution is feasible
        :return: Flag informing if the solution is feasible
        """

        return ((0 <= self.X) & (self.X <= 1)).all()

    def mutate(self, n_generation: int) -> None:
        """
        Method to mutate the solution
        :param n_generation: Current generation number
        """

        # Swap
        if self.mutation_method == 0:
            if np.random.uniform() < self.mutation_ratio:
                mat = self.X
                n_indexes = np.random.randint(self.M)
                indexes = list(np.random.choice(self.M, size=n_indexes, replace=False))

                while len(indexes) > 1:
                    row1_idx = indexes.pop()
                    n_idx = np.random.randint(len(indexes))
                    row2_idx = indexes.pop(n_idx)

                    mat[row1_idx, :] = self.X[row2_idx, :]
                    mat[row2_idx, :] = self.X[row1_idx, :]

                self.X = mat

        # Borrow
        elif self.mutation_method == 1:
            if np.random.uniform() < self.mutation_ratio:
                mat = self.X
                n_indexes = np.random.randint(self.M)
                indexes = list(np.random.choice(self.M, size=n_indexes, replace=False))

                while len(indexes) > 1:
                    row1_idx = indexes.pop()
                    n_idx = np.random.randint(len(indexes))
                    row2_idx = indexes.pop(n_idx)
                    value = np.random.uniform(high=min(min(self.X[row2_idx, :]), 1 - max(self.X[row1_idx, :])))

                    mat[row1_idx, :] = self.X[row1_idx, :] + value
                    mat[row2_idx, :] = self.X[row2_idx, :] - value

                self.X = mat

        # Non-uniform
        elif self.mutation_method == 2:
            b = self.mutation_method_value

            for i in range(self.M):
                for j in range(self.N):
                    if np.random.uniform() < self.mutation_ratio:
                        if np.random.uniform() < 0.5:
                            alpha = 1 - self.X[i, j]

                        else:
                            alpha = -self.X[i, j]

                        beta = np.random.uniform()

                        self.X[i, j] = self.X[i, j] + alpha * (1 - beta**(1 - n_generation / self.n_generations)**b)

        # Polynomial
        if self.mutation_method == 3:
            eta = self.mutation_method_value

            for i in range(self.M):
                for j in range(self.N):
                    if np.random.uniform() < self.mutation_ratio:
                        u = np.random.uniform()

                        if u <= 0.5:
                            self.X[i, j] = self.X[i, j] + (2 * u)**(1 / (1 + eta)) - 1

                        else:
                            self.X[i, j] = self.X[i, j] + (1 - (2 - 2 * u)**(1 / (1 + eta))) * (1 - self.X[i, j])

            self.calculate()
