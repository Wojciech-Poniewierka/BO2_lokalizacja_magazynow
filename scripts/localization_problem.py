#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np

from typing import Optional


# CLASSES
class LocalizationProblem:
    def __init__(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, f: Optional[np.ndarray] = None,
                 c: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None, d: Optional[np.ndarray] = None,
                 s: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None) -> None:
        """
        Constructor
        :param x: Flags determining if the warehouses are going to be built, dim: Mx1
        :param y: Covered demand fractions, dim: MxN
        :param f: Transport costs from factory to warehouses, dim: Mx1
        :param c: Capacities of the shops, dim: Nx1
        :param b: Warehouses building costs, dim: Mx1
        :param d: Demands of the shops, dim: Nx1
        :param s: Transport costs from warehouses to shops, dim: MxN
        :param v: Sugar values established between warehouses and shops, dim: MxN
        """

        self.x = x
        self.Y = y
        self.f = f
        self.c = c
        self.b = b
        self.d = d
        self.S = s
        self.V = v

    def objective_function(self) -> int:
        """
        Method to calculate the objective function value with the current variables values
        :return: Objective function value with the current variables values
        """

        # M, N = self.Y.shape
        # income: int = 0
        # cost: int = 0
        #
        # for i in range(M):
        #     for j in range(N):
        #         taken_goods = self.d[j] * self.Y[i, j]
        #         income += taken_goods * self.V[i, j]
        #
        # for i in range(M):
        #     warehouse_cost: int = self.f[i] + self.b[i]
        #     warehouse_stores_transport_cost: int = 0
        #
        #     for j in range(N):
        #         warehouse_stores_transport_cost += np.ceil(self.Y[i, j]) * self.S[i, j]
        #
        #     warehouse_cost += warehouse_stores_transport_cost
        #     warehouse_cost *= self.x[i]
        #     cost += warehouse_cost

        income = ((self.V * self.Y) @ self.d).sum(axis=0)
        cost = np.dot(self.f + self.b + (np.ceil(self.Y) * self.S).sum(axis=0), self.x)

        return income - cost

    def check_decision_variables(self) -> bool:
        """
        Method to check the decision variables constraint
        :return: Flag informing if the constraint has been met
        """

        # return 0 <= self.Y.all() <= 1

        # return ((self.x == 0) | (self.x == 1)).all() and ((self.Y <= 1) & (0 <= self.Y)).all()

        return np.isin(self.x, [0, 1]).all() and ((self.Y <= 1) & (0 <= self.Y)).all()

    def check_shop_demand(self) -> bool:
        """
        Function to check the shop demand constraint
        :return: Flag informing if the constraint has been met
        """

        return (self.Y.sum(axis=0) == 1).all()

    def check_validity(self) -> bool:
        """
        Function to check the validity constraint
        :return: Flag informing if the constraint has been met
        """

        # return (np.all(self.Y, axis=0) <= self.x).all()

        return (self.Y.max(axis=1) <= self.x).all()

    def check_capacity(self) -> bool:
        """
        Function to check the capacity constraint
        :return: Flag informing if the constraint has been met
        """

        # for i in range(self.Y.shape[0]):
        #     check_val = np.dot(self.d, self.Y[i]) <= self.c[i] * self.x[i]
        #
        #     if not check_val:
        #         return False
        #
        # return True

        # return (np.dot(self.Y, self.d) <= self.c * self.x).all()

        return (np.dot(self.Y, self.d.T) <= self.c * self.x).all()


# FUNCTIONS
def initialize_matrix(num_of_potential_warehouses: int, num_of_shops: int) -> np.ndarray:
    """
    Function to initialize the matrix Y
    :param num_of_potential_warehouses: Number of potential warehouse localizations
    :param num_of_shops: Number of shops
    :return: Covered demand fractions matrix, dim: MxN
    """

    return np.random.uniform(size=(num_of_potential_warehouses, num_of_shops))


def initialize_vector(num_of_potential_warehouses: int) -> np.ndarray:
    """
    Function to initialize the vector x
    :param num_of_potential_warehouses: Number of potential warehouse localizations
    :return: Covered demand fractions matrix, dim: MxN
    """

    return np.random.randint(2, size=num_of_potential_warehouses)


# MAIN
def main():
    M = 10
    N = 50
    x = initialize_vector(M)
    y = initialize_matrix(M, N)
    lp = LocalizationProblem(x=x, y=y)
    print(lp.check_decision_variables())


    # mat2 = np.array([[0.7, 2.4],
    #                  [0.3, -1.4]])
    #
    # mat3 = np.array([[0.7, 2.4],
    #                  [0.3, -1.5]])

    # print(check_shop_demand(mat2))
    # print(check_shop_demand(mat3))

    # print(check_decision_variables(mat2))
    # print(check_decision_variables(mat3))

    # y = np.array([[0.7, 0.8],
    #               [0.3, 0.5]])
    # x1 = np.array([0, 1])
    # x2 = np.array([1, 1])
    #
    # print((y.max(axis=1) <= x1).all())


# RUN
if __name__ == "__main__":
    main()
