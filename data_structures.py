#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np


# FUNCTIONS
def F(d: np.ndarray, y: np.ndarray, v: np.ndarray, f: np.ndarray, b: np.ndarray, s: np.ndarray,
                       x: np.ndarray) -> int:
    """
    Objective function
    :param d: Demands of the shops, dim: Nx1
    :param y: Covered demand fractions, dim: MxN
    :param v: Sugar values established between warehouses and shops, dim: MxN
    :param f: Transport costs from factory to warehouses, dim: Mx1
    :param b: Warehouses building costs, dim: Mx1
    :param s: Transport costs from warehouses to shops, dim: MxN
    :param x: Flags determining if the warehouses are going to be built, dim: Mx1
    :return: Objective function value
    """

    # income: int = 0
    #
    # for i in range(M):
    #     for j in range(N):
    #         taken_goods = d[j] * y[i, j]
    #         income += taken_goods * v[i, j]

    # income = np.sum(v.T @ y @ d)
    income = ((v * y) @ d).sum(axis=0)

    # cost: int = 0
    #
    # for i in range(M):
    #     warehouse_cost: int = f[i] + b[i]
    #     warehouse_stores_transport_cost: int = 0
    #
    #     for j in range(N):
    #         warehouse_stores_transport_cost += np.ceil(y[i, j]) * s[i, j]
    #
    #     warehouse_cost += warehouse_stores_transport_cost
    #     warehouse_cost *= x[i]
    #     cost += warehouse_cost

    cost = np.dot(f + b + (np.ceil(y) * s).sum(axis=0), x)

    return income - cost


def initialize_matrix(num_of_shops: int, num_of_potential_warehouses: int) -> np.ndarray:
    """
    Function to initialize vector x
    :param num_of_shops: Number of shops
    :param num_of_potential_warehouses: Number of potential warehouse localizations
    :return: Vector x
    """

    matrix = np.random.randn(num_of_potential_warehouses, num_of_shops)

    return matrix


def check_decision_variables(matrix: np.ndarray):
    """
    Function to check the decision variables constraint
    :param matrix: Covered demand fractions, dim: MxN
    :return: Flag informing if the constraint has been met
    """

    # return 0 <= matrix.all() <= 1

    return ((matrix <= 1) & (0 <= matrix)).all()


def check_shop_demand(matrix: np.ndarray) -> bool:
    """
    Function to check the shop demand constraint
    :param matrix: Covered demand fractions, dim: MxN
    :return: Flag informing if the constraint has been met
    """

    return (matrix.sum(axis=0) == 1).all()


def check_validity(y: np.ndarray, x: np.array) -> bool:
    """
    Function to check the validity constraint
    :param y: Covered demand fractions, dim: MxN
    :param x: Flags determining if the warehouses are going to be built, dim: Mx1
    :return: Flag informing if the constraint has been met
    """

    # return (np.all(y, axis=0) <= x).all()

    return (y.max(axis=1) <= x).all()


def check_capacity(matrix: np.ndarray, demand: np.array, capacity: np.array, x: np.ndarray) -> bool:
    """
    Function to check the capacity constraint
    :param matrix: Covered demand fractions, dim: MxN
    :param demand: Demands of the shops, dim: Nx1
    :param capacity: Capacities of the shops, dim: Nx1
    :param x: Flags determining if the warehouses are going to be built, dim: Mx1
    :return: Flag informing if the constraint has been met
    """

    # for i in range(matrix.shape[0]):
    #     check_val = np.dot(demand, matrix[i]) <= capacity[i] * x[i]
    #
    #     if not check_val:
    #         return False
    #
    # return True

    # return (np.dot(matrix, demand) <= capacity * x).all()

    return (np.dot(matrix, demand.T) <= capacity * x).all()


# MAIN
def main():
    mat = initialize_matrix(10, 20)
    print(check_decision_variables(mat))


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
