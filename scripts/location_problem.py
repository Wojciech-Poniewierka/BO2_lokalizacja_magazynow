#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Tuple, List


# TYPE ALIASES
Locations = Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]


# CLASSES
class LocationProblem:
    def __init__(self, x: np.ndarray, y: np.ndarray, l: Locations, f: Optional[np.ndarray] = None,
                 c: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None, d: Optional[np.ndarray] = None,
                 s: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None) -> None:
        """
        Constructor
        :param x: Flags determining if the warehouses are going to be built, dim: Mx1
        :param y: Covered demand fractions, dim: MxN
        :param l: Locations of warehouses and shops as a tuple of lists of coordinates (x, y)
        :param f: Transport costs from factory to warehouses, dim: Mx1
        :param c: Capacities of the shops, dim: Nx1
        :param b: Warehouses building costs, dim: Mx1
        :param d: Demands of the shops, dim: Nx1
        :param s: Transport costs from warehouses to shops, dim: MxN
        :param v: Sugar values established between warehouses and shops, dim: MxN
        """

        self.x = x
        self.Y = y
        self.l = l
        self.f = f
        self.c = c
        self.b = b
        self.d = d
        self.S = s
        self.V = v

    def draw_graph(self, factory_vertex_color: str, warehouse_vertex_color: str, shop_vertex_color: str,
                   factory_warehouse_edge_color: str, warehouse_shop_edge_color: str) -> None:
        """
        Method to draw the graph representing the locations of factory, warehouses and shops
        :param factory_vertex_color: Factory vertex color
        :param warehouse_vertex_color: Warehouse vertex color
        :param shop_vertex_color: Shop vertex color
        :param factory_warehouse_edge_color: Edge between factory and warehouse color
        :param warehouse_shop_edge_color: Edge between warehouse and shop color
        """

        warehouses, shops = self.l

        plt.figure()
        plt.title("Area map"), plt.xlabel("x"), plt.ylabel("y")

        # Vertices
        plt.scatter(0, 0, c=factory_vertex_color, label="Factory")

        for n, (x, y) in enumerate(warehouses):
            plt.scatter(x, y, c=warehouse_vertex_color, label="Warehouse" if n == 0 else "_nolegend_")

        for n, (x, y) in enumerate(shops):
            plt.scatter(x, y, c=shop_vertex_color, label="Shop" if n == 0 else "_nolegend_")

        # # Edges
        # for x1, y1 in warehouses:
        #     plt.plot([0, x1], [0, y1], c=factory_warehouse_edge_color)
        #
        #     for x2, y2 in shops:
        #         plt.plot([x1, x2], [y1, y2], c=warehouse_shop_edge_color, ls="-.")

        plt.legend(bbox_to_anchor=(1, 1)), plt.grid()
        plt.show()

    def objective_function(self) -> int:
        """
        Method to calculate the objective function value with the current variables values
        :return: Objective function value with the current variables values
        """

        income = ((self.V * self.Y) @ self.d).sum(axis=0)
        cost = np.dot(self.f + self.b + (np.ceil(self.Y) * self.S).sum(axis=0), self.x)

        return income - cost

    def check_decision_variables(self) -> bool:
        """
        Method to check the decision variables constraint
        :return: Flag informing if the constraint has been met
        """

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

        return (self.Y.max(axis=1) <= self.x).all()

    def check_capacity(self) -> bool:
        """
        Function to check the capacity constraint
        :return: Flag informing if the constraint has been met
        """

        return (np.dot(self.Y, self.d.T) <= self.c * self.x).all()


# FUNCTIONS
def determine_locations(num_of_potential_warehouses: int, num_of_shops: int, area_size: float = 1000) -> Locations:
    """
    Function to determine the locations of factory, warehouses and shops
    :param num_of_potential_warehouses: Number of potential warehouse localizations
    :param num_of_shops: Number of shops
    :param area_size: Area size
    :return: Locations of factory, warehouses and shops as a tuple of lists of coordinates
    """

    warehouses_x = np.random.uniform(low=-area_size, high=area_size, size=num_of_potential_warehouses)
    warehouses_y = np.random.uniform(low=-area_size, high=area_size, size=num_of_potential_warehouses)
    warehouses_coordinates = [(x, y) for x, y in zip(warehouses_x, warehouses_y)]

    shops_x = np.random.uniform(low=-area_size, high=area_size, size=num_of_shops)
    shops_y = np.random.uniform(low=-area_size, high=area_size, size=num_of_shops)
    shops_coordinates = [(x, y) for x, y in zip(shops_x, shops_y)]

    return warehouses_coordinates, shops_coordinates


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
    # Number of locations
    M = 4
    N = 20

    # Area size
    A = 500

    # Vector and matrices
    x = initialize_vector(M)
    y = initialize_matrix(M, N)
    l = determine_locations(M, N, area_size=A)

    # Problem instance
    lp = LocationProblem(x, y, l)

    # Optimization algorithm
    lp.draw_graph("red", "green", "blue", "black", "yellow")
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
