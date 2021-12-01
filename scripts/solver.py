#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import Optional, Tuple, List

# PROJECT MODULES
from config import Location, Locations, M, N, TRANSPORT_COST_AMPLIFIER, BUILDING_COST_AMPLIFIER, CAPACITY_RANGE,\
    DEMAND_RANGE, COST_RANGE
from solution import Solution
from population import Population


# CLASSES
class Area:
    """
    Class to represent the area
    """

    def __init__(self, size: float) -> None:
        """
        Constructor
        :param size: Area size, square side
        """

        warehouses_x = np.random.uniform(low=-size, high=size, size=M)
        warehouses_y = np.random.uniform(low=-size, high=size, size=M)
        self.warehouses_coordinates = [(x, y) for x, y in zip(warehouses_x, warehouses_y)]

        shops_x = np.random.uniform(low=-size, high=size, size=N)
        shops_y = np.random.uniform(low=-size, high=size, size=N)
        self.shops_coordinates = [(x, y) for x, y in zip(shops_x, shops_y)]

    def calculate_cost_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to calculate the cost matrices
        :return: Cost matrices (f, S)
        """

        f = np.array([calculate_transport_cost((0, 0), w) for w in self.warehouses_coordinates])
        S = np.array([[calculate_transport_cost(w, s) for s in self.shops_coordinates] for w in self.warehouses_coordinates])

        return f, S


class LocationProblem:
    """
    Class to represent the location problem
    """

    def __init__(self, c: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None,
                 d: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None) -> None:
        """
        Constructor
        :param c: Warehouses capacities, dim: Mx1
        :param b: Warehouses building costs, dim: Mx1
        :param d: Demands of the shops, dim: Nx1
        :param v: Sugar values established between warehouses and shops, dim: MxN
        """

        # Area
        self.area = Area(500)

        # Transport
        self.f, self.S = self.area.calculate_cost_matrices()

        # Warehouses
        min_capacity, max_capacity = CAPACITY_RANGE
        self.c = np.random.uniform(low=min_capacity, high=max_capacity, size=M) if c is None else c
        self.b = np.array([BUILDING_COST_AMPLIFIER * capacity for capacity in self.c]) if b is None else b

        # Goods exchange
        min_demand, max_demand = DEMAND_RANGE
        self.d = np.random.uniform(low=min_demand, high=max_demand, size=N) if d is None else d
        min_cost, max_cost = COST_RANGE
        self.V = np.random.uniform(low=min_cost, high=max_cost, size=(M, N)) if v is None else v

        self.population = Population(M, N, self.f, self.S, self.c, self.b, self.d, self.V)

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

        plt.figure()
        plt.title("Area map"), plt.xlabel("x"), plt.ylabel("y")

        # Vertices
        plt.scatter(0, 0, c=factory_vertex_color, label="Factory")

        for n, (x, y) in enumerate(self.area.warehouses_coordinates):
            plt.scatter(x, y, c=warehouse_vertex_color, label="Warehouse" if n == 0 else "_nolegend_")

        for n, (x, y) in enumerate(self.area.shops_coordinates):
            plt.scatter(x, y, c=shop_vertex_color, label="Shop" if n == 0 else "_nolegend_")

        # # Edges
        # for x1, y1 in self.area.warehouses_coordinates:
        #     plt.plot([0, x1], [0, y1], c=factory_warehouse_edge_color)
        #
        #     for x2, y2 in self.area.shops_coordinates:
        #         plt.plot([x1, x2], [y1, y2], c=warehouse_shop_edge_color, ls="-.")

        plt.legend(bbox_to_anchor=(1, 1)), plt.grid()
        plt.show()

    def solve(self) -> Tuple[Tuple[float, Solution], List[Solution], List[float]]:
        """
        Method to solve the problem
        :return: Tuple: (Tuple: (Best solution fitness, Best solution), Best generation, Best fitnesses)
        """

        return self.population.genetic_algorithm()


# FUNCTIONS
def calculate_transport_cost(l1: Location, l2: Location) -> float:
    """
    Method to calculate the transport cost between the locations
    :param l1: First location
    :param l2: Second location
    :return: Transport cost
    """

    x1, y1 = l1
    x2, y2 = l2

    return TRANSPORT_COST_AMPLIFIER * np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def save_to_csv(arr: np.ndarray, arr_name: str) -> None:
    """
    Function to save the array to the .csv file
    :param arr: Array
    :param arr_name: Array name
    """

    df = pd.DataFrame(arr)
    df.to_csv(path_or_buf=f"{arr_name}.csv", index=False)


def generate_parameters(m: int, n: int) -> None:
    """
    Function to generate parameters
    :param m:
    :param n:
    """

    # Warehouses
    min_capacity, max_capacity = CAPACITY_RANGE
    c = np.random.uniform(low=min_capacity, high=max_capacity, size=m)
    b = np.array([BUILDING_COST_AMPLIFIER * capacity for capacity in c])

    # Goods exchange
    min_demand, max_demand = DEMAND_RANGE
    d = np.random.uniform(low=min_demand, high=max_demand, size=n)
    min_cost, max_cost = COST_RANGE
    V = np.random.uniform(low=min_cost, high=max_cost, size=(m, n))

    dic = {"c": c, "b": b, "d": d, "V": V}

    for arr, arr_name in dic.items():
        save_to_csv(arr_name, arr)


def read_parameters() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to read the parameters
    :return: Parameters (c, d, b, V)
    """

    dic = {"c": None, "b": None, "d": None, "V": None}

    for arr_name in dic:
        df = pd.read_csv(f"{arr_name}.csv")
        dic[arr_name] = df.to_numpy()

    return dic["c"].flatten(), dic["b"].flatten(), dic["d"].flatten(), dic["V"]


# MAIN
def main():
    # Read parameters
    # generate_parameters(M, N)
    # c, b, d, V = read_parameters()

    # Problem instance
    lp = LocationProblem()
    lp.draw_graph("red", "green", "blue", "black", "yellow")

    # Solve the problem
    best_solution, generation, fitness = lp.solve()
    f, X = best_solution
    print(f)
    print(X)
    # print(generation)
    # print(fitness)


# RUN
if __name__ == "__main__":
    main()
