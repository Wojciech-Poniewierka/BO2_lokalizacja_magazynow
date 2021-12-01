#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import Optional, Tuple, List

# PROJECT MODULES
from solution import Solution
from population import Population

# TYPE ALIASES
Location = Tuple[float, float]


# CLASSES
class Area:
    """
    Class to represent the area
    """

    def __init__(self, size: float, shape: Tuple[int, int]) -> None:
        """
        Constructor
        :param size: Area size, square side
        :param shape: Problem shape
        """

        M, N = shape

        warehouses_x = np.random.uniform(low=-size, high=size, size=M)
        warehouses_y = np.random.uniform(low=-size, high=size, size=M)
        self.warehouses_coordinates = [(x, y) for x, y in zip(warehouses_x, warehouses_y)]

        shops_x = np.random.uniform(low=-size, high=size, size=N)
        shops_y = np.random.uniform(low=-size, high=size, size=N)
        self.shops_coordinates = [(x, y) for x, y in zip(shops_x, shops_y)]

    def calculate_cost_matrices(self, transport_cost_amplifier: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to calculate the cost matrices
        :param transport_cost_amplifier: Transport cost amplifier
        :return: Tuple: (Factory to warehouses transport costs - dim: Mx1,
        warehouses to shops transport costs - dim: MxN)
        """

        f = np.array([calculate_transport_cost((0, 0), w, transport_cost_amplifier)
                      for w in self.warehouses_coordinates])
        S = np.array([[calculate_transport_cost(w, s, transport_cost_amplifier) for s in self.shops_coordinates]
                      for w in self.warehouses_coordinates])

        return f, S

    def draw_graph(self, ax: plt.Axes) -> None:
        """
        Method to draw the graph representing the locations of factory, warehouses and shops
        :param ax: Axis
        """

        # Description
        ax.set_title("Area map"), ax.set_xlabel("x"), ax.set_ylabel("y")

        # Vertices
        ax.scatter(0, 0, c="red", label="Factory")

        for n, (x, y) in enumerate(self.warehouses_coordinates):
            ax.scatter(x, y, c="green", label="Warehouse" if n == 0 else "_nolegend_")

        for n, (x, y) in enumerate(self.shops_coordinates):
            ax.scatter(x, y, c="blue", label="Shop" if n == 0 else "_nolegend_")

        # # Edges
        # for x1, y1 in self.warehouses_coordinates:
        #     plt.plot([0, x1], [0, y1], c="black")
        #
        #     for x2, y2 in self.shops_coordinates:
        #         plt.plot([x1, x2], [y1, y2], c="yellow", ls="-.")

        ax.legend(bbox_to_anchor=(1, 1)), ax.grid()
        plt.show()


class LocationProblem:
    """
    Class to represent the location problem
    """

    def __init__(self, shape: Tuple[int, int], f: np.ndarray, s: np.ndarray, parameters, c: Optional[np.ndarray] = None,
                 b: Optional[np.ndarray] = None, d: Optional[np.ndarray] = None,
                 v: Optional[np.ndarray] = None) -> None:
        """
        Constructor
        :param shape: Problem shape
        :param f: Factory to warehouses transport costs, dim: Mx1
        :param s: Warehouses to shops transport costs, dim: MxN
        :param parameters: List: [Mutation_ratio, noise, constraint_accuracy, population_size, min_fitness, max_generations,
                      crossover_ratio, transport_cost_amplifier, building_cost_amplifier, capacity_range, demand_range,
                      cost_range]
        :param c: Warehouses capacities, dim: Mx1
        :param b: Warehouses building costs, dim: Mx1
        :param d: Demands of the shops, dim: Nx1
        :param v: Sugar values established between warehouses and shops, dim: MxN
        """

        # Transport
        self.M, self.N = shape

        # Problem parameters
        self.f = f
        self.S = s

        # Algorithm parameters
        self.building_cost_amplifier = parameters[7]
        self.capacity_range = parameters[8]
        self.demand_range = parameters[9]
        self.cost_range = parameters[10]

        # Warehouses
        min_capacity, max_capacity = self.capacity_range
        self.c = np.random.uniform(low=min_capacity, high=max_capacity, size=self.M) if c is None else c
        self.b = np.array([self.building_cost_amplifier * capacity for capacity in self.c]) if b is None else b

        # Goods exchange
        min_demand, max_demand = self.demand_range
        self.d = np.random.uniform(low=min_demand, high=max_demand, size=self.N) if d is None else d
        min_cost, max_cost = self.cost_range
        self.V = np.random.uniform(low=min_cost, high=max_cost, size=(self.M, self.N)) if v is None else v

        # First population
        self.population = Population((self.M, self.N), self.f, self.S, self.c, self.b, self.d, self.V, parameters[:-4])

    def solve(self) -> Tuple[Tuple[float, Solution], List[Solution], List[float]]:
        """
        Method to solve the problem
        :return: Tuple: (Tuple: (Best solution fitness, Best solution), Best generation, Best fitnesses)
        """

        return self.population.genetic_algorithm()

    def generate_parameters(self) -> None:
        """
        Function to generate parameters
        """

        # Warehouses
        min_capacity, max_capacity = self.capacity_range
        c = np.random.uniform(low=min_capacity, high=max_capacity, size=self.M)
        b = np.array([self.building_cost_amplifier * capacity for capacity in c])

        # Goods exchange
        min_demand, max_demand = self.demand_range
        d = np.random.uniform(low=min_demand, high=max_demand, size=self.N)
        min_cost, max_cost = self.cost_range
        V = np.random.uniform(low=min_cost, high=max_cost, size=(self.M, self.N))

        dic = {"c": c, "b": b, "d": d, "V": V}

        for arr, arr_name in dic.items():
            save_to_csv(arr_name, arr)


# FUNCTIONS
def calculate_transport_cost(l1: Location, l2: Location, transport_cost_amplifier: float) -> float:
    """
    Method to calculate the transport cost between the locations
    :param l1: First location
    :param l2: Second location
    :param transport_cost_amplifier: Transport cost amplifier
    :return: Transport cost
    """

    x1, y1 = l1
    x2, y2 = l2

    return transport_cost_amplifier * np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def save_to_csv(arr: np.ndarray, arr_name: str) -> None:
    """
    Function to save the array to the .csv file
    :param arr: Array
    :param arr_name: Array name
    """

    df = pd.DataFrame(arr)
    df.to_csv(path_or_buf=f"{arr_name}.csv", index=False)


def read_parameters() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to read the parameters
    :return: Parameters: (Warehouses capacities - dim: Mx1, warehouses building costs - dim: Mx1,
    demands of the shops - dim: Nx1, sugar values established between warehouses and shops - dim: MxN)
    """

    dic = {"c": None, "b": None, "d": None, "V": None}

    for arr_name in dic:
        df = pd.read_csv(f"{arr_name}.csv")
        dic[arr_name] = df.to_numpy()

    return dic["c"].flatten(), dic["b"].flatten(), dic["d"].flatten(), dic["V"]
