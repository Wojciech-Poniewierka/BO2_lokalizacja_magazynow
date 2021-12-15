#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np
import pandas as pd

from typing import Tuple, Optional

# PROJECT MODULES
from solution import Solution
from population import Population


# CLASSES
class LocationProblem:
    """
    Class to represent the location problem
    """

    def __init__(self, shape: Tuple[int, int], problem_parameters: Tuple[np.ndarray, np.ndarray],
                 algorithm_parameters: Tuple[float, float, float, int, float, int, float, float, float,
                                             Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 c: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None, d: Optional[np.ndarray] = None,
                 v: Optional[np.ndarray] = None) -> None:
        """
        Constructor
        :param problem_parameters: Tuple: (Factory to warehouses transport costs, shape: Mx1,
        Warehouses to shops transport costs, shape: MxN)
        :param algorithm_parameters: Tuple: (Mutation_ratio, Noise, Constraint_accuracy, Population_size, Min_fitness,
        Max_generations, Crossover_ratio, Transport_cost_amplifier, Building_cost_amplifier, Capacity_range,
        Demand_range, Cost_range)
        :param c: Warehouses capacities, shape: Mx1
        :param b: Warehouses building costs, shape: Mx1
        :param d: Shops demands, shape: Nx1
        :param v: Sugar values established between warehouses and shops, shape: MxN
        """

        # Transport
        self.M, self.N = shape

        # Problem parameters
        self.f, self.S = problem_parameters

        # Algorithm parameters
        self.building_cost_amplifier = algorithm_parameters[8]
        self.capacity_range = algorithm_parameters[9]
        self.demand_range = algorithm_parameters[10]
        self.cost_range = algorithm_parameters[11]

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
        self.population = Population((self.M, self.N), (self.f, self.S, self.c, self.b, self.d, self.V),
                                     algorithm_parameters)

    def solve(self) -> Solution:
        """
        Method to solve the problem
        :return: Best solution
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
    :return: Parameters: (Warehouses capacities - determine_shape: Mx1, warehouses building costs - determine_shape: Mx1,
    demands of the shops - determine_shape: Nx1, sugar values established between warehouses and shops - determine_shape: MxN)
    """

    dic = {"c": None, "b": None, "d": None, "V": None}

    for arr_name in dic:
        df = pd.read_csv(f"{arr_name}.csv")
        dic[arr_name] = df.to_numpy()

    return dic["c"].flatten(), dic["b"].flatten(), dic["d"].flatten(), dic["V"]
