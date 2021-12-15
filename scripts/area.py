#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple


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
        :return: Tuple: (Factory to warehouses transport costs - shape: Mx1,
        warehouses to shops transport costs - shape: MxN)
        """

        f = [calculate_transport_cost((0, 0), w, transport_cost_amplifier)
                      for w in self.warehouses_coordinates]
        S = [[calculate_transport_cost(w, s, transport_cost_amplifier) for s in self.shops_coordinates]
                      for w in self.warehouses_coordinates]

        return np.array(f), np.array(S)

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


# FUNCTIONS
def calculate_transport_cost(location1: Tuple[float, float], location2: Tuple[float, float],
                             transport_cost_amplifier: float) -> float:
    """
    Method to calculate the transport cost between the locations
    :param location1: First location
    :param location2: Second location
    :param transport_cost_amplifier: Transport cost amplifier
    :return: Transport cost
    """

    x1, y1 = location1
    x2, y2 = location2

    return transport_cost_amplifier * np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
