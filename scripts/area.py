#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import random as rd

from typing import Tuple, Optional
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# PROJECT MODULES
from config import TRANSPORT_COST_AMPLIFIER
from data import ProblemSize


# CLASSES
class Area:
    """
    Class to represent the area
    """

    def __init__(self, problem_size: ProblemSize, tab: tk.Frame) -> None:
        """
        Constructor
        :param problem_size: Problem size
        :param tab: Notebook tab
        """

        # Notebook tab
        self.tab = tab

        # Problem size
        self.problem_size = problem_size

        # Facilities coordinates
        self.warehouses = np.random.uniform(low=-500, high=500, size=(problem_size.M, 2))
        self.shops = np.random.uniform(low=-500, high=500, size=(problem_size.N, 2))

        # Distances
        self.factory_warehouses_distances = np.array([np.sqrt(x**2 + y**2)
                                                      for (x, y) in self.warehouses]).reshape(problem_size.M, 1)
        self.shops_warehouses_distances = np.array([[np.sqrt((w_x - s_x)**2 + (w_y - s_y)**2)
                                                     for (s_x, s_y) in self.shops]
                                                    for (w_x, w_y) in self.warehouses])

        # Draw a map
        self.draw()

    def draw(self, warehouses: Optional[np.ndarray] = None) -> None:
        """
        Method to draw a map
        :param warehouses: Warehouse
        """

        # Clean the notebook tab
        for widget in self.tab.winfo_children():
            widget.destroy()

        # Create a chart
        fig = plt.Figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        FigureCanvasTkAgg(fig, self.tab).get_tk_widget().pack()

        # Description
        ax.set_title("Area map"), ax.set_xlabel("x"), ax.set_ylabel("y")

        # Edges
        if warehouses is not None:
            for i in range(self.problem_size.M):
                if warehouses[i] > 0:
                    x0, y0 = self.warehouses[i]
                    ax.plot([x0, 0], [y0, 0], c="black", ls="--", label="_nolegend_")

        # Vertices
        ax.scatter(0, 0, s=60, c="red", label="Factory")

        for m, (x, y) in enumerate(self.warehouses):
            ax.scatter(x, y, c="green", s=40, label="Warehouse" if m == 0 else "_nolegend_")

        for n, (x, y) in enumerate(self.shops):
            ax.scatter(x, y, c="blue", s=20, label="Shop" if n == 0 else "_nolegend_")

        # Legend
        ax.legend(bbox_to_anchor=(1, 1)), ax.grid()
        plt.show()

    def calculate_cost_matrices(self, amplifier: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to calculate the cost matrices
        :param amplifier: Transport cost amplifier
        :return: Tuple: (Factory to warehouses transport costs - shape: Mx1,
        warehouses to shops transport costs - shape: MxN)
        """

        return amplifier * self.factory_warehouses_distances, amplifier * self.shops_warehouses_distances

    def calculate_coordinates(self, f: np.ndarray, S: np.ndarray) -> None:
        """
        Method to calculate the coordinates given the cost matrices
        :param f: Factory to warehouses transport costs - shape: Mx1
        :param S: Warehouses to shops transport costs - shape: MxN
        """

        self.factory_warehouses_distances = f / TRANSPORT_COST_AMPLIFIER
        self.shops_warehouses_distances = S / TRANSPORT_COST_AMPLIFIER
        self.warehouses = np.zeros((self.problem_size.M, 2))
        self.shops = np.zeros((self.problem_size.N, 2))

        for i in range(self.problem_size.M):
            d = self.factory_warehouses_distances[i]
            self.warehouses[i, 0] = np.random.uniform(low=-d, high=d)
            self.warehouses[i, 1] = np.sqrt(self.factory_warehouses_distances[i]**2 - self.warehouses[i, 0]**2)

        for j in range(self.problem_size.N):
            low = max([self.warehouses[i, 0] - self.shops_warehouses_distances[i, j]
                       for i in range(self.problem_size.M)])
            high = min([self.warehouses[i, 0] + self.shops_warehouses_distances[i, j]
                        for i in range(self.problem_size.M)])
            self.shops[j, 0] = np.random.uniform(low=low, high=high)
            self.shops[j, 1] = self.warehouses[0, 1] + rd.choice([-1, 1]) *\
                               np.sqrt(self.shops_warehouses_distances[0, j]**2 - (self.shops[j, 0] - self.warehouses[0, 0])**2)

        self.draw()
