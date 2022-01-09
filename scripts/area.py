#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Optional
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# PROJECT MODULES
from data import ProblemSize


# CLASSES
class Area:
    """
    Class to represent the area
    """

    def __init__(self, length: float, problem_size: ProblemSize, tab: tk.Frame) -> None:
        """
        Constructor
        :param length: Area square side length
        :param problem_size: Problem size
        :param tab: Notebook tab
        """

        # Notebook tab
        self.tab = tab

        # Problem size
        self.problem_size = problem_size

        # Facilities coordinates
        self.warehouses = np.random.uniform(low=-length, high=length, size=(problem_size.M, 2))
        self.shops = np.random.uniform(low=-length, high=length, size=(problem_size.N, 2))

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
                    ax.plot([x0, 0], [y0, 0], c="black", label="_nolegend_")

                    for n, (x, y) in enumerate(self.shops):
                        ax.plot([x0, x], [y0, y], c="black", ls='--', label="_nolegend_")

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
