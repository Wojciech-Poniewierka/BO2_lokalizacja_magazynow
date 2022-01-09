#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import tkinter as tk
import numpy as np

from typing import Tuple, List, Optional

# PROJECT MODULES
from data import ProblemSize


# GLOBAL CONSTANTS
DEFAULT_CAPACITY = 50000
DEFAULT_DEMAND = 4000
DEFAULT_COST = 15

TEXT_COLOR: str = "#121212"


# CLASSES
class Matrix:
    """
    Class to represent the matrix to insert the data
    """

    def __init__(self, master: tk.Frame, problem_size: ProblemSize, symbol: str, title: str) -> None:
        """
        Constructor
        :param master: Master frame
        :param problem_size: Problem size
        :param symbol: Problem parameter char symbol
        :param title: Notebook tab title
        """

        if symbol in ("f", "S", "b"):
            self.blank_replacement = None
            self.state = "disabled"

        else:
            d = {"c": DEFAULT_CAPACITY, "d": DEFAULT_DEMAND, "V": DEFAULT_COST}
            self.blank_replacement = d[symbol]
            self.state = "normal"

        self.cells: List[List[tk.Entry]] = []
        tk.Label(master, text=title, font=("Garamond", 16, "bold")).pack()
        tk.Button(master, text=f"Confirm data", command=self.save).pack()

        self.insert_frame = tk.Frame(master)
        self.insert_frame.pack()

        self.display_frame = tk.Frame(master)
        self.display_frame.pack()

        warning_frame = tk.Frame(master)
        warning_frame.pack()

        self.M = problem_size.M
        self.N = problem_size.N

        self.array: Optional[np.ndarray] = None

        for i in range(self.M):
            row = []

            for j in range(self.N):
                cell = tk.Entry(self.insert_frame, fg=TEXT_COLOR, bg="white", width=8, state=self.state)
                cell.grid(row=i, column=j)
                cell.bind("<Up>", lambda event: self.move_up())
                cell.bind("<Down>", lambda event: self.move_down())
                cell.bind("<Right>", lambda event: self.move_right())
                cell.bind("<Left>", lambda event: self.move_left())
                row.append(cell)

            self.cells.append(row)

        # Warning label
        self.warning_label_text = tk.StringVar(value="")
        tk.Label(warning_frame, textvariable=self.warning_label_text).grid()

    def get_cursor_location(self) -> Tuple[int, int]:
        """
        Method to get the coordinates of the current cell
        :return: Coordinates of the current cell
        """

        current_cell = self.insert_frame.focus_get()

        for i in range(self.M):
            for j in range(self.N):
                if self.cells[i][j] is current_cell:
                    return i, j

    def move_up(self) -> None:
        """
        Method to move to the cell above
        """

        i, j = self.get_cursor_location()
        self.cells[max(i - 1, 0)][j].focus()

    def move_down(self) -> None:
        """
        Method to move to the cell below
        """

        i, j = self.get_cursor_location()
        self.cells[min(i + 1, self.M - 1)][j].focus()

    def move_right(self) -> None:
        """
        Method to move to the cell on the right
        """

        i, j = self.get_cursor_location()
        self.cells[i][min(j + 1, self.N - 1)].focus()

    def move_left(self) -> None:
        """
        Method to move to the cell on the left
        """

        i, j = self.get_cursor_location()
        self.cells[i][max(j - 1, 0)].focus()

    def display(self) -> None:
        """
        Method to display the matrix
        """

        self.array = np.around(self.array, decimals=2)

        for widget in self.display_frame.winfo_children():
            widget.destroy()

        mat = "\n".join([" ".join([str(round(elem, 2)).center(9) for elem in self.array[i, :]]) for i in range(self.M)])
        tk.Label(self.display_frame, text=f"\nMatrix:\n{mat}").grid()

        if self.state == "normal":
            for i, row in enumerate(self.array):
                for j, elem in enumerate(row):
                    self.cells[i][j].insert(0, elem)

        else:
            for i, row in enumerate(self.array):
                for j, elem in enumerate(row):
                    self.cells[i][j]["state"] = "normal"
                    self.cells[i][j].insert(0, elem)
                    self.cells[i][j]["state"] = "disabled"

    def set_array(self, array: np.ndarray) -> None:
        """
        Method to set the array values
        """

        self.array = array
        self.display()

    def save(self):
        """
        Method to save the inserted data
        """

        self.array = np.array([[self.get_cell_value(cell) for cell in row] for row in self.cells])
        self.display()

    def get_cell_value(self, cell: tk.Entry) -> float:
        """
        Method to get the cell value and change it in case it is incorrect
        :param cell: Cell entry
        :return: The cell value is not the actual number -> The blank replacement float value,
        The cell value is actually a number -> The cell value as a float
        """

        cell_value = cell.get()

        if cell_value != "" and cell_value.count(".") <= 1:
            for char in cell_value:
                if not char.isdigit() and char != ".":
                    break

            else:
                self.warning_label_text.set("")

                return float(cell_value)

        self.warning_label_text.set("Cell value should be a float")

        return self.blank_replacement
