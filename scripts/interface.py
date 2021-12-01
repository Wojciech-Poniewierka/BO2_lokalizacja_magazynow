#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import tkinter as tk
import tkinter.ttk as ttk
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Tuple

# PROJECT MODULES
from solver import Area, LocationProblem


# GLOBAL CONSTANTS
WIDTH: int = 900
HEIGHT: int = 500
WINDOW_SIZE: str = f"{WIDTH}x{HEIGHT}"
BACKGROUND_COLOR: str = "#3b414a"
TEXT_COLOR: str = "#121212"

ROOT = tk.Tk()


# CLASSES
class Matrix:
    """
    Class to represent the matrix to insert the data
    """

    def __init__(self, master: tk.Frame, shape: Tuple[int, int]):
        """
        Constructor
        :param master: Master frame
        :param shape: Problem shape
        """

        self.mat = []
        tk.Button(master, text=f"Confirm data", command=self.save).pack()

        self.frame = tk.Frame(master)
        self.frame.pack()

        self.M, self.N = shape
        self.array = None

        for i in range(self.M):
            row = []

            for j in range(self.N):
                cell = tk.Entry(self.frame, fg=TEXT_COLOR, bg="white", width=10, state="normal", disabledbackground='#000000')
                cell.grid(row=i, column=j)
                cell.bind("<Up>", lambda event: self.move_up())
                cell.bind("<Down>", lambda event: self.move_down())
                cell.bind("<Right>", lambda event: self.move_right())
                cell.bind("<Left>", lambda event: self.move_left())
                row.append(cell)

            self.mat.append(row)

    def get_cursor_location(self) -> Tuple[int, int]:
        """
        Method to get the coordinates of the current cell
        :return: Coordinates of the current cell
        """

        current_cell = self.frame.focus_get()
        
        for i in range(self.M):
            for j in range(self.N):
                if self.mat[i][j] is current_cell:
                    return i, j

    def move_up(self) -> None:
        """
        Method to move to the cell above
        """

        i, j = self.get_cursor_location()
        i = max(i - 1, 0)
        self.mat[i][j].focus()

    def move_down(self) -> None:
        """
        Method to move to the cell below
        """

        i, j = self.get_cursor_location()
        i = min(i + 1, self.M - 1)
        self.mat[i][j].focus()

    def move_right(self) -> None:
        """
        Method to move to the cell on the right
        """

        i, j = self.get_cursor_location()
        j = min(j + 1, self.N - 1)
        self.mat[i][j].focus()

    def move_left(self) -> None:
        """
        Method to move to the cell on the left
        """

        i, j = self.get_cursor_location()
        j = max(j - 1, 0)
        self.mat[i][j].focus()

    def save(self) -> None:
        """
        Method to save the inserted data
        """

        self.array = []

        for i in range(self.M):
            row = []
            for j in range(self.N):
                value = 0 if self.mat[i][j].get() == "" else float(self.mat[i][j].get())
                row.append(value)

            self.array.append(row)


class Application:
    """
    Class to represent the application
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        self.M = None
        self.N = None
        self.area = None
        self.chart = None

        self.notebook = ttk.Notebook(ROOT)
        self.notebook.pack()

        tab = tk.Frame(self.notebook)
        tab.grid_rowconfigure(0)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        self.notebook.add(tab, text="Main window")
        self.tabs = [tab]
        self.dim_button = tk.Button(tab, text="Choose dimensions", command=self.dim)
        self.dim_button.grid(row=0, column=0, columnspan=2)
        self.M_entry = tk.Entry(tab)
        self.M_entry.grid(row=1, column=0)
        self.N_entry = tk.Entry(tab)
        self.N_entry.grid(row=1, column=1)

        tk.Label(tab, text="Parameters").grid(row=2, column=0, columnspan=2)

        # Solution
        tk.Label(tab, text="Mutation ratio").grid(row=3, column=0)
        self.mutation_ratio_entry = tk.Entry(tab)
        self.mutation_ratio_entry.grid(row=3, column=1)
        self.mutation_ratio_entry.insert(0, 0.1)

        tk.Label(tab, text="Noise").grid(row=4, column=0)
        self.noise_entry = tk.Entry(tab)
        self.noise_entry.grid(row=4, column=1)
        self.noise_entry.insert(0, 0.0001)

        tk.Label(tab, text="Constraint accuracy").grid(row=5, column=0)
        self.constraint_accuracy_entry = tk.Entry(tab)
        self.constraint_accuracy_entry.grid(row=5, column=1)
        self.constraint_accuracy_entry.insert(0, 0.1)

        # Population
        tk.Label(tab, text="Population size").grid(row=6, column=0)
        self.population_size_entry = tk.Entry(tab)
        self.population_size_entry.grid(row=6, column=1)
        self.population_size_entry.insert(0, 30)

        tk.Label(tab, text="Min fitness").grid(row=7, column=0)
        self.min_fitness_entry = tk.Entry(tab)
        self.min_fitness_entry.grid(row=7, column=1)
        self.min_fitness_entry.insert(0, 0.01)

        tk.Label(tab, text="Max generations").grid(row=8, column=0)
        self.max_generations_entry = tk.Entry(tab)
        self.max_generations_entry.grid(row=8, column=1)
        self.max_generations_entry.insert(0, 15)

        tk.Label(tab, text="Crossover ratio").grid(row=9, column=0)
        self.crossover_ratio_entry = tk.Entry(tab)
        self.crossover_ratio_entry.grid(row=9, column=1)
        self.crossover_ratio_entry.insert(0, 0.8)

        # Other
        tk.Label(tab, text="Transport cost amplifier").grid(row=10, column=0)
        self.transport_cost_amplifier_entry = tk.Entry(tab)
        self.transport_cost_amplifier_entry.grid(row=10, column=1)
        self.transport_cost_amplifier_entry.insert(0, 0.5)

        tk.Label(tab, text="Building cost amplifier").grid(row=11, column=0)
        self.building_cost_amplifier_entry = tk.Entry(tab)
        self.building_cost_amplifier_entry.grid(row=11, column=1)
        self.building_cost_amplifier_entry.insert(0, 1)

        tk.Label(tab, text="Capacity range").grid(row=12, column=0)
        self.capacity_range_entry = tk.Entry(tab)
        self.capacity_range_entry.grid(row=12, column=1)
        self.capacity_range_entry.insert(0, "10000, 25000")

        tk.Label(tab, text="Demand range").grid(row=13, column=0)
        self.demand_range_entry = tk.Entry(tab)
        self.demand_range_entry.grid(row=13, column=1)
        self.demand_range_entry.insert(0, "100, 7000")

        tk.Label(tab, text="Cost range").grid(row=14, column=0)
        self.cost_range_entry = tk.Entry(tab)
        self.cost_range_entry.grid(row=14, column=1)
        self.cost_range_entry.insert(0, "5, 12")

        self.run_button = tk.Button(tab, text="Run", command=self.run, state="disabled")
        self.run_button.grid(row=15, column=0, columnspan=2)

        self.notebook.bind("<Tab>", lambda event: self.move())
        self.notebook.bind("<Shift-KeyPress-Tab>", lambda event: self.move(reverse=True))

        self.mat3 = None
        self.mat4 = None
        self.mat5 = None
        self.mat6 = None

    def get_cursor_location(self) -> int:
        """
        Method to get the current tab index
        :return: Current tab index
        """

        digit = self.notebook.select()[-1]

        if digit in "".join([str(n) for n in range(10)]):
            return int(digit) - 1

        else:
            return 0

    def move(self, reverse: bool = False) -> None:
        """
        Method to move to the adjacent tab
        :param reverse: Flag determining the direction of the move: True -> Move right, False -> Move left
        """

        i = self.get_cursor_location()
        idx = (i - 1) % len(self.tabs) if reverse else (i + 1) % len(self.tabs)
        self.notebook.select(self.tabs[idx])

    def dim(self) -> None:
        """
        Method to determine the problem shape
        """

        self.M = self.M if self.M_entry.get() == "" else int(self.M_entry.get())
        self.N = self.N if self.N_entry.get() == "" else int(self.N_entry.get())
        self.dim_button["text"] = "Change dimensions"
        self.run_button["state"] = "normal"
        self.update()

    def update(self) -> None:
        """
        Method to update the area and matrices tabs
        """

        for tab in self.tabs[1:]:
            tab.destroy()

        if self.chart is not None:
            self.chart.destroy()

        self.tabs = self.tabs[:1]

        tab2 = tk.Frame(self.notebook)
        self.notebook.add(tab2, text="Area")
        self.tabs.append(tab2)

        self.area = Area(500, (self.M, self.N))
        fig = plt.Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        self.chart = FigureCanvasTkAgg(fig, self.tabs[1]).get_tk_widget()
        self.chart.pack()
        self.area.draw_graph(ax)

        tab3 = tk.Frame(self.notebook)
        self.notebook.add(tab3, text="Warehouses capacities")
        self.tabs.append(tab3)

        tab4 = tk.Frame(self.notebook)
        self.notebook.add(tab4, text="Warehouses building costs")
        self.tabs.append(tab4)

        tab5 = tk.Frame(self.notebook)
        self.notebook.add(tab5, text="Shops demands")
        self.tabs.append(tab5)

        tab6 = tk.Frame(self.notebook)
        self.notebook.add(tab6, text="Costs established between warehouses and shops")
        self.tabs.append(tab6)

        self.mat3 = Matrix(tab3, (self.M, 1))
        self.mat4 = Matrix(tab4, (self.M, 1))
        self.mat5 = Matrix(tab5, (1, self.N))
        self.mat6 = Matrix(tab6, (self.M, self.N))

    def run(self) -> None:
        """
        Method to run the algorithm and display the results
        """

        # Solution
        mutation_ratio = float(self.mutation_ratio_entry.get())
        noise = float(self.noise_entry.get())
        constraint_accuracy = float(self.constraint_accuracy_entry.get())

        # Population
        population_size = int(self.population_size_entry.get())
        min_fitness = float(self.min_fitness_entry.get())
        max_generations = int(self.max_generations_entry.get())
        crossover_ratio = float(self.crossover_ratio_entry.get())

        # Other
        transport_cost_amplifier = float(self.transport_cost_amplifier_entry.get())
        building_cost_amplifier = float(self.building_cost_amplifier_entry.get())
        capacity_lst = self.capacity_range_entry.get().split(", ")
        capacity_range = (float(capacity_lst[0]), float(capacity_lst[1]))
        demand_lst = self.demand_range_entry.get().split(", ")
        demand_range = (float(demand_lst[0]), float(demand_lst[1]))
        cost_lst = self.cost_range_entry.get().split(", ")
        cost_range = (float(cost_lst[0]), float(cost_lst[1]))

        parameters = [mutation_ratio, noise, constraint_accuracy, population_size, min_fitness, max_generations,
                      crossover_ratio, building_cost_amplifier, capacity_range, demand_range, cost_range]

        f, S = self.area.calculate_cost_matrices(transport_cost_amplifier)
        lp = LocationProblem((self.M, self.N), f, S, parameters)
        best_solution, generation, fitness = lp.solve()
        fitness, solution = best_solution
        tab = self.tabs[0]
        tk.Label(tab, text=f"Fitness: {fitness}").grid(row=16, column=0, columnspan=2)
        tk.Label(tab, text=f"Solution\n{solution}").grid(row=17, column=0, columnspan=2)
        print(self.mat4.array)


# FUNCTIONS
def window_config(window: tk.Tk, title: str) -> tk.Tk:
    """
    Function to scale and name the window
    :param window: Main window
    :param title: Title
    :return: Updated window
    """

    x = int((ROOT.winfo_screenwidth() - WIDTH) / 2)
    y = int((ROOT.winfo_screenheight() - HEIGHT) / 2)
    window.geometry(f'{WIDTH}x{HEIGHT}+{x}+{y}')
    window.title(title)
    window.config(background=BACKGROUND_COLOR)

    return window


# MAIN
if __name__ == "__main__":
    ROOT = window_config(ROOT, "WareLoc")
    ROOT.bind("<Escape>", lambda event: ROOT.quit())
    app = Application()
    ROOT.mainloop()
