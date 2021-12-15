#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import tkinter as tk
import tkinter.ttk as ttk
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Tuple, List

# PROJECT MODULES
from area import Area
from solver import LocationProblem


# GLOBAL CONSTANTS
WIDTH: int = 900
HEIGHT: int = 500
TITLE: str = "WareLoc"
BACKGROUND_COLOR: str = "#3b414a"
TEXT_COLOR: str = "#121212"

# ROOT
ROOT = tk.Tk()
x = int((ROOT.winfo_screenwidth() - WIDTH) / 2)
y = int((ROOT.winfo_screenheight() - HEIGHT) / 2)
ROOT.geometry(f"{WIDTH}x{HEIGHT}+{x}+{y}")
ROOT.title(TITLE)
ROOT.config(background=BACKGROUND_COLOR)
ROOT.bind("<Escape>", lambda event: ROOT.quit())


# CLASSES
class Matrix:
    """
    Class to represent the matrix to insert the data
    """

    def __init__(self, master: tk.Frame, shape: Tuple[int, int]):
        """
        Constructor
        :param master: Master frame
        :param shape: Matrix shape
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
                cell = tk.Entry(self.frame, fg=TEXT_COLOR, bg="white", width=10, state="normal",
                                disabledbackground='#000000')
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

        print(self.array)


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
        self.mats: List[Matrix] = []

        self.notebook = ttk.Notebook(ROOT)
        self.notebook.pack()

        tab = tk.Frame(self.notebook)
        tab.grid_rowconfigure(0)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)

        self.notebook.add(tab, text="Main window")
        self.tabs = [tab]

        self.dim_button = tk.Button(tab, text="Choose shape", command=self.determine_shape)
        self.dim_button.grid(row=0, column=0, columnspan=2)

        self.M_entry = tk.Entry(tab)
        self.M_entry.grid(row=1, column=0)
        self.M_entry.insert(0, 3)

        self.N_entry = tk.Entry(tab)
        self.N_entry.grid(row=1, column=1)
        self.N_entry.insert(0, 5)

        # Algorithm parameters
        tk.Label(tab, text="Algorithm parameters").grid(row=2, column=0, columnspan=2)

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

        tk.Label(tab, text="Maximal fitness").grid(row=7, column=0)
        self.max_fitness_entry = tk.Entry(tab)
        self.max_fitness_entry.grid(row=7, column=1)
        self.max_fitness_entry.insert(0, 100000)

        tk.Label(tab, text="Maximal number of generations").grid(row=8, column=0)
        self.max_generations_entry = tk.Entry(tab)
        self.max_generations_entry.grid(row=8, column=1)
        self.max_generations_entry.insert(0, 15)

        tk.Label(tab, text="Crossover ratio").grid(row=9, column=0)
        self.crossover_ratio_entry = tk.Entry(tab)
        self.crossover_ratio_entry.grid(row=9, column=1)
        self.crossover_ratio_entry.insert(0, 0.8)

        # Problem parameters
        tk.Label(tab, text="Problem parameters").grid(row=10, column=0, columnspan=2)

        # Other
        tk.Label(tab, text="Transport cost amplifier").grid(row=11, column=0)
        self.transport_cost_amplifier_entry = tk.Entry(tab)
        self.transport_cost_amplifier_entry.grid(row=11, column=1)
        self.transport_cost_amplifier_entry.insert(0, 0.5)

        tk.Label(tab, text="Building cost amplifier").grid(row=12, column=0)
        self.building_cost_amplifier_entry = tk.Entry(tab)
        self.building_cost_amplifier_entry.grid(row=12, column=1)
        self.building_cost_amplifier_entry.insert(0, 1)

        tk.Label(tab, text="Capacity range").grid(row=13, column=0)
        self.capacity_range_entry = tk.Entry(tab)
        self.capacity_range_entry.grid(row=13, column=1)
        self.capacity_range_entry.insert(0, "40000, 60000")

        tk.Label(tab, text="Demand range").grid(row=14, column=0)
        self.demand_range_entry = tk.Entry(tab)
        self.demand_range_entry.grid(row=14, column=1)
        self.demand_range_entry.insert(0, "100, 7000")

        tk.Label(tab, text="Cost range").grid(row=15, column=0)
        self.cost_range_entry = tk.Entry(tab)
        self.cost_range_entry.grid(row=15, column=1)
        self.cost_range_entry.insert(0, "10, 20")

        # Run button
        self.run_button = tk.Button(tab, text="Run", command=self.run, state="disabled")
        self.run_button.grid(row=16, column=0, columnspan=2, pady=10)

    def determine_shape(self) -> None:
        """
        Method to determine the problem shape
        """

        self.M = self.M if self.M_entry.get() == "" else int(self.M_entry.get())
        self.N = self.N if self.N_entry.get() == "" else int(self.N_entry.get())
        self.dim_button["text"] = "Change shape"
        self.run_button["state"] = "normal"
        self.update()

    def update(self) -> None:
        """
        Method to update the area and matrices tabs
        """

        # Remove previous tabs
        for tab in self.tabs[1:]:
            tab.destroy()

        if self.chart is not None:
            self.chart.destroy()

        self.tabs = self.tabs[:1]

        # Area tab
        self.tabs.append(tk.Frame(self.notebook))
        self.notebook.add(self.tabs[1], text="Area")

        self.area = Area(500, (self.M, self.N))
        fig = plt.Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        self.chart = FigureCanvasTkAgg(fig, self.tabs[1]).get_tk_widget()
        self.chart.pack()
        self.area.draw_graph(ax)

        # Another tabs
        for text, shape in [("Warehouses capacities", (self.M, 1)), ("Warehouses building costs", (self.M, 1)),
                            ("Shops demands", (1, self.N)),
                            ("Costs established between warehouses and shops", (self.M, self.N))]:
            self.tabs.append(tk.Frame(self.notebook))
            self.notebook.add(self.tabs[-1], text=text)
            self.mats.append(Matrix(self.tabs[-1], shape))

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
        max_fitness = float(self.max_fitness_entry.get())
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

        algorithm_parameters = (mutation_ratio, noise, constraint_accuracy, population_size, max_fitness,
                                max_generations, crossover_ratio, transport_cost_amplifier, building_cost_amplifier,
                                capacity_range, demand_range, cost_range)

        f, S = self.area.calculate_cost_matrices(transport_cost_amplifier)
        lp = LocationProblem((self.M, self.N), (f, S), algorithm_parameters)
        best_solution = lp.solve()
        tk.Label(self.tabs[0], text=f"Solution\n{best_solution}").grid(row=16, column=0, columnspan=2)
        tk.Label(self.tabs[0], text=f"Fitness: {best_solution.fitness}").grid(row=17, column=0, columnspan=2)


# MAIN
if __name__ == "__main__":
    app = Application()
    ROOT.mainloop()
