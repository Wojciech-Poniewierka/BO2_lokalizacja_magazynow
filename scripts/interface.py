#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import os
import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# PROJECT MODULES
from config import *
from data import ProblemSize, ProblemParameters, AlgorithmParameters
from area import Area
from matrix import Matrix
from solver import Solver
from solution import Solution


# ROOT
ROOT = tk.Tk()
ROOT.geometry(f"{ROOT.winfo_screenwidth()}x{ROOT.winfo_screenheight()}")
ROOT.title(TITLE)
ROOT.bind("<Escape>", lambda event: ROOT.quit())


# CLASSES
class Application:
    """
    Class to represent the application
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        # Basic parameters
        self.problem_size: Optional[ProblemSize] = None
        self.area: Optional[Area] = None
        self.mats: List[Matrix] = []

        # Notebook
        self.notebook = ttk.Notebook(ROOT)
        self.notebook.pack()

        main_tab = tk.Frame(self.notebook, highlightbackground=HIGHLIGHT, highlightthickness=BORDER)
        main_tab.grid_rowconfigure(0, weight=1), main_tab.grid_rowconfigure(1, weight=1)
        main_tab.grid_columnconfigure(0, weight=1), main_tab.grid_columnconfigure(1, weight=1)

        frames: List[tk.Frame] = [tk.Frame(main_tab) for _ in range(3)]

        for i, frame in enumerate(frames):
            frame.grid(row=i % 2, column=i // 2, rowspan=i // 2 + 1, sticky="NW")

        plot_tab = tk.Frame(self.notebook, highlightbackground=HIGHLIGHT, highlightthickness=BORDER)
        generator_tab = tk.Frame(self.notebook, highlightbackground=HIGHLIGHT, highlightthickness=BORDER)

        self.notebook.add(main_tab, text="Main")
        self.notebook.add(plot_tab, text="Plot")
        self.notebook.add(generator_tab, text="Generator")
        self.tabs = [main_tab, plot_tab, generator_tab]
        #--------------------------------------------------------------------------------------------------------------#
        # Problem size
        self.dim_button = tk.Button(frames[0], text="Set size", command=self.determine_size)
        self.dim_button.grid(row=0, column=0, columnspan=2, ipadx=10, ipady=10)
        self.size_entries: List[tk.Entry] = [tk.Entry(frames[0], width=8), tk.Entry(frames[0], width=8)]

        for i, dimension in enumerate(PROBLEM_SIZE):
            self.size_entries[i].grid(row=1, column=i, ipadx=5, ipady=5)
            self.size_entries[i].insert(0, dimension)

        # Run algorithm
        self.solver = None

        self.run_button = tk.Button(frames[0], text="Run", command=self.solve, state="disabled")
        self.run_button.grid(row=0, column=2, ipadx=10, ipady=10)

        self.run_entry = tk.Entry(frames[0], width=3)
        self.run_entry.grid(row=0, column=4)
        self.run_entry.insert(0, 1)

        progress = ttk.Progressbar(frames[0], orient=tk.HORIZONTAL, mode="determinate")
        progress.grid(row=1, column=3, columnspan=2)
        self.progress_with_frame: Tuple[ttk.Progressbar, tk.Frame] = (progress, frames[0])

        # Algorithm parameters
        tk.Label(frames[0], text="Parameters", font=BOLD_FONT).grid(row=2, column=0, columnspan=2)
        self.algorithm_parameters_entries: List[tk.Entry] = []
        algorithm_parameters_texts: List[str] = ["Population size", "Number of generations", "Crossover ratio",
                                                 "Mutation ratio", "Equality constraint penalty coefficient",
                                                 "Inequality constraint penalty coefficient"]
        algorithm_parameters_default_values: List[Number] = [POPULATION_SIZE, N_GENERATIONS, CROSSOVER_RATIO,
                                                             MUTATION_RATIO, EQUALITY_PENALTY, INEQUALITY_PENALTY]

        for i, (text, default_value) in enumerate(zip(algorithm_parameters_texts, algorithm_parameters_default_values)):
            tk.Label(frames[0], text=text).grid(row=i + 3, column=0, columnspan=2)
            algorithm_parameters_entry = tk.Entry(frames[0])
            algorithm_parameters_entry.grid(row=i + 3, column=2, columnspan=2)
            algorithm_parameters_entry.insert(0, default_value)
            self.algorithm_parameters_entries.append(algorithm_parameters_entry)

        # Warning labels
        tk.Label(frames[0], text="Warnings", font=BOLD_FONT).grid(row=9, column=0, columnspan=4)
        self.size_warning_label_text = tk.StringVar(value="")
        tk.Label(frames[0], textvariable=self.size_warning_label_text,
                 justify=tk.LEFT).grid(row=10, column=0, columnspan=4)
        self.main_warning_label_text = tk.StringVar(value="")
        tk.Label(frames[0], textvariable=self.main_warning_label_text,
                 justify=tk.LEFT).grid(row=11, column=0, columnspan=4)
        #--------------------------------------------------------------------------------------------------------------#
        # Radio buttons
        self.methods: List[tk.IntVar] = [tk.IntVar() for _ in range(5)]
        methods_frames: List[tk.Frame] = [tk.Frame(frames[1]) for _ in range(5)]

        for i, text in enumerate(["Start solution", "Selection", "Crossover", "Mutation", "Inheritance"]):
            methods_frames[i].grid(row=i // 2, column=i % 2, sticky="NW", pady=(0, 10))
            tk.Label(methods_frames[i], text=text, font=BOLD_FONT).grid(row=0, column=0, columnspan=2)

        methods_info: List[List[Tuple[str, Optional[Number]]]] = [[("Scaling", None), ("Filling", None)],
            [("Roulette wheel", None), ("Sorting grouping strategy", S_SORTING_GROUPING_STRATEGY),
             ("Tournament", S_TOURNAMENT), ("Linear rank", S_LINEAR_RANK), ("Non-linear rank", S_NON_LINEAR_RANK)],
            [("Uniform", None), ("Point", None), ("Linear", C_LINEAR), ("Blend", C_BLEND),
             ("Simulated binary", C_SIMULATED_BINARY)], [("Swap", None), ("Borrow", None),
            ("Non-uniform", M_NON_UNIFORM), ("Polynomial", M_POLYNOMIAL)]]
        self.methods_entries: List[List[tk.Entry]] = [[] for _ in range(4)]

        for i, method_info in enumerate(methods_info):
            for j, (text, value) in enumerate(method_info):
                b = tk.Radiobutton(methods_frames[i], text=text, variable=self.methods[i], value=j,
                    command=lambda r=i, idx=None if value is None else len(self.methods_entries[i]):
                    self.switch_entries(r, idx=idx))
                b.grid(row=j + 1, column=0, sticky=tk.W)

                if value is not None:
                    self.methods_entries[i].append(tk.Entry(methods_frames[i]))
                    self.methods_entries[i][-1].grid(row=j + 1, column=1, sticky=tk.W)
                    self.methods_entries[i][-1].insert(0, value)
                    self.methods_entries[i][-1]["state"] = "disabled"

        # Entry
        self.start_entry = tk.Entry(methods_frames[0])
        self.start_entry.grid(row=3, column=0, sticky=tk.W)
        self.start_entry.insert(0, S_PROBABILITY)
        tk.Label(methods_frames[0], text="Infeasible solution probability").grid(row=3, column=1, sticky=tk.W)

        # Checkbox
        tk.Checkbutton(methods_frames[4], text="Kill parents", variable=self.methods[4]).grid(row=1, column=0, sticky=tk.W)
        #--------------------------------------------------------------------------------------------------------------#
        # Result label
        tk.Label(frames[2], text="Start solution", font=BOLD_FONT).grid(row=0, column=0)
        self.start_solution_frame: tk.Frame = tk.Frame(frames[2])
        self.start_solution_frame.grid(row=1, column=0)

        tk.Label(frames[2], text="Start solution fitness", font=BOLD_FONT).grid(row=2, column=0)
        self.start_fitness_frame: tk.Frame = tk.Frame(frames[2])
        self.start_fitness_frame.grid(row=3, column=0)

        tk.Label(frames[2], text="Best solution", font=BOLD_FONT).grid(row=4, column=0)
        self.best_solution_frame: tk.Frame = tk.Frame(frames[2])
        self.best_solution_frame.grid(row=5, column=0)

        tk.Label(frames[2], text="Best solution fitness", font=BOLD_FONT).grid(row=6, column=0)
        self.best_fitness_frame: tk.Frame = tk.Frame(frames[2])
        self.best_fitness_frame.grid(row=7, column=0)

        tk.Label(frames[2], text="Capacity constraint", font=BOLD_FONT).grid(row=8, column=0)
        self.constraint_frame: tk.Frame = tk.Frame(frames[2])
        self.constraint_frame.grid(row=9, column=0)
        #--------------------------------------------------------------------------------------------------------------#
        # Problem parameters
        generator_frame = tk.Frame(generator_tab)
        generator_frame.pack()
        problem_parameters_texts: List[str] = ["Transport cost amplifier", "Minimal capacity", "Maximal capacity",
                                    "Building cost amplifier", "Minimal demand", "Maximal demand", "Minimal cost",
                                    "Maximal cost"]
        problem_parameters_default_values: List[str] = [TRANSPORT_COST_AMPLIFIER, CAPACITY_MIN, CAPACITY_MAX,
                                             BUILDING_COST_AMPLIFIER, DEMAND_MIN, DEMAND_MAX, COST_MIN, COST_MAX]
        buttons = [("Generate f, S", 0, 1), ("Generate c", 1, 2), ("Generate b", 3, 1), ("Generate d", 4, 2),
                   ("Generate V", 6, 2)]
        self.problem_parameters_entries: List[tk.Entry] = []

        for i, (text, default_value) in enumerate(zip(problem_parameters_texts, problem_parameters_default_values)):
            tk.Label(generator_frame, text=text).grid(row=i, column=0, sticky=tk.W)
            self.problem_parameters_entries.append(tk.Entry(generator_frame))
            self.problem_parameters_entries[i].grid(row=i, column=1, sticky=tk.W)
            self.problem_parameters_entries[i].insert(0, default_value)

        for text, i, size in buttons:
            tk.Button(generator_frame, text=text,
                      command=lambda idx=i: self.generate(idx)).grid(row=i, column=2, rowspan=size, sticky=tk.W)

        combobox_values: List[str] = []
        n_instances = len(next(os.walk("instances"), (None, None, []))[2]) // 6
        i: int = 1

        while len(combobox_values) < n_instances:
            try:
                M, N = pd.read_csv(f"instances/S_{i}.csv").to_numpy().shape
                combobox_values.append(f"{i}: {M}x{N}")

            except:
                pass

            i += 1

        tk.Label(generator_frame, text="Load instance").grid(row=8, column=0)
        self.loaded_instance = tk.StringVar()
        self.load_combobox = ttk.Combobox(generator_frame, width=27, values=combobox_values,
                                     textvariable=self.loaded_instance)
        self.load_combobox.grid(row=9, column=0)
        self.load_combobox.bind("<<ComboboxSelected>>", lambda event: self.load())

        if n_instances != 0:
            self.load_combobox.current(0)

        tk.Button(generator_frame, text="Save instance", command=self.save).grid(row=8, column=1, rowspan=2)
        tk.Label(generator_frame, text="Remove instance").grid(row=8, column=2)
        self.removed_instance = tk.StringVar()
        self.remove_combobox = ttk.Combobox(generator_frame, width=27, values=combobox_values,
                                     textvariable=self.removed_instance)
        self.remove_combobox.grid(row=9, column=2)
        self.remove_combobox.bind("<<ComboboxSelected>>", lambda event: self.remove())

        self.generator_warning_label_text = tk.StringVar(value="")
        tk.Label(generator_frame, textvariable=self.generator_warning_label_text).grid(row=10, column=0, columnspan=3)

        self.building_cost_amplifier: Optional[Number] = None

    def switch_entries(self, radiobutton_type: int, idx: Optional[int]) -> None:
        """
        Method to switch on/off the entries
        :param radiobutton_type: Radiobutton type number
        :param idx: Index of the entry to switch on
        """

        for i, entry in enumerate(self.methods_entries[radiobutton_type]):
            entry["state"] = "normal" if i == idx else "disabled"

    def generate(self, idx: int) -> None:
        """
        Method to generate the problem parameters matrices
        :param idx: Index of row in which the button is located
        """

        self.generator_warning_label_text.set("")

        if idx == 0:
            transport_cost_amplifier = self.get(self.problem_parameters_entries[0], tab="generator",
                                                warning="Transport cost amplifier: positive float")

            if transport_cost_amplifier is None:
                return None

            f, S = self.area.calculate_cost_matrices(transport_cost_amplifier)
            self.mats[0].set_array(f)
            self.mats[1].set_array(S)

        elif idx == 1:
            capacity_min = self.get(self.problem_parameters_entries[1], tab="generator",
                                    warning="Minimal capacity: non-negative float")
            capacity_max = self.get(self.problem_parameters_entries[2], tab="generator",
                                    warning="Maximal capacity: positive float")

            if capacity_min is None or capacity_max is None:
                return None

            c = np.random.uniform(low=capacity_min, high=capacity_max, size=(self.problem_size.M, 1))
            self.mats[2].set_array(c)

            if self.building_cost_amplifier is not None:
                b = self.building_cost_amplifier * c
                self.mats[3].set_array(b)

        elif idx == 3:
            self.building_cost_amplifier = self.get(self.problem_parameters_entries[3], tab="generator",
                                                    warning="Building cost amplifier: positive float")

            if self.building_cost_amplifier is None:
                return None

            if self.mats[2].array is None:
                self.generator_warning_label_text.set("Matrix c is missing")

                return None

            b = self.building_cost_amplifier * self.mats[2].array
            self.mats[3].set_array(b)

        elif idx == 4:
            demand_min = self.get(self.problem_parameters_entries[4], tab="generator",
                                  warning="Minimal demand: non-negative float")
            demand_max = self.get(self.problem_parameters_entries[5], tab="generator",
                                  warning="Maximal demand: positive float")

            if demand_min is None or demand_max is None:
                return None

            d = np.random.uniform(low=demand_min, high=demand_max, size=(1, self.problem_size.N))
            self.mats[4].set_array(d)

        elif idx == 6:
            cost_min = self.get(self.problem_parameters_entries[6], tab="generator",
                                warning="Minimal cost: non-negative float")
            cost_max = self.get(self.problem_parameters_entries[7], tab="generator",
                                warning="Maximal cos: positive float")

            if cost_min is None or cost_max is None:
                return None

            V = np.random.uniform(low=cost_min, high=cost_max, size=(self.problem_size.M, self.problem_size.N))
            self.mats[5].set_array(V)

    def load(self) -> None:
        """
        Method to load problem parameters from files
        """

        n_instance = int(self.loaded_instance.get()[0])
        S = pd.read_csv(f"instances/S_{n_instance}.csv").to_numpy()
        M, N = S.shape
        self.problem_size = ProblemSize(M, N)
        self.dim_button["text"] = "Resize"
        self.run_button["state"] = "normal"
        self.mats: List[Matrix] = []
        self.update(n_instance=n_instance)
        self.area.calculate_coordinates(self.mats[0].array, self.mats[1].array)

    def remove(self) -> None:
        """
        Method to remove the problem parameters files
        """

        n_instance = int(self.removed_instance.get()[0])

        for symbol in ["f", "S", "c", "b", "d", "V"]:
            os.remove(f"instances/{symbol}_{n_instance}.csv")

        combobox_values = list(self.load_combobox["values"])

        for value in combobox_values:
            if n_instance == int(value[0]):
                combobox_values.remove(value)
                break

        self.load_combobox["values"] = combobox_values
        self.remove_combobox["values"] = combobox_values

    def save(self) -> None:
        """
        Method to save the problem parameters into files
        """

        if self.problem_size is None:
            return None

        combobox_values = list(self.load_combobox["values"])
        n_instance: int = 1

        while True:
            for value in combobox_values:
                if n_instance == int(value[0]):
                    break

            else:
                break

            n_instance += 1

        for i, symbol in enumerate(["f", "S", "c", "b", "d", "V"]):
            array = self.mats[i].array
            pd.DataFrame(array).to_csv(path_or_buf=f"instances/{symbol}_{n_instance}.csv", index=False)

        M, N = self.mats[1].array.shape
        combobox_values.append(f"{n_instance}: {M}x{N}")
        combobox_values.sort(key=lambda v: int(v[0]))
        self.load_combobox["values"] = combobox_values
        self.remove_combobox["values"] = combobox_values

    def get(self, entry: tk.Entry, is_int: bool = False, warning: str = "", tab: str = "main",
            value_range: Optional[Tuple[Number, Number]] = None) -> Optional[Number]:
        """
        Method to convert the entry content to a number if it is possible, otherwise to inform that it is not possible
        :param entry: Entry
        :param is_int: Flag information if the entry value should be an integer
        :param warning: Warning information in case the entry content is not a number string representation
        :param tab: Notebook tab in which to set up the warning
        :param value_range: Desired value range
        :return: The entry content is a number string representation -> The entry content as a float or an integer,
        Otherwise -> None
        """

        d = {"main": self.main_warning_label_text, "generator": self.generator_warning_label_text,
             "size": self.size_warning_label_text}
        label_text = d[tab]

        value = entry.get()
        warning = f"{label_text.get()}\n{warning}" if label_text.get() != "" else warning

        if value == "" or value.count(".") > 1:
            label_text.set(warning)

            return None

        for char in value:
            if not char.isdigit() and (is_int or char != "."):
                label_text.set(warning)

                return None

        if value_range is not None:
            x, y = value_range

            if float(value) < x or float(value) > y:
                label_text.set(warning)

                return None

        return int(value) if is_int else float(value)

    def determine_size(self) -> None:
        """
        Method to determine the problem shape
        """

        self.size_warning_label_text.set("")

        M = self.get(self.size_entries[0], is_int=True, warning="Number of rows: positive integer", tab="size")
        N = self.get(self.size_entries[1], is_int=True, warning="Number of columns: positive integer", tab="size")

        if M is None or N is None:
            return None

        if self.problem_size is not None and M == self.problem_size.M and N == self.problem_size.N:
            return None

        self.problem_size = ProblemSize(M, N)
        self.dim_button["text"] = "Change problem size"
        self.run_button["state"] = "normal"
        self.mats: List[Matrix] = []
        self.update()

        self.building_cost_amplifier = None
        entries_values: List[str] = [entry.get() for entry in self.problem_parameters_entries]

        for entry in self.problem_parameters_entries:
            entry.delete(0, tk.END)
            entry.insert(0, "")

        for idx in [0, 1, 3, 4, 6]:
            self.generate(idx)

        for i, entry in enumerate(self.problem_parameters_entries):
            entry.delete(0, tk.END)
            entry.insert(0, entries_values[i])

    def update(self, n_instance: Optional[int] = None) -> None:
        """
        Method to update the area and matrices tabs
        :param n_instance: Number of instance to load
        """

        # Remove previous tabs
        for tab in self.tabs[3:]:
            tab.destroy()

        self.tabs = self.tabs[:3]

        # Area tab
        self.tabs.append(tk.Frame(self.notebook, highlightbackground=HIGHLIGHT, highlightthickness=BORDER))
        self.notebook.add(self.tabs[3], text="Area")
        self.area = Area(self.problem_size, self.tabs[3])

        # Another tabs
        size = (self.problem_size.M, self.problem_size.N)
        sizes: List[Tuple[int, int]] = [(size[0], 1), (size[0], size[1]), (size[0], 1), (size[0], 1), (1, size[1]),
                                        (size[0], size[1])]
        symbols: List[str] = ["f", "S", "c", "b", "d", "V"]
        titles: List[str] = ["Factory to warehouses transport costs", "Warehouses to shops transport costs",
                 "Warehouses capacities", "Warehouses building costs", "Shops demands",
                 "Costs established between warehouses and shops"]

        for (M, N), symbol, title in zip(sizes, symbols, titles):
            self.tabs.append(tk.Frame(self.notebook, highlightbackground=HIGHLIGHT, highlightthickness=BORDER))
            self.notebook.add(self.tabs[-1], text=symbol)
            self.mats.append(Matrix(self.tabs[-1], ProblemSize(M, N), symbol, title))

            if n_instance is not None:
                array = pd.read_csv(f"instances/{symbol}_{n_instance}.csv").to_numpy()
                self.mats[-1].set_array(array)

    def solve(self) -> None:
        """
        Method to solve the problem and display the results
        """

        # Clear the labels
        self.main_warning_label_text.set("")

        # Number of runs
        n_runs = self.get(self.run_entry, is_int=True, warning="Number of runs: positive integer")

        if n_runs is None:
            return None

        # Problem parameters
        f, S, c = self.mats[0].array, self.mats[1].array, self.mats[2].array
        b, d, V = self.mats[3].array, self.mats[4].array, self.mats[5].array

        # Algorithm parameters
        warnings: List[str] = ["Population size: positive integer", "Number of generations: positive integer",
                               "Crossover ratio: float from interval [0, 1]",
                               "Mutation ratio: float from interval [0, 1]",
                               "Equality constraint penalty coefficient: non-negative float",
                               "Inequality constraint penalty coefficient: non-negative float"]
        algorithm_parameters_lst: List[float] = [self.get(self.algorithm_parameters_entries[i], is_int=i < 2,
                                                          warning=warning, value_range=(0, 1) if i in (2, 3) else None)
                                                 for i, warning in enumerate(warnings)]

        methods: List[int] = [int(self.methods[i].get()) for i in range(5)]
        methods_values: List[Optional[Number]] = [-1 for _ in range(4)]

        start_method_value = self.get(self.start_entry, value_range=(0, 1),
                                      warning="Infeasible solution probability: float from interval [0, 1]")

        if methods[1] > 0:
            warnings: List[str] = ["Sorting grouping strategy selection offset: non-negative integer",
                                   "Tournament selection participants number: positive integer",
                                   "Linear rank selection coefficient: float from interval (1, 2)",
                                   "Non-linear rank selection coefficient: float from interval (0, 1)"]
            value_ranges: List[Optional[Tuple[int, int]]] = [None, None, (1, 2), (0, 1)]
            methods_values[1] = self.get(self.methods_entries[1][methods[1] - 1], is_int=methods[1] < 3,
                                         warning=warnings[methods[1] - 1], value_range=value_ranges[methods[1] - 1])

        if methods[2] > 1:
            warnings: List[str] = ["Linear", "Blend", "Simulated binary"]
            methods_values[2] = self.get(self.methods_entries[2][methods[2] - 2],
                                         warning=f"{warnings[methods[2] - 2]} crossover coefficient: positive float")

        if methods[3] > 1:
            warnings: List[str] = ["Non-uniform", "Polynomial"]
            methods_values[3] = self.get(self.methods_entries[3][methods[3] - 2],
                                         warning=f"{warnings[methods[3] - 2]} mutation coefficient: positive float")

        if any([elem is None for elem in algorithm_parameters_lst + methods_values]) or start_method_value is None:
            return None

        methods_values[0] = start_method_value
        problem_parameters = ProblemParameters(f, S, c, b, d, V)
        algorithm_parameters = AlgorithmParameters(methods, methods_values, *algorithm_parameters_lst)

        # Solver
        self.solver = Solver(self.problem_size, problem_parameters, algorithm_parameters, self.progress_with_frame)
        best_fitness: float = float("-inf")
        best_solutions: Optional[List[Solution]] = None

        for _ in range(n_runs):
            best_solutions_run, best_fitness_run = self.run()

            if best_fitness_run > best_fitness:
                best_fitness = best_fitness_run
                best_solutions = best_solutions_run

        best_feasible_solutions: List[Solution] = [solution.scale() for solution in best_solutions]
        best_fitnesses: List[float] = [solution.fitness for solution in best_solutions]
        # best_feasible_fitnesses_idx: List[int] = [i + 1 for i, solution in enumerate(best_feasible_solutions) if solution is not None]
        # best_feasible_fitnesses: List[float] = [solution.fitness for solution in best_feasible_solutions if solution is not None]
        best_feasible_fitnesses: List[float] = [solution.fitness for solution in best_feasible_solutions]
        penalties_equality: List[float] = [solution.penalty_equality for solution in best_solutions]
        penalties_inequality: List[float] = [solution.penalty_inequality for solution in best_solutions]

        start_solution: Solution = best_feasible_solutions[0]
        best_solution: Solution = sorted(best_feasible_solutions, key=lambda sol: sol.fitness, reverse=True)[0]

        # start_solution: Solution = best_feasible_solutions[0]
        # best_solution: Solution = sorted([solution for solution in best_feasible_solutions if solution is not None],
        #                                  key=lambda sol: sol.fitness, reverse=True)[0]

        # H = sorted(best_solutions, key=lambda sol: sol.fitness, reverse=True)[0]
        # print(H.X.sum(axis=0))
        #
        # for j in range(self.problem_size.N):
        #     col_sum = H.X[:, j].sum()
        #     H.X[:, j] /= col_sum
        #
        # print(H)
        # H.calculate()
        # print(H.fitness)

        # Results
        for frame, fitness_frame, result in [(self.start_solution_frame, self.start_fitness_frame, start_solution),
                                             (self.best_solution_frame, self.best_fitness_frame, best_solution)]:
            tk.Label(frame, text="Shops", font=("Garamond", 12, "bold")).grid(row=0, column=1, columnspan=self.problem_size.N)
            tk.Label(frame, text="Warehouses", font=("Garamond", 12, "bold")).grid(row=1, column=0, rowspan=self.problem_size.M)

            for i in range(self.problem_size.M + 1):
                for j in range(self.problem_size.N):
                    font = ("Garamond", 10, "bold") if i == self.problem_size.M else ("Garamond", 10, "normal")
                    cell = tk.Entry(frame, fg=TEXT_COLOR, bg="white", width=5, justify=tk.CENTER, font=font)
                    value = round(result.X[:, j].sum(), 2) if i == self.problem_size.M else round(result[i, j], 2)
                    cell.grid(row=i + 1, column=j + 1), cell.insert(0, value)
                    cell["state"] = "disabled"

            cell = tk.Entry(fitness_frame, fg=TEXT_COLOR, bg="white", width=25, justify=tk.CENTER, font=("Garamond", 14, "bold"))
            cell.grid(row=0, column=1), cell.insert(0, round(result.fitness, 2))
            cell["state"] = "disabled"

        capacity_constraint: np.ndarray = np.concatenate((best_solution.X @ best_solution.d.T, c), axis=1)

        for i in range(self.problem_size.M):
            for j in range(3):
                font = ("Garamond", 10, "bold") if j == 1 else ("Garamond", 10, "normal")
                cell = tk.Entry(self.constraint_frame, fg=TEXT_COLOR, bg="white", width=15, justify=tk.CENTER, font=font)
                value = round(capacity_constraint[i, 0], 2) \
                    if j == 0 else ("<=" if j == 1 else round(capacity_constraint[i, 1], 2))
                cell.grid(row=i, column=j), cell.insert(0, value)
                cell["state"] = "disabled"

        # Best solutions' plots
        plot_tab = self.tabs[1]

        for widget in plot_tab.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(10, 10), dpi=100)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        FigureCanvasTkAgg(fig, plot_tab).get_tk_widget().pack()

        factor = 0.1 if len(np.unique(best_fitnesses)) + len(np.unique(best_feasible_fitnesses)) <= 2 else 0
        step = 100 if len(np.unique(best_fitnesses)) + len(np.unique(best_feasible_fitnesses)) <= 2 and best_fitnesses[0] == 0 and best_feasible_fitnesses[0] == 0 else 0
        ax1.plot([i for i in range(len(best_fitnesses) + 1)], [0.0] + best_fitnesses, label="All")
        ax1.plot([i for i in range(len(best_fitnesses) + 1)], [0.0] + best_feasible_fitnesses, label="Feasible", color="red", ls="--")
        # ax1.scatter(best_feasible_fitnesses_idx, best_feasible_fitnesses, label="Feasible", color="red")
        ax1.axis([1, len(best_fitnesses), (1 - factor) * min(min(best_fitnesses), min(best_feasible_fitnesses)) - step,
                  (1 + factor) * max(max(best_fitnesses), max(best_feasible_fitnesses)) + step])
        ax1.grid()
        ax1.legend()

        factor = 0.1 if len(np.unique(penalties_equality)) + len(np.unique(penalties_inequality)) <= 2 else 0
        step = 100 if len(np.unique(penalties_equality)) + len(np.unique(penalties_inequality)) <= 2 and penalties_equality[0] == 0 and penalties_inequality[0] == 0 else 0
        ax2.plot([i for i in range(len(penalties_equality) + 1)], [0.0] + penalties_equality, label="Equality constraint")
        ax2.plot([i for i in range(len(penalties_inequality) + 1)], [0.0] + penalties_inequality, label="Inequality constraint")
        ax2.axis([1, len(penalties_equality), (1 - factor) * min(min(penalties_equality), min(penalties_inequality)) - step,
                  (1 + factor) * max(max(penalties_equality), max(penalties_inequality)) + step])
        ax2.grid()
        ax2.legend()

        ax1.set_title("Best solutions' objective function and penalty plots")
        ax1.set_ylabel("Objective function value")
        ax2.set_xlabel("Number of generation")
        ax2.set_ylabel("Penalty value")
        plt.show()

        # Show connections
        self.area.draw(warehouses=np.array([best_solution.X[i, :].sum() > 0 for i in range(self.problem_size.M)]))

    def run(self) -> Tuple[List[Solution], float]:
        """
        Method to run the algorithm
        :return: Best solutions, best solution's fitness
        """

        best_solutions = self.solver.genetic_algorithm()
        best_feasible_solutions: List[Solution] = [solution.scale() for solution in best_solutions]
        best_solution: Solution = sorted(best_feasible_solutions, key=lambda sol: sol.fitness, reverse=True)[0]

        return best_solutions, best_solution.fitness


# MAIN
if __name__ == "__main__":
    app = Application()
    ROOT.mainloop()
