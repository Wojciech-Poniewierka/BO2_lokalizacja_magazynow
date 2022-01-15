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
        self.notebook.add(generator_tab, text="Problem parameters generator")
        self.tabs = [main_tab, plot_tab, generator_tab]
        #--------------------------------------------------------------------------------------------------------------#
        # Buttons
        self.dim_button = tk.Button(frames[0], text="Choose problem size", command=self.determine_size)
        self.dim_button.grid(row=0, column=0, ipadx=10, ipady=10)

        self.run_button = tk.Button(frames[0], text="Run algorithm", command=self.run, state="disabled")
        self.run_button.grid(row=0, column=1, ipadx=10, ipady=10)

        progress = ttk.Progressbar(frames[0], orient=tk.HORIZONTAL, mode="determinate")
        progress.grid(row=0, column=2)
        self.progress_with_frame: Tuple[ttk.Progressbar, tk.Frame] = (progress, frames[0])

        # Problem size
        tk.Label(frames[0], text="Problem size", font=BOLD_FONT).grid(row=1, column=0, columnspan=2)
        self.size_entries: List[tk.Entry] = [tk.Entry(frames[0]), tk.Entry(frames[0])]

        for i, dimension in enumerate(PROBLEM_SIZE):
            self.size_entries[i].grid(row=2, column=i)
            self.size_entries[i].insert(0, dimension)

        # Algorithm parameters
        tk.Label(frames[0], text="Algorithm parameters", font=BOLD_FONT).grid(row=3, column=0, columnspan=2)
        self.algorithm_parameters_entries: List[tk.Entry] = []
        algorithm_parameters_texts: List[str] = ["Population size", "Number of generations", "Crossover ratio",
                                                 "Mutation ratio", "Equality constraint penalty coefficient",
                                                 "Inequality constraint penalty coefficient"]
        algorithm_parameters_default_values: List[Number] = [POPULATION_SIZE, N_GENERATIONS, CROSSOVER_RATIO,
                                                             MUTATION_RATIO, EQUALITY_PENALTY, INEQUALITY_PENALTY]

        for i, (text, default_value) in enumerate(zip(algorithm_parameters_texts, algorithm_parameters_default_values)):
            tk.Label(frames[0], text=text).grid(row=i + 4, column=0)
            algorithm_parameters_entry = tk.Entry(frames[0])
            algorithm_parameters_entry.grid(row=i + 4, column=1)
            algorithm_parameters_entry.insert(0, default_value)
            self.algorithm_parameters_entries.append(algorithm_parameters_entry)

        # Warning labels
        tk.Label(frames[0], text="Warnings", font=BOLD_FONT).grid(row=10, column=0, columnspan=2)
        self.size_warning_label_text = tk.StringVar(value="")
        tk.Label(frames[0], textvariable=self.size_warning_label_text,
                 justify=tk.LEFT).grid(row=11, column=0, columnspan=2)
        self.main_warning_label_text = tk.StringVar(value="")
        tk.Label(frames[0], textvariable=self.main_warning_label_text,
                 justify=tk.LEFT).grid(row=12, column=0, columnspan=2)
        #--------------------------------------------------------------------------------------------------------------#
        # Radio buttons
        self.methods: List[tk.IntVar] = [tk.IntVar() for _ in range(4)]
        methods_frames: List[tk.Frame] = [tk.Frame(frames[1]) for _ in range(4)]

        for i, text in enumerate(["Start solution", "Selection", "Crossover", "Mutation"]):
            methods_frames[i].grid(row=i // 2, column=i % 2, sticky="NW", pady=(0, 10))
            tk.Label(methods_frames[i], text=text, font=BOLD_FONT).grid(row=0, column=0, columnspan=2)

        methods_info: List[List[Tuple[str, Optional[Number]]]] = [[("Random", None), ("Best", START_SOLUTION),
            ("Worst", START_SOLUTION)], [("Roulette wheel", None),
            ("Sorting grouping strategy", SELECTION_SORTING_GROUPING_STRATEGY), ("Tournament", SELECTION_TOURNAMENT),
            ("Linear rank", SELECTION_LINEAR_RANK), ("Non-linear rank", SELECTION_NON_LINEAR_RANK)], [("Uniform", None),
            ("Point", None), ("Linear", CROSSOVER_LINEAR), ("Blend", CROSSOVER_BLEND),
            ("Simulated binary", CROSSOVER_SIMULATED_BINARY)], [("Swap", None), ("Borrow", None),
            ("Non-uniform", PROBLEM_SIZE[0] * PROBLEM_SIZE[1]), ("Polynomial", MUTATION_POLYNOMIAL)]]
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
            transport_cost_amplifier = self.get(self.problem_parameters_entries[0],
                                                replacement=TRANSPORT_COST_AMPLIFIER,
                                                warning="Transport cost amplifier should be a positive float",
                                                tab="generator")

            if transport_cost_amplifier is None:
                return None

            f, S = self.area.calculate_cost_matrices(transport_cost_amplifier)
            self.mats[0].set_array(f)
            self.mats[1].set_array(S)

        elif idx == 1:
            capacity_min = self.get(self.problem_parameters_entries[1], replacement=CAPACITY_MIN,
                                    warning="Minimal capacity should be a non-negative float", tab="generator")
            capacity_max = self.get(self.problem_parameters_entries[2], replacement=CAPACITY_MAX,
                                    warning="Maximal capacity should be a positive float", tab="generator")

            if capacity_min is None or capacity_max is None:
                return None

            c = np.random.uniform(low=capacity_min, high=capacity_max, size=(self.problem_size.M, 1))
            self.mats[2].set_array(c)

            if self.building_cost_amplifier is not None:
                b = self.building_cost_amplifier * c
                self.mats[3].set_array(b)

        elif idx == 3:
            self.building_cost_amplifier = self.get(self.problem_parameters_entries[3],
                                                    replacement=BUILDING_COST_AMPLIFIER,
                                                    warning="Building cost amplifier should be a positive float",
                                                    tab="generator")

            if self.building_cost_amplifier is None:
                return None

            if self.mats[2].array is None:
                self.generator_warning_label_text.set("Matrix c is missing")

                return None

            b = self.building_cost_amplifier * self.mats[2].array
            self.mats[3].set_array(b)

        elif idx == 4:
            demand_min = self.get(self.problem_parameters_entries[4], replacement=DEMAND_MIN,
                                  warning="Minimal demand should be a non-negative float", tab="generator")
            demand_max = self.get(self.problem_parameters_entries[5], replacement=DEMAND_MAX,
                                  warning="Maximal demand should be a positive float", tab="generator")

            if demand_min is None or demand_max is None:
                return None

            d = np.random.uniform(low=demand_min, high=demand_max, size=(1, self.problem_size.N))
            self.mats[4].set_array(d)

        elif idx == 6:
            cost_min = self.get(self.problem_parameters_entries[6], replacement=COST_MIN,
                                warning="Minimal cost should be a non-negative float", tab="generator")
            cost_max = self.get(self.problem_parameters_entries[7], replacement=COST_MAX,
                                warning="Maximal cost should be a positive float", tab="generator")

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
        self.dim_button["text"] = "Change problem size"
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

    def get(self, entry: tk.Entry, is_int: bool = False, replacement: Number = 0, warning: str = "", tab: str = "main",
            value_range: Optional[Tuple[Number, Number]] = None) -> Optional[Number]:
        """
        Method to convert the entry content to a number if it is possible, otherwise to inform that it is not possible
        :param entry: Entry
        :param is_int: Flag information if the entry value should be an integer
        :param replacement: Replacement value in case the entry content is an empty string
        :param warning: Warning information in case the entry content is neither an empty string, nor the actual number
        string representation
        :param tab: Notebook tab in which to set up the warning
        :param value_range: Desired value range
        :return: The entry content is an empty string -> The blank replacement float or integer value,
        The entry content is actually a number -> The entry content as a float or an integer,
        The entry content is neither an empty string, nor the actual number string representation -> None
        """

        if tab == "main":
            label_text = self.main_warning_label_text

        elif tab == "generator":
            label_text = self.generator_warning_label_text

        else:
            label_text = self.size_warning_label_text

        value = entry.get()
        warning = f"{label_text.get()}\n{warning}" if label_text.get() != "" else warning

        if value == "":
            return replacement

        if value.count(".") > 1:
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

        M_replacement = PROBLEM_SIZE[0] if self.problem_size is None else self.problem_size.M
        N_replacement = PROBLEM_SIZE[1] if self.problem_size is None else self.problem_size.N

        M = self.get(self.size_entries[0], is_int=True, replacement=M_replacement,
                     warning="Number of rows should be a positive integer", tab="size")
        N = self.get(self.size_entries[1], is_int=True, replacement=N_replacement,
                     warning="Number of columns should be a positive integer", tab="size")

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

    def run(self) -> None:
        """
        Method to run the algorithm and display the results
        """

        # Clear the labels
        self.main_warning_label_text.set("")

        # Problem parameters
        f, S, c = self.mats[0].array, self.mats[1].array, self.mats[2].array
        b, d, V = self.mats[3].array, self.mats[4].array, self.mats[5].array

        # Algorithm parameters
        default_values: List[Number] = [POPULATION_SIZE, N_GENERATIONS, CROSSOVER_RATIO, MUTATION_RATIO,
                                        EQUALITY_PENALTY, INEQUALITY_PENALTY]
        warnings: List[str] = ["Population size should be a positive integer",
                               "Number of generations should be a positive integer",
                               "Crossover ratio should be a float from interval [0, 1]",
                               "Mutation ratio should be a float from interval [0, 1]",
                               "Equality constraint penalty coefficient should be a non-negative float",
                               "Inequality constraint penalty coefficient should be a non-negative float"]
        algorithm_parameters_lst: List[float] = [self.get(self.algorithm_parameters_entries[i], is_int=i < 2,
                                    replacement=default_value, warning=warning,
                                    value_range=(0, 1) if i in (2, 3) else None)
                                    for i, (default_value, warning) in enumerate(zip(default_values, warnings))]

        methods: List[int] = [int(self.methods[i].get()) for i in range(4)]
        methods_values: List[Optional[Number]] = [-1 for _ in range(4)]

        if methods[0] > 0:
            warnings: List[str] = ["Best", "Start"]
            methods_values[0] = self.get(self.methods_entries[0][methods[0] - 1], is_int=True, replacement=START_SOLUTION,
                                         warning=f"{warnings[methods[0] - 1]} start solution range should be a positive integer")

        if methods[1] > 0:
            replacements: List[Number] = [SELECTION_SORTING_GROUPING_STRATEGY, SELECTION_TOURNAMENT, SELECTION_LINEAR_RANK,
                                          SELECTION_NON_LINEAR_RANK]
            warnings: List[str] = ["Sorting grouping strategy selection offset should be a non-negative integer",
                                   "Tournament selection participants number should be a positive integer",
                                   "Linear rank selection coefficient should be a float from interval (1, 2)",
                                   "Non-linear rank selection coefficient should be a float from interval (0, 1)"]
            value_ranges: List[Optional[Tuple[int, int]]] = [None, None, (1, 2), (0, 1)]
            methods_values[1] = self.get(self.methods_entries[1][methods[1] - 1], is_int=methods[1] < 3,
                                         replacement=replacements[methods[1] - 1], warning=warnings[methods[1] - 1],
                                         value_range=value_ranges[methods[1] - 1])

        if methods[2] > 1:
            replacements: List[Number] = [CROSSOVER_LINEAR, CROSSOVER_BLEND, CROSSOVER_SIMULATED_BINARY]
            warnings: List[str] = ["Linear", "Blend", "Simulated binary"]
            methods_values[2] = self.get(self.methods_entries[2][methods[2] - 2], replacement=replacements[methods[2] - 2],
                                         warning=f"{warnings[methods[2] - 2]} crossover coefficient should be a positive float")

        if methods[3] > 1:
            replacements: List[Number] = [PROBLEM_SIZE[0] * PROBLEM_SIZE[1], MUTATION_POLYNOMIAL]
            warnings: List[str] = ["Non-uniform", "Polynomial"]
            methods_values[3] = self.get(self.methods_entries[3][methods[3] - 2], replacement=replacements[methods[3] - 2],
                                         warning=f"{warnings[methods[3] - 2]} mutation coefficient should be a positive float")

        if any([elem is None for elem in algorithm_parameters_lst + methods_values]):
            return None

        problem_parameters = ProblemParameters(f, S, c, b, d, V)
        algorithm_parameters = AlgorithmParameters(methods, methods_values, *algorithm_parameters_lst)

        # Solver
        solver = Solver(self.problem_size, problem_parameters, algorithm_parameters, self.progress_with_frame)
        best_solutions, best_feasible_solutions = solver.genetic_algorithm()
        start_solution: Solution = best_feasible_solutions[0]
        best_solution: Solution = sorted([solution for solution in best_feasible_solutions if solution is not None], key=lambda sol: sol.fitness)[-1]
        best_fitnesses: List[float] = [solution.fitness for solution in best_solutions]
        best_feasible_fitnesses: List[float] = [solution.fitness if solution is not None else 0 for solution in best_feasible_solutions]
        penalties_equality: List[float] = [solution.penalty_equality for solution in best_solutions]
        penalties_inequality: List[float] = [solution.penalty_inequality for solution in best_solutions]

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
                value = round(capacity_constraint[i, 0], 2)\
                    if j == 0 else ("<=" if j == 1 else round(capacity_constraint[i, 1], 2))
                cell.grid(row=i, column=j), cell.insert(0, value)
                cell["state"] = "disabled"

        # Best solutions' plots
        plot_tab = self.tabs[1]

        for widget in plot_tab.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(10, 10), dpi=100)
        ax1 = fig.add_subplot(2, 2, (1, 2))
        ax2 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2, 2, 4)
        FigureCanvasTkAgg(fig, plot_tab).get_tk_widget().pack()

        factor = 0.1 if len(np.unique(best_fitnesses)) + len(np.unique(best_feasible_fitnesses)) == 2 else 0
        step = 100 if len(np.unique(best_fitnesses)) + len(np.unique(best_feasible_fitnesses)) == 2 and best_fitnesses[0] == 0 and best_feasible_fitnesses[0] == 0 else 0
        ax1.plot([i for i in range(len(best_fitnesses) + 1)], [0.0] + best_fitnesses, label="All")
        ax1.plot([i for i in range(len(best_feasible_fitnesses) + 1)], [0.0] + best_feasible_fitnesses, label="Feasible")
        ax1.axis([1, len(best_fitnesses), (1 - factor) * min(min(best_fitnesses), min(best_feasible_fitnesses)) - step, (1 + factor) * max(max(best_fitnesses), max(best_feasible_fitnesses)) + step])
        ax1.grid()
        ax1.set_title("Best solutions' objective function and penalty plots")
        ax1.set_xlabel("Number of generation"), ax1.set_ylabel("Objective function value")
        ax1.legend()

        for ax, values, text in zip((ax2, ax3), (penalties_equality, penalties_inequality), ("Equality", "Inequality")):
            factor = 0.1 if len(np.unique(values)) == 1 else 0
            step = 100 if len(np.unique(values)) == 1 and values[0] == 0 else 0
            ax.plot([i for i in range(len(values) + 1)], [0.0] + values)
            ax.axis([1, len(values), (1 - factor) * min(values) - step, (1 + factor) * max(values) + step])
            ax.grid()
            ax.set_xlabel("Number of generation"), ax.set_ylabel(f"{text} penalty value")

        plt.show()

        # Show connections
        self.area.draw(warehouses=np.array([best_solution.X[i, :].sum() > 0 for i in range(self.problem_size.M)]))


# MAIN
if __name__ == "__main__":
    app = Application()
    ROOT.mainloop()
