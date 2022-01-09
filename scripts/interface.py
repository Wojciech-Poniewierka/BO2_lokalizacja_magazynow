#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Optional, Union, Tuple
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# PROJECT MODULES
from data import ProblemSize, ProblemParameters, AlgorithmParameters
from area import Area
from matrix import Matrix
from solver import Solver

# ALIASES
Number = Union[int, float]


# GLOBAL CONSTANTS
DEFAULT_PROBLEM_SIZE = (10, 18)

DEFAULT_TRANSPORT_COST_AMPLIFIER = 0.01
DEFAULT_BUILDING_COST_AMPLIFIER = 0.01
DEFAULT_CAPACITY_MIN = 40000
DEFAULT_CAPACITY_MAX = 60000
DEFAULT_DEMAND_MIN = 100
DEFAULT_DEMAND_MAX = 7000
DEFAULT_COST_MIN = 10
DEFAULT_COST_MAX = 20

DEFAULT_MUTATION_RATIO = 0.1
DEFAULT_POPULATION_SIZE = 45
DEFAULT_N_GENERATIONS = 30
DEFAULT_CROSSOVER_RATIO = 0.9
DEFAULT_PENALTY_COEFFICIENT = 1000

DEFAULT_START_SOLUTION_METHOD_RANGE = 20
DEFAULT_SELECTION_SORTING_GROUPING_STRATEGY_OFFSET = 6
DEFAULT_SELECTION_TOURNAMENT_PARTICIPANTS = 4
DEFAULT_SELECTION_LINEAR_RANK_COEFFICIENT = 1.5
DEFAULT_SELECTION_NON_LINEAR_RANK_COEFFICIENT = 0.5
DEFAULT_CROSSOVER_LINEAR_COEFFICIENT = 0.5
DEFAULT_CROSSOVER_BLEND_COEFFICIENT = 0.5
DEFAULT_CROSSOVER_SIMULATED_BINARY_COEFFICIENT = 7
DEFAULT_MUTATION_NON_UNIFORM_COEFFICIENT = 30
DEFAULT_MUTATION_POLYNOMIAL_COEFFICIENT = 60

AREA_SIZE = 500
WIDTH: int = 900
HEIGHT: int = 700
TITLE: str = "WareLoc"
BACKGROUND_COLOR: str = "#3b414a"
TEXT_COLOR: str = "#121212"
BOLD_FONT = ("Garamond", 16, "bold")

# ROOT
ROOT = tk.Tk()
ROOT.geometry(f"{WIDTH}x{HEIGHT}+{(ROOT.winfo_screenwidth() - WIDTH) // 2}+{(ROOT.winfo_screenheight() - HEIGHT) // 2}")
ROOT.title(TITLE)
ROOT.config(background=BACKGROUND_COLOR)
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

        main_tab = tk.Frame(self.notebook)
        main_tab.grid_rowconfigure(0, weight=1), main_tab.grid_rowconfigure(1, weight=1)
        main_tab.grid_columnconfigure(0, weight=1), main_tab.grid_columnconfigure(1, weight=1)

        settings_frame = tk.Frame(main_tab)
        settings_frame.grid(row=0, column=0, sticky="NW")
        methods_frame = tk.Frame(main_tab)
        methods_frame.grid(row=0, column=1, sticky="NE")
        solution_frame = tk.Frame(main_tab)
        solution_frame.grid(row=1, column=0, sticky="NW")
        fitness_frame = tk.Frame(main_tab)
        fitness_frame.grid(row=1, column=1, sticky="NE")

        plot_tab = tk.Frame(self.notebook)
        generator_tab = tk.Frame(self.notebook)

        self.notebook.add(main_tab, text="Main")
        self.notebook.add(plot_tab, text="Plot")
        self.notebook.add(generator_tab, text="Problem parameters generator")
        self.tabs = [main_tab, plot_tab, generator_tab]

        #--------------------------------------------------------------------------------------------------------------#
        # Buttons
        self.dim_button = tk.Button(settings_frame, text="Choose problem size", command=self.determine_size)
        self.dim_button.grid(row=0, column=0, ipadx=10, ipady=10)

        self.run_button = tk.Button(settings_frame, text="Run algorithm", command=self.run, state="disabled")
        self.run_button.grid(row=0, column=1, ipadx=10, ipady=10)

        # Problem size
        tk.Label(settings_frame, text="Problem size", font=BOLD_FONT).grid(row=1, column=0, columnspan=2)
        self.size_entries: List[tk.Entry] = [tk.Entry(settings_frame), tk.Entry(settings_frame)]

        for i, dimension in enumerate(DEFAULT_PROBLEM_SIZE):
            self.size_entries[i].grid(row=2, column=i)
            self.size_entries[i].insert(0, dimension)

        # Algorithm parameters
        tk.Label(settings_frame, text="Algorithm parameters", font=BOLD_FONT).grid(row=3, column=0, columnspan=2)
        self.algorithm_parameters_entries: List[tk.Entry] = []
        self.mutation_ratio = DEFAULT_MUTATION_RATIO
        self.population_size = DEFAULT_POPULATION_SIZE
        self.n_generations = DEFAULT_N_GENERATIONS
        self.crossover_ratio = DEFAULT_CROSSOVER_RATIO
        self.penalty_coefficient = DEFAULT_PENALTY_COEFFICIENT
        algorithm_parameters_texts = ["Mutation ratio", "Population size", "Number of generations", "Crossover ratio",
                                      "Penalty coefficient"]
        algorithm_parameters_default_values = [self.mutation_ratio, self.population_size, self.n_generations,
                                               self.crossover_ratio, self.penalty_coefficient]

        for i, (text, default_value) in enumerate(zip(algorithm_parameters_texts, algorithm_parameters_default_values)):
            tk.Label(settings_frame, text=text).grid(row=i + 4, column=0)
            algorithm_parameters_entry = tk.Entry(settings_frame)
            algorithm_parameters_entry.grid(row=i + 4, column=1)
            algorithm_parameters_entry.insert(0, default_value)
            self.algorithm_parameters_entries.append(algorithm_parameters_entry)

        # Warning labels
        tk.Label(settings_frame, text="Warnings", font=BOLD_FONT).grid(row=9, column=0, columnspan=2)
        self.main_warning_label_text = tk.StringVar(value="")
        tk.Label(settings_frame, textvariable=self.main_warning_label_text,
                 justify=tk.LEFT).grid(row=10, column=0, columnspan=2)
        #--------------------------------------------------------------------------------------------------------------#
        # Radio buttons
        self.methods: List[tk.IntVar] = [tk.IntVar() for _ in range(4)]

        start_frame = tk.Frame(methods_frame)
        start_frame.grid(row=0, column=0, sticky="NW", pady=(0, 10))
        tk.Label(start_frame, text="Start solution", font=BOLD_FONT).grid(row=0, column=0, columnspan=2)
        self.start_method_entries: List[tk.Entry] = []

        tk.Radiobutton(start_frame, text="Random", variable=self.methods[0], value=0,
                       command=lambda r="start": self.switch_entries(r)).grid(row=1, column=0, sticky=tk.W)

        for i, method in enumerate(["Best", "Worst"]):
            radiobutton = tk.Radiobutton(start_frame, text=method, variable=self.methods[0], value=i + 1,
                                         command=lambda r="start", idx=i: self.switch_entries(r, idx))
            radiobutton.grid(row=i + 2, column=0, sticky=tk.W)
            self.start_method_entries.append(tk.Entry(start_frame))
            self.start_method_entries[i].grid(row=i + 2, column=1, sticky=tk.W)
            self.start_method_entries[i].insert(0, DEFAULT_START_SOLUTION_METHOD_RANGE)
            self.start_method_entries[i]["state"] = "disabled"

        selection_frame = tk.Frame(methods_frame)
        selection_frame.grid(row=0, column=1, sticky="NW", pady=(0, 10))
        tk.Label(selection_frame, text="Selection", font=BOLD_FONT).grid(row=0, column=0, columnspan=2)
        self.selection_method_entries: List[tk.Entry] = []

        tk.Radiobutton(selection_frame, text="Roulette wheel", variable=self.methods[1], value=0,
                       command=lambda r="selection": self.switch_entries(r)).grid(row=1, column=0, sticky=tk.W)

        for i, (method, default_value) in enumerate([("Sorting grouping strategy", DEFAULT_SELECTION_SORTING_GROUPING_STRATEGY_OFFSET),
                                                     ("Tournament", DEFAULT_SELECTION_TOURNAMENT_PARTICIPANTS),
                                                     ("Linear rank", DEFAULT_SELECTION_LINEAR_RANK_COEFFICIENT),
                                                     ("Non-linear rank", DEFAULT_SELECTION_NON_LINEAR_RANK_COEFFICIENT)]):
            radiobutton = tk.Radiobutton(selection_frame, text=method, variable=self.methods[1], value=i + 1,
                                         command=lambda r="selection", idx=i: self.switch_entries(r, idx))
            radiobutton.grid(row=i + 2, column=0, sticky=tk.W)
            self.selection_method_entries.append(tk.Entry(selection_frame))
            self.selection_method_entries[i].grid(row=i + 2, column=1, sticky=tk.W)
            self.selection_method_entries[i].insert(0, default_value)
            self.selection_method_entries[i]["state"] = "disabled"

        crossover_frame = tk.Frame(methods_frame)
        crossover_frame.grid(row=1, column=0, sticky="NW", pady=(0, 10))
        tk.Label(crossover_frame, text="Crossover", font=BOLD_FONT).grid(row=0, column=0, columnspan=2)
        self.crossover_method_entries: List[tk.Entry] = []

        for i, method in enumerate(["Uniform", "Point", "Other linear", "Another", "Other blend"]):
            tk.Radiobutton(crossover_frame, text=method, variable=self.methods[2], value=i,
                           command=lambda r="crossover": self.switch_entries(r)).grid(row=i + 1, column=0, sticky=tk.W)

        for i, (method, default_value) in enumerate([("Linear", DEFAULT_CROSSOVER_LINEAR_COEFFICIENT),
                                                     ("Blend", DEFAULT_CROSSOVER_BLEND_COEFFICIENT),
                                                     ("Simulated binary", DEFAULT_CROSSOVER_SIMULATED_BINARY_COEFFICIENT)]):
            radiobutton = tk.Radiobutton(crossover_frame, text=method, variable=self.methods[2], value=i + 5,
                                         command=lambda r="crossover", idx=i: self.switch_entries(r, idx))
            radiobutton.grid(row=i + 6, column=0, sticky=tk.W)
            self.crossover_method_entries.append(tk.Entry(crossover_frame))
            self.crossover_method_entries[i].grid(row=i + 6, column=1, sticky=tk.W)
            self.crossover_method_entries[i].insert(0, default_value)
            self.crossover_method_entries[i]["state"] = "disabled"

        mutation_frame = tk.Frame(methods_frame)
        mutation_frame.grid(row=1, column=1, sticky="NW", pady=(0, 10))
        tk.Label(mutation_frame, text="Mutation", font=BOLD_FONT).grid(row=0, column=0, columnspan=2)
        self.mutation_method_entries: List[tk.Entry] = []

        for i, method in enumerate(["Swap", "Borrow"]):
            tk.Radiobutton(mutation_frame, text=method, variable=self.methods[3], value=i,
                           command=lambda r="mutation": self.switch_entries(r)).grid(row=i + 1, column=0, sticky=tk.W)

        for i, (method, default_value) in enumerate([("Non-uniform", DEFAULT_MUTATION_NON_UNIFORM_COEFFICIENT),
                                                     ("Polynomial", DEFAULT_MUTATION_POLYNOMIAL_COEFFICIENT)]):
            radiobutton = tk.Radiobutton(mutation_frame, text=method, variable=self.methods[3], value=i + 2,
                           command=lambda r="mutation", idx=i: self.switch_entries(r, idx))
            radiobutton.grid(row=i + 3, column=0, sticky=tk.W)
            self.mutation_method_entries.append(tk.Entry(mutation_frame))
            self.mutation_method_entries[i].grid(row=i + 3, column=1, sticky=tk.W)
            self.mutation_method_entries[i].insert(0, default_value)
            self.mutation_method_entries[i]["state"] = "disabled"
        #--------------------------------------------------------------------------------------------------------------#
        # Result label
        tk.Label(solution_frame, text="Solution", font=BOLD_FONT).grid(row=0, column=0)
        self.solution_label_text = tk.StringVar(value="")
        tk.Label(solution_frame, textvariable=self.solution_label_text, justify=tk.LEFT).grid(row=1, column=0)

        tk.Label(fitness_frame, text="Fitness", font=BOLD_FONT).grid(row=0, column=0)
        self.fitness_label_text = tk.StringVar(value="")
        tk.Label(fitness_frame, textvariable=self.fitness_label_text, justify=tk.LEFT).grid(row=1, column=0)
        #--------------------------------------------------------------------------------------------------------------#
        # Problem parameters
        generator_frame = tk.Frame(generator_tab)
        generator_frame.pack()
        problem_parameters_texts = ["Transport cost amplifier", "Minimal capacity", "Maximal capacity",
                                    "Building cost amplifier", "Minimal demand", "Maximal demand", "Minimal cost",
                                    "Maximal cost"]
        problem_parameters_default_values = [DEFAULT_TRANSPORT_COST_AMPLIFIER, DEFAULT_CAPACITY_MIN,
                                             DEFAULT_CAPACITY_MAX, DEFAULT_BUILDING_COST_AMPLIFIER, DEFAULT_DEMAND_MIN,
                                             DEFAULT_DEMAND_MAX, DEFAULT_COST_MIN, DEFAULT_COST_MAX]
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

        tk.Button(generator_frame, text="Load problem parameters 1", command=self.load).grid(row=8, column=0)
        tk.Button(generator_frame, text="Save problem parameters", command=self.save).grid(row=9, column=0)

        self.generator_warning_label_text = tk.StringVar(value="")
        tk.Label(generator_frame, textvariable=self.generator_warning_label_text).grid(row=10, column=0, columnspan=3)

        self.building_cost_amplifier = None
        self.is_loaded = False

    def switch_entries(self, radiobutton_type: str, *idx: int) -> None:
        """
        Method to switch on/off the entries
        :param radiobutton_type: Radiobutton type
        :param idx: Index of the entry to switch on
        """

        d = {"start": self.start_method_entries, "selection": self.selection_method_entries,
             "crossover": self.crossover_method_entries, "mutation": self.mutation_method_entries}

        for i, entry in enumerate(d[radiobutton_type]):
            entry["state"] = "normal" if i in idx else "disabled"

    def generate(self, idx: int) -> None:
        """
        Method to generate the problem parameters matrices
        :param idx: Index of row in which the button is located
        """

        self.generator_warning_label_text.set("")

        if idx == 0:
            transport_cost_amplifier = self.get(self.problem_parameters_entries[0],
                                                replacement=DEFAULT_TRANSPORT_COST_AMPLIFIER,
                                                warning="Transport cost amplifier should be a positive float",
                                                tab="generator")

            if transport_cost_amplifier is None:
                return None

            f, S = self.area.calculate_cost_matrices(transport_cost_amplifier)
            self.mats[0].set_array(f)
            self.mats[1].set_array(S)

        elif idx == 1:
            capacity_min = self.get(self.problem_parameters_entries[1], replacement=DEFAULT_CAPACITY_MIN,
                                    warning="Minimal capacity should be a non-negative float", tab="generator")
            capacity_max = self.get(self.problem_parameters_entries[2], replacement=DEFAULT_CAPACITY_MAX,
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
                                                    replacement=DEFAULT_BUILDING_COST_AMPLIFIER,
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
            demand_min = self.get(self.problem_parameters_entries[4], replacement=DEFAULT_DEMAND_MIN,
                                  warning="Minimal demand should be a non-negative float", tab="generator")
            demand_max = self.get(self.problem_parameters_entries[5], replacement=DEFAULT_DEMAND_MAX,
                                  warning="Maximal demand should be a positive float", tab="generator")

            if demand_min is None or demand_max is None:
                return None

            d = np.random.uniform(low=demand_min, high=demand_max, size=(1, self.problem_size.N))
            self.mats[4].set_array(d)

        elif idx == 6:
            cost_min = self.get(self.problem_parameters_entries[6], replacement=DEFAULT_COST_MIN,
                                warning="Minimal cost should be a non-negative float", tab="generator")
            cost_max = self.get(self.problem_parameters_entries[7], replacement=DEFAULT_COST_MAX,
                                warning="Maximal cost should be a positive float", tab="generator")

            if cost_min is None or cost_max is None:
                return None

            V = np.random.uniform(low=cost_min, high=cost_max, size=(self.problem_size.M, self.problem_size.N))
            self.mats[5].set_array(V)

    def load(self) -> None:
        """
        Method to load problem parameters from files
        """

        S = pd.read_csv(f"S_1.csv").to_numpy()
        M, N = S.shape
        self.problem_size = ProblemSize(M, N)
        self.dim_button["text"] = "Change problem size"
        self.run_button["state"] = "normal"
        self.mats: List[Matrix] = []

        self.is_loaded = True

        # Remove previous tabs
        idx = 3

        for tab in self.tabs[idx:]:
            tab.destroy()

        self.tabs = self.tabs[:idx]

        # Area tab
        self.tabs.append(tk.Frame(self.notebook))
        self.notebook.add(self.tabs[idx], text="Area")

        self.area = Area(AREA_SIZE, self.problem_size, self.tabs[idx])

        # Another tabs
        sizes = [(self.problem_size.M, 1), (self.problem_size.M, self.problem_size.N), (self.problem_size.M, 1),
                 (self.problem_size.M, 1), (1, self.problem_size.N), (self.problem_size.M, self.problem_size.N)]
        symbols = ["f", "S", "c", "b", "d", "V"]
        titles = ["Factory to warehouses transport costs", "Warehouses to shops transport costs",
                  "Warehouses capacities", "Warehouses building costs", "Shops demands",
                  "Costs established between warehouses and shops"]

        for size, symbol, title in zip(sizes, symbols, titles):
            self.tabs.append(tk.Frame(self.notebook))
            self.notebook.add(self.tabs[-1], text=symbol)
            M, N = size
            self.mats.append(Matrix(self.tabs[-1], ProblemSize(M, N), symbol, title))
            array = pd.read_csv(f"{symbol}_1.csv").to_numpy()
            self.mats[-1].set_array(array)

    def save(self) -> None:
        """
        Method to save the problem parameters into files
        """

        symbols = ["f", "S", "c", "b", "d", "V"]

        for i, symbol in enumerate(symbols):
            array = self.mats[i].array
            pd.DataFrame(array).to_csv(path_or_buf=f"{symbol}_1.csv", index=False)

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

        label_text = self.main_warning_label_text if tab == "main" else self.generator_warning_label_text
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

        M_blank_replacement = DEFAULT_PROBLEM_SIZE[0] if self.problem_size is None else self.problem_size.M
        N_blank_replacement = DEFAULT_PROBLEM_SIZE[1] if self.problem_size is None else self.problem_size.N

        M = self.get(self.size_entries[0], is_int=True, replacement=M_blank_replacement,
                     warning="Number of rows should be a positive integer")
        N = self.get(self.size_entries[1], is_int=True, replacement=N_blank_replacement,
                     warning="Number of columns should be a positive integer")

        if M is None or N is None:
            return None

        if self.problem_size is not None and M == self.problem_size.M and N == self.problem_size.N:
            return None

        self.problem_size = ProblemSize(M, N)
        self.dim_button["text"] = "Change problem size"
        self.run_button["state"] = "normal"
        self.mats: List[Matrix] = []
        self.update()

    def update(self) -> None:
        """
        Method to update the area and matrices tabs
        """

        # Remove previous tabs
        idx = 3

        for tab in self.tabs[idx:]:
            tab.destroy()

        self.tabs = self.tabs[:idx]

        # Area tab
        self.tabs.append(tk.Frame(self.notebook))
        self.notebook.add(self.tabs[idx], text="Area")

        self.area = Area(AREA_SIZE, self.problem_size, self.tabs[idx])

        # Another tabs
        sizes = [(self.problem_size.M, 1), (self.problem_size.M, self.problem_size.N), (self.problem_size.M, 1),
                 (self.problem_size.M, 1), (1, self.problem_size.N), (self.problem_size.M, self.problem_size.N)]
        symbols = ["f", "S", "c", "b", "d", "V"]
        titles = ["Factory to warehouses transport costs", "Warehouses to shops transport costs",
                 "Warehouses capacities", "Warehouses building costs", "Shops demands",
                 "Costs established between warehouses and shops"]

        for size, symbol, title in zip(sizes, symbols, titles):
            self.tabs.append(tk.Frame(self.notebook))
            self.notebook.add(self.tabs[-1], text=symbol)
            M, N = size
            self.mats.append(Matrix(self.tabs[-1], ProblemSize(M, N), symbol, title))

    def run(self) -> None:
        """
        Method to run the algorithm and display the results
        """

        # Clear the labels
        self.main_warning_label_text.set("")
        self.solution_label_text.set("")
        self.fitness_label_text.set("")

        # Problem parameters
        f = self.mats[0].array
        S = self.mats[1].array
        c = self.mats[2].array
        b = self.mats[3].array
        d = self.mats[4].array
        V = self.mats[5].array

        if any([a is None for a in [f, S, c, b, d, V]]):
            self.main_warning_label_text.set("Not all problem parameters have been determined")

            return None

        problem_parameters = ProblemParameters(f, S, c, b, d, V)

        # Algorithm parameters
        mutation_ratio = self.get(self.algorithm_parameters_entries[0], replacement=self.mutation_ratio,
                                  warning="Mutation ratio should be a float from interval [0, 1]", value_range=(0, 1))
        population_size = self.get(self.algorithm_parameters_entries[1], is_int=True, replacement=self.population_size,
                                   warning="Population size should be a positive integer")
        n_generations = self.get(self.algorithm_parameters_entries[2], is_int=True, replacement=self.n_generations,
                                 warning="Number of generations should be a positive integer")
        crossover_ratio = self.get(self.algorithm_parameters_entries[3], replacement=self.crossover_ratio,
                                   warning="Crossover ratio should be a float from interval [0, 1]", value_range=(0, 1))
        penalty_coefficient = self.get(self.algorithm_parameters_entries[4], replacement=self.penalty_coefficient,
                                       warning="Penalty coefficient should be a non-negative float")
        algorithm_parameters = [mutation_ratio, population_size, n_generations, crossover_ratio, penalty_coefficient]

        if any([p is None for p in algorithm_parameters]):
            return None

        methods: List[int] = [int(self.methods[i].get()) for i in range(4)]
        methods_values: List[Optional[Number]] = [-1 for _ in range(4)]

        warnings = ["Best", "Start"]

        for i in range(2):
            if methods[0] == i + 1:
                methods_values[0] = self.get(self.start_method_entries[i], is_int=True,
                                             replacement=DEFAULT_START_SOLUTION_METHOD_RANGE,
                                             warning=f"{warnings[i]} start solution range should be a positive integer")
                break

        replacements = [DEFAULT_SELECTION_SORTING_GROUPING_STRATEGY_OFFSET, DEFAULT_SELECTION_TOURNAMENT_PARTICIPANTS,
                        DEFAULT_SELECTION_LINEAR_RANK_COEFFICIENT, DEFAULT_SELECTION_NON_LINEAR_RANK_COEFFICIENT]
        warnings = ["Sorting grouping strategy selection offset should be a non-negative integer",
                    "Tournament selection participants number should be a positive integer",
                    "Linear rank selection coefficient should be a float from interval (1, 2)",
                    "Non-linear rank selection coefficient should be a float from interval (0, 1)"]
        value_range = [None, None, (1, 2), (0, 1)]

        for i in range(4):
            if methods[1] == i + 1:
                methods_values[1] = self.get(self.selection_method_entries[i], is_int=i < 2,
                                             replacement=replacements[i], warning=warnings[i],
                                             value_range=value_range[i])
                break

        replacements = [DEFAULT_CROSSOVER_LINEAR_COEFFICIENT, DEFAULT_CROSSOVER_BLEND_COEFFICIENT,
                        DEFAULT_CROSSOVER_SIMULATED_BINARY_COEFFICIENT]
        warnings = ["Linear", "Blend", "Simulated binary"]

        for i in range(3):
            if methods[2] == i + 5:
                methods_values[2] = self.get(self.crossover_method_entries[i], replacement=replacements[i],
                                             warning=f"{warnings[i]} crossover coefficient should be a positive float")
                break

        replacements = [DEFAULT_MUTATION_NON_UNIFORM_COEFFICIENT, DEFAULT_MUTATION_POLYNOMIAL_COEFFICIENT]
        warnings = ["Non-uniform", "Polynomial"]

        for i in range(2):
            if methods[3] == i + 2:
                methods_values[3] = self.get(self.mutation_method_entries[i], replacement=replacements[i],
                                             warning=f"{warnings[i]} mutation coefficient should be a positive float")
                break

        if any([v is None for v in methods_values]):
            return None

        algorithm_parameters = AlgorithmParameters(mutation_ratio, population_size, n_generations, crossover_ratio,
                                                   penalty_coefficient, methods, methods_values)

        # Solver
        solver = Solver(self.problem_size, problem_parameters, algorithm_parameters)
        history = solver.genetic_algorithm()
        best_solution = sorted(history, key=lambda sol: sol.fitness)[-1]
        fitnesses = [solution.fitness for solution in history]
        penalties = [solution.penalty for solution in history]

        # Result
        self.solution_label_text.set(best_solution)
        self.fitness_label_text.set(best_solution.fitness)

        # Best solutions plot
        plot_tab = self.tabs[1]

        for widget in plot_tab.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(10, 10), dpi=100)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        FigureCanvasTkAgg(fig, plot_tab).get_tk_widget().pack()

        factor = 0.1 if len(np.unique(penalties)) == 1 else 0
        step = 100 if len(np.unique(penalties)) == 1 and penalties[0] == 0 else 0

        ax1.set_title("Best solutions' objective function and penalty plots")
        ax1.set_ylabel("Objective function value")
        ax2.set_xlabel("Number of generation"), ax2.set_ylabel("Penalty value")
        ax1.plot(*range(1, self.n_generations + 1), fitnesses)
        ax1.axis([1, len(fitnesses), (1 - factor) * min(fitnesses) - step, (1 + factor) * max(fitnesses) + step])
        ax1.grid()

        ax2.plot(*range(1, self.n_generations + 1), penalties)
        ax2.axis([1, len(penalties), (1 - factor) * min(penalties) - step, (1 + factor) * max(penalties) + step])
        ax2.grid()
        plt.show()

        # Show connections
        self.area.draw(np.array([1 if best_solution.X[i, :].sum() > 0 else 0 for i in range(self.problem_size.M)]))


# MAIN
if __name__ == "__main__":
    app = Application()
    ROOT.mainloop()
