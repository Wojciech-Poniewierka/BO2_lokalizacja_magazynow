#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
from typing import Union


# ALIASES
Number = Union[int, float]

# GLOBAL CONSTANTS
TITLE: str = "WareLoc"
TEXT_COLOR: str = "#121212"
BOLD_FONT = ("Garamond", 16, "bold")
HIGHLIGHT = "black"
BORDER = 2

# DEFAULT VALUES
PROBLEM_SIZE = (10, 18)

TRANSPORT_COST_AMPLIFIER = 0.01             # Transport cost amplifier
BUILDING_COST_AMPLIFIER = 0.01              # Building cost amplifier
CAPACITY_MIN = 40000                        # Minimal capacity
CAPACITY_MAX = 60000                        # Maximal capacity
DEMAND_MIN = 100                            # Minimal demand
DEMAND_MAX = 7000                           # Maximal demand
COST_MIN = 10                               # Minimal cost
COST_MAX = 20                               # Maximal cost

POPULATION_SIZE = 50                        # Population size
N_GENERATIONS = 50                          # Number of generations
CROSSOVER_RATIO = 0.85                      # Crossover ratio
EQUALITY_PENALTY = 1000000                  # Equality constraint penalty coefficient
INEQUALITY_PENALTY = 1000000                # Inequality constraint penalty coefficient

START_SOLUTION = 20                         # Start solution method range

SELECTION_SORTING_GROUPING_STRATEGY = 6     # Sorting grouping strategy selection offset
SELECTION_TOURNAMENT = 4                    # Tournament selection participants
SELECTION_LINEAR_RANK = 1.5                 # Linear rank selection coefficient
SELECTION_NON_LINEAR_RANK = 0.5             # Non-linear rank selection coefficient

CROSSOVER_LINEAR = 0.5                      # Linear crossover coefficient
CROSSOVER_BLEND = 0.2                       # Blend crossover coefficient
CROSSOVER_SIMULATED_BINARY = 7              # Simulated binary crossover coefficient

MUTATION_POLYNOMIAL = 100                   # Polynomial mutation coefficient
