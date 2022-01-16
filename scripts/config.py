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
N_GENERATIONS = 500                         # Number of generations
CROSSOVER_RATIO = 0.9                       # Crossover ratio
MUTATION_RATIO = 0.1                        # Mutation ratio
EQUALITY_PENALTY = 50000                    # Equality constraint penalty coefficient
INEQUALITY_PENALTY = 0                      # Inequality constraint penalty coefficient

CONSTRAINT_ACCURACY = 0.01                  # Equality constraint accuracy

S_PROBABILITY = 0.25                        # Infeasible start solution occurrence probability

S_SORTING_GROUPING_STRATEGY = 6             # Sorting grouping strategy selection offset
S_TOURNAMENT = 10                           # Tournament selection participants
S_LINEAR_RANK = 1.9                         # Linear rank selection coefficient
S_NON_LINEAR_RANK = 0.5                     # Non-linear rank selection coefficient

C_LINEAR = 0.5                              # Linear crossover coefficient
C_BLEND = 0.5                               # Blend crossover coefficient
C_SIMULATED_BINARY = 7                      # Simulated binary crossover coefficient

M_NON_UNIFORM = 5                           # Non-uniform mutation coefficient
M_POLYNOMIAL = 60                           # Polynomial mutation coefficient
