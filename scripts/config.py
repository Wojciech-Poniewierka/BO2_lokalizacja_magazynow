#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
from typing import Tuple, List

# GLOBAL VARIABLES
# SOLUTION
MUTATION_RATIO: float = 0.1
NOISE = 0.0001
CONSTRAINT_ACCURACY = 0.1

# POPULATION
POPULATION_SIZE: int = 30
MIN_FITNESS: float = 0.01
MAX_GENERATIONS: int = 15
CROSSOVER_RATIO: float = 0.8

# SOLVER
M = 4
N = 10

TRANSPORT_COST_AMPLIFIER: float = 0.5
BUILDING_COST_AMPLIFIER: float = 1
CAPACITY_RANGE: Tuple[float, float] = (10000, 25000)
DEMAND_RANGE: Tuple[float, float] = (100, 7000)
COST_RANGE: Tuple[float, float] = (5, 12)

# --------------------------------------------------#
# TYPE ALIASES
Location = Tuple[float, float]
Locations = Tuple[List[Location], List[Location]]
