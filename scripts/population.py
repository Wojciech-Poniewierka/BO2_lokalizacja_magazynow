#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import random
import numpy as np

from typing import Tuple, List
from copy import deepcopy

# PROJECT MODULES
from config import POPULATION_SIZE, MIN_FITNESS, MAX_GENERATIONS, CROSSOVER_RATIO
from solution import Solution


# CLASSES
class Population:
    """
    Class to represent the solutions population
    """

    def __init__(self, n_warehouses: int, n_shops: int, f: np.ndarray, s: np.ndarray, c: np.ndarray, b: np.ndarray,
                 d: np.ndarray, v: np.ndarray) -> None:
        """
        Constructor
        :param n_warehouses: Number of warehouses
        :param n_shops: Number of shops
        :param f: Factory to warehouses transport costs, dim: Mx1
        :param s: Warehouses to shops transport costs, dim: MxN
        :param c: Warehouses capacities, dim: Mx1
        :param b: Warehouses building costs, dim: Mx1
        :param d: Demands of the shops, dim: Nx1
        :param v: Sugar values established between warehouses and shops, dim: MxN
        """

        # Number of facilities
        self.M = n_warehouses
        self.N = n_shops

        # Population
        self.generation = [Solution(self.M, self.N, f, s, c, b, d, v) for _ in range(POPULATION_SIZE)]
        self.fitness: List[float] = []
        self.best_fitness: Tuple[float, int] = (0, 0)
        self.n_generations: int = 0

    def evaluate(self) -> None:
        """
        Method to evaluate the fitness of each solution in the population
        """

        self.fitness = [self.generation[i].calculate_fitness() for i in range(POPULATION_SIZE)]
        self.best_fitness = sorted([(self.fitness[i], i) for i in range(POPULATION_SIZE)], key=lambda tup: tup[0])[-1]
        self.n_generations += 1

    def roulette_wheel(self) -> Solution:
        """
        Method to draw a solution
        :return: Tuple: Drawn solution
        """

        # Get the fitness ratios
        fitness_sum = sum(self.fitness)
        fitness_ratios = [self.fitness[i] / fitness_sum * 100 for i in range(POPULATION_SIZE)]

        # Create the roulette wheel
        wheel = np.zeros(POPULATION_SIZE + 1)

        for i in range(1, POPULATION_SIZE + 1):
            wheel[i] = fitness_ratios[i - 1] + wheel[i - 1]

        # Draw a random solution
        r = np.random.randint(100)

        for i in range(POPULATION_SIZE):
            if wheel[i] < r < wheel[i + 1]:
                return self.generation[i]

        return self.generation[0]

    def select(self) -> Solution:
        """
        Method to select the parent from the population
        :return: Parent
        """

        return self.roulette_wheel()

    # Blend crossover
    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """
        Method to perform the crossover
        :param parent1: First parent
        :param parent2: Second parent
        :return: Offsprings
        """

        if random.uniform(0, 1) < CROSSOVER_RATIO:
            alpha = 0.1
            mat1 = np.zeros((self.M, self.N))
            mat2 = np.zeros((self.M, self.N))

            for i in range(self.M):
                for j in range(self.N):
                    gamma = (1 + 2 * alpha) * random.uniform(0, 2 / self.N) - alpha
                    mat1[i, j] = (1 - gamma) * parent1[i, j] + gamma * parent2[i, j]
                    mat2[i, j] = gamma * parent1[i, j] + (1. - gamma) * parent2[i, j]

            offspring1 = deepcopy(parent1)
            offspring1.X = mat1

            offspring2 = deepcopy(parent2)
            offspring2.X = mat2

            offspring1.scale()
            offspring2.scale()

            if not offspring1.is_feasible() or not offspring2.is_feasible():
                return self.crossover(parent1, parent2)

            else:
                return offspring1, offspring2

        else:
            return parent1, parent2

    def genetic_algorithm(self) -> Tuple[Tuple[float, Solution], List[Solution], List[float]]:
        """
        Method to perform the genetic algorithm
        :return: Tuple: (Tuple: (Best solution fitness, Best solution), Best generation, Best fitnesses)
        """

        best_solution: Tuple[float, Solution] = (self.generation[0].calculate_fitness(), self.generation[0])
        generation = [solution for solution in self.generation]

        while self.n_generations < MAX_GENERATIONS:
            self.evaluate()

            for i in range(0, POPULATION_SIZE, 2):
                parent1 = self.select()
                parent2 = self.select()

                while parent1 == parent2:
                    parent1 = self.select()
                    parent2 = self.select()

                offspring1, offspring2 = self.crossover(parent1, parent2)

                offspring1.mutate()
                offspring2.mutate()

                generation[i] = offspring1
                generation[i + 1] = offspring2

            if self.best_fitness[0] < best_solution[0]:
                best_solution = (self.best_fitness[0], self.generation[self.best_fitness[1]])

            self.generation = generation
            self.generation[0] = best_solution[1]

            if best_solution[0] < MIN_FITNESS:
                print("Reached minimum fitness")

                break

        return best_solution, self.generation, self.fitness