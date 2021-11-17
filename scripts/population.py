#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np
import random as rnd

from typing import Tuple, List
from copy import deepcopy

# PROJECT MODULES
from solution import Solution


# GLOBAL VARIABLES
POPULATION_SIZE: int = 30
MIN_FITNESS: float = 0.01
MAX_INDIVIDUALS: int = 100
CROSSOVER_RATIO: float = 0.8


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
        :param f:
        :param s:
        :param c:
        :param b: Warehouses building costs, dim: Mx1
        :param d: Demands of the shops, dim: Nx1
        :param v: Sugar values established between warehouses and shops, dim: MxN
        """

        # Number of facilities
        self.M = n_warehouses
        self.N = n_shops

        # Problem parameters
        self.f = f
        self.S = s
        self.c = c
        self.b = b
        self.d = d
        self.V = v

        # Population
        self.generation = [Solution(self.M, self.N, self.f, self.S, self.c, self.b, self.d, self.V)
                           for _ in range(POPULATION_SIZE)]

        self.fitness: List[float] = []
        self.best_fitness: Tuple[float, int] = (0, 0)
        self.n_individuals: int = 0

    def evaluate(self) -> None:
        """
        Method to evaluate the fitness of each solution in the population
        """

        self.fitness = [self.generation[i].calculate_fitness() for i in range(POPULATION_SIZE)]
        self.best_fitness = sorted([(self.fitness[i], i) for i in range(POPULATION_SIZE)], key=lambda tup: tup[0])[0]
        self.n_individuals += POPULATION_SIZE

    # def select(self):
    #     """
    #     Method to select best individuals from the population
    #     """
    #
    #     population = np.array([x for x, fitness in population_fitness])
    #     next_generation = population[:BEST_SAMPLE]
    #     population = population[BEST_SAMPLE:]
    #
    #     for _ in range(LUCKY_FEW):
    #         x = rnd.choice(population)
    #         next_generation.append(x)
    #         population.remove(x)
    #
    #     rnd.shuffle(next_generation)
    #
    #     return next_generation

    def roulette_wheel(self) -> Solution:
        """
        Method to draw a solution
        :return: Drawn solution
        """

        fitness_sum = sum(self.fitness)
        fitness_ratio = [self.fitness[i] / fitness_sum * 100 for i in range(POPULATION_SIZE)]

        wheel = np.zeros(POPULATION_SIZE + 1)
        r = np.random.randint(100)

        for i in range(1, POPULATION_SIZE + 1):
            wheel[i] = fitness_ratio[i - 1] + wheel[i - 1]

        for i in range(POPULATION_SIZE):
            if wheel[i] < r < wheel[i + 1]:
                return self.generation[i]

        return self.generation[0]

    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """
        Method to perform the crossover
        :param parent1: First parent
        :param parent2: Second parent
        :return: Offsprings
        """

        if rnd.uniform(0, 1) < CROSSOVER_RATIO:
            alpha = 0.1
            mat1 = np.zeros((self.M, self.N))
            mat2 = np.zeros((self.M, self.N))

            for i in range(self.M):
                for j in range(self.N):
                    gamma = (1. + 2. * alpha) * rnd.uniform(0, 2 / self.N) - alpha
                    mat1[i, j] = (1. - gamma) * parent1[(i, j)] + gamma * parent2[(i, j)]
                    mat2[i, j] = gamma * parent1[(i, j)] + (1. - gamma) * parent2[(i, j)]

            offspring1 = deepcopy(parent1)
            offspring1.X = mat1

            offspring2 = deepcopy(parent2)
            offspring2.X = mat2

            if not offspring1.is_feasible() or not offspring2.is_feasible():
                return self.crossover(parent1, parent2)

        else:
            offspring1 = parent1
            offspring2 = parent2

        return offspring1, offspring2

    def genetic_algorithm(self) -> Tuple[Tuple[float, Solution], List[Solution], List[float]]:
        """
        Method to perform the genetic algorithm
        :return:
        """

        best_solution: Tuple[float, Solution] = (float("inf"), self.generation[0])
        generation = [solution for solution in self.generation]

        while self.n_individuals < MAX_INDIVIDUALS:
            self.evaluate()

            for i in range(0, POPULATION_SIZE, 2):
                parent1 = self.roulette_wheel()
                parent2 = self.roulette_wheel()

                while parent1 == parent2:
                    parent1 = self.roulette_wheel()
                    parent2 = self.roulette_wheel()

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
                print("Reached min")

                break

        return best_solution, self.generation, self.fitness
