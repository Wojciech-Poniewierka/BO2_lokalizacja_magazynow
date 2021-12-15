#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np

from typing import Tuple, List
from copy import deepcopy

# PROJECT MODULES
from solution import Solution


# CLASSES
class Population:
    """
    Class to represent the solutions population
    """

    def __init__(self, shape: Tuple[int, int],
                 problem_parameters: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                 algorithm_parameters: Tuple[float, float, float, int, float, int, float, float, float,
                                             Tuple[float, float], Tuple[float, float], Tuple[float, float]]) -> None:
        """
        Constructor
        :param shape: Problem shape
        :param problem_parameters: Tuple: (Factory to warehouses transport costs, shape: Mx1,
        Warehouses to shops transport costs, shape: MxN, Warehouses capacities, shape: Mx1,
        Warehouses building costs, shape: Mx1, Shops demands, shape: Nx1,
        Sugar values established between warehouses and shops, shape: MxN)
        :param algorithm_parameters: Tuple: (Mutation_ratio, Noise, Constraint_accuracy, Population_size, Min_fitness,
        Max_generations, Crossover_ratio, Transport_cost_amplifier, Building_cost_amplifier, Capacity_range,
        Demand_range, Cost_range)
        """

        # Number of facilities
        self.M, self.N = shape

        # Problem parameters
        self.f, self.S, self.c, self.b, self.d, self.V = problem_parameters

        # Algorithm parameters
        self.population_size = algorithm_parameters[3]
        self.max_fitness = algorithm_parameters[4]
        self.max_generations = algorithm_parameters[5]
        self.crossover_ratio = algorithm_parameters[6]
        self.algorithm_parameters = algorithm_parameters

        # Population
        self.generation = [Solution(shape, problem_parameters, algorithm_parameters)
                           for _ in range(self.population_size)]
        self.best_solution = self.generation[0]
        self.n_generations: int = 0

    def evaluate(self) -> None:
        """
        Method to evaluate the fitness of each solution in the population
        """

        self.best_solution = sorted(self.generation, key=lambda sol: sol.fitness)[-1]
        self.n_generations += 1

    def roulette_wheel(self) -> Solution:
        """
        Method to draw a solution with the roulette wheel method
        :return: Drawn solution
        """

        # Create the roulette wheel
        population_fitness = [solution.fitness for solution in self.generation]
        fitness_sum = sum(population_fitness)
        wheel = np.zeros(self.population_size + 1)

        for i in range(1, self.population_size + 1):
            wheel[i] = population_fitness[i - 1] / fitness_sum * 100 + wheel[i - 1]

        # Draw a random solution
        r = np.random.randint(100)

        for i in range(self.population_size):
            if wheel[i] < r < wheel[i + 1]:
                return self.generation[i]

        return self.generation[0]

    def select(self, method: str) -> Solution:
        """
        Method to select the parent from the population
        :param: method: Selection method
        :return: Solution chosen to be a parent
        """

        if method == "Roulette wheel":
            return self.roulette_wheel()

    def crossover(self, parent1: Solution, parent2: Solution, method: str) -> Tuple[Solution, Solution]:
        """
        Method to perform the crossover
        :param parent1: First parent
        :param parent2: Second parent
        :param method: Crossover method
        :return: Tuple: (Offspring1, offspring2)
        """

        if np.random.normal() < self.crossover_ratio:
            if method == "blend":
                alpha = 0.1
                mat1 = np.zeros((self.M, self.N))
                mat2 = np.zeros((self.M, self.N))

                for i in range(self.M):
                    for j in range(self.N):
                        gamma = (1 + 2 * alpha) * np.random.normal() - alpha
                        mat1[i, j] = (1 - gamma) * parent1[i, j] + gamma * parent2[i, j]
                        mat2[i, j] = gamma * parent1[i, j] + (1 - gamma) * parent2[i, j]

                offspring1 = Solution((self.M, self.N), (self.f, self.S, self.c, self.b, self.d, self.V),
                                      self.algorithm_parameters, mat=mat1)
                offspring2 = Solution((self.M, self.N), (self.f, self.S, self.c, self.b, self.d, self.V),
                                      self.algorithm_parameters, mat=mat2)
            
            elif method == "linear":
                solutions = [(parent1 + parent2).mul(0.5), parent1.mul(-0.5) + parent2.mul(1.5),
                        parent1.mul(-1.5) + parent2.mul(0.5)]
                
                temp_fitness, idx = float("inf"), None

                for i, solution in enumerate(solutions):
                    ftn = solution.calculate_fitness()

                    if ftn < temp_fitness:
                        temp_fitness, idx = ftn, i

                solutions.pop(idx)
                offspring1 = deepcopy(solutions[0])
                offspring2 = deepcopy(solutions[1])
            
            elif method == "mix":
                beta = np.random.normal()
                offspring1 = parent1 - (parent2 - parent1).mul(beta)
                offspring2 = parent2 + (parent2 - parent1).mul(beta)
            
            else:
                raise Exception("Wrong crossover method")

            offspring1.scale()
            offspring2.scale()

            if not offspring1.is_feasible() or not offspring2.is_feasible():
                print("here")
                return self.crossover(parent1, parent2, method)

            else:
                return offspring1, offspring2

        else:
            return parent1, parent2

    def genetic_algorithm(self) -> Solution:
        """
        Method to perform the genetic algorithm
        :return: Best solution
        """

        best_solution = self.generation[0]
        new_generation: List[Solution] = []

        while self.n_generations < self.max_generations:
            self.evaluate()

            for _ in range(0, self.population_size, 2):
                parent1 = self.select("Roulette wheel")
                parent2 = self.select("Roulette wheel")

                while parent1 == parent2:
                    parent1 = self.select("Roulette wheel")
                    parent2 = self.select("Roulette wheel")

                offspring1, offspring2 = self.crossover(parent1, parent2, "linear")

                offspring1.mutate(), offspring2.mutate()

                new_generation.append(offspring1), new_generation.append(offspring2)

            if self.best_solution.fitness < best_solution.fitness:
                best_solution = self.best_solution

            self.generation = new_generation
            self.generation[0] = best_solution

            print(best_solution.fitness)

            if best_solution.fitness > self.max_fitness:
                print("Reached maximal fitness")
                break

        return best_solution
