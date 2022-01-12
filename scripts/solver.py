#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np
import pandas as pd

from typing import Tuple, List

# PROJECT MODULES
from data import ProblemSize, ProblemParameters, AlgorithmParameters
from solution import Solution


# CLASSES
class Solver:
    """
    Class to represent the solver
    """

    def __init__(self, problem_size: ProblemSize, problem_parameters: ProblemParameters,
                 algorithm_parameters: AlgorithmParameters) -> None:
        """
        Constructor
        :param problem_size: Problem size
        :param problem_parameters: Problem parameters
        :param algorithm_parameters: Algorithm parameters
        """

        # Problem size
        self.problem_size = problem_size

        # Problem parameters
        self.problem_parameters = problem_parameters

        # Algorithm parameters
        self.algorithm_parameters = algorithm_parameters
        self.population_size = algorithm_parameters.population_size
        self.n_generations = algorithm_parameters.n_generations
        self.crossover_ratio = algorithm_parameters.crossover_ratio

        self.selection_method = algorithm_parameters.methods[1]
        self.selection_method_value = algorithm_parameters.methods_values[1]
        self.crossover_method = algorithm_parameters.methods[2]
        self.crossover_method_value = algorithm_parameters.methods[2]

        # Number of parent in the sorting grouping strategy
        self.n_parent = 0

        # First generation
        self.population = []

        # Random
        if algorithm_parameters.methods[0] == 0:
            self.population = [Solution(problem_size, problem_parameters, algorithm_parameters)
                               for _ in range(self.population_size)]

        # Best or worst
        else:
            idx = -1 if algorithm_parameters.methods[0] == 1 else 0

            for _ in range(self.population_size):
                solutions = [Solution(problem_size, problem_parameters, algorithm_parameters)
                             for _ in range(algorithm_parameters.methods_values[0])]
                self.population.append(sorted(solutions, key=lambda sol: sol.fitness)[idx])

        self.sorted_population = self.population

    def roulette_wheel(self) -> Solution:
        """
        Method to select a solution with the roulette wheel method
        :return: Selected solution
        """

        # Create the roulette wheel
        population_fitness = [solution.fitness for solution in self.population]
        fitness_sum = sum(population_fitness)
        wheel = np.zeros(self.population_size + 1)

        for i in range(1, self.population_size + 1):
            wheel[i] = population_fitness[i - 1] / fitness_sum + wheel[i - 1]

        # Draw a random solution
        r = np.random.uniform()

        for i in range(self.population_size):
            if wheel[i] < r < wheel[i + 1]:
                return self.population[i]

        return self.population[0]

    def tournament(self) -> Solution:
        """
        Method to select a solution with the tournament method
        :return: Selected solution
        """

        solutions = [self.population[np.random.randint(self.population_size)] for _ in range(self.selection_method_value)]

        return sorted(solutions, key=lambda sol: sol.fitness)[-1]

    def rank(self) -> Solution:
        """
        Method to select a solution with the rank method
        :return: Selected solution
        """

        # Create the roulette wheel
        wheel = np.zeros(self.population_size + 1)

        # Linear
        if self.selection_method == 3:
            eta = self.selection_method_value

            for i in range(1, self.population_size + 1):
                wheel[i] = (eta - 2 * (eta - 1) * (i - 1) / (self.population_size - 1)) / self.population_size + wheel[i - 1]

        else:
            q = self.selection_method_value

            for i in range(1, self.population_size + 1):
                wheel[i] = q * (1 - q)**(i - 1) + wheel[i - 1]

        # Draw a random solution
        r = np.random.uniform()

        for i in range(self.population_size):
            if wheel[i] < r < wheel[i + 1]:
                return self.sorted_population[i]

        return self.population[0]

    def select(self) -> Tuple[Solution, Solution]:
        """
        Method to select the parents from the population
        :return: Parents
        """

        # Roulette wheel
        if self.selection_method == 0:
            parent1 = self.roulette_wheel()
            parent2 = self.roulette_wheel()

            while parent1 == parent2:
                parent1 = self.roulette_wheel()
                parent2 = self.roulette_wheel()

            return parent1, parent2

        # Sorting grouping strategy
        elif self.selection_method == 1:
            parent1 = self.sorted_population[self.n_parent]
            idx = (self.n_parent + self.selection_method_value) % ((self.population_size + 1)//2) + self.population_size//2
            parent2 = self.sorted_population[idx]
            self.n_parent = (self.n_parent + 1) % (self.population_size//2)

            return parent1, parent2

        # Tournament
        elif self.selection_method == 2:
            parent1 = self.tournament()
            parent2 = self.tournament()

            while parent1 == parent2:
                parent1 = self.tournament()
                parent2 = self.tournament()

            return parent1, parent2

        # Rank
        else:
            parent1 = self.rank()
            parent2 = self.rank()

            while parent1 == parent2:
                parent1 = self.rank()
                parent2 = self.rank()

            return parent1, parent2

    def crossover(self) -> Tuple[Solution, Solution]:
        """
        Method to perform the crossover
        :return: Tuple: (Offspring1, offspring2)
        """

        # Choose the parents
        parent1, parent2 = self.select()

        # Check if to perform the crossover
        if np.random.uniform() >= self.crossover_ratio:
            return parent1, parent2

        # Uniform
        if self.crossover_method == 0:
            alpha = np.random.uniform(size=(self.problem_size.M, self.problem_size.N))
            offspring1 = parent1.mul(1 - alpha) + parent2.mul(alpha)
            offspring2 = parent1.mul(alpha) + parent2.mul(1 - alpha)

        # Point
        elif self.crossover_method == 1:
            mat1 = np.zeros((self.problem_size.M, self.problem_size.N))
            mat2 = np.zeros((self.problem_size.M, self.problem_size.N))

            r = np.random.randint(self.problem_size.M)
            mat1[:r, :] = parent1.X[:r, :]
            mat1[r:, ] = parent2.X[r:, :]
            mat2[:r, :] = parent2.X[:r, :]
            mat2[r:, :] = parent1.X[r:, :]

            offspring1 = Solution(self.problem_size, self.problem_parameters, self.algorithm_parameters, mat=mat1)
            offspring2 = Solution(self.problem_size, self.problem_parameters, self.algorithm_parameters, mat=mat2)

        # Linear
        elif self.crossover_method == 2:
            alpha = self.crossover_method_value
            solutions = [(parent1 + parent2).mul(alpha), parent1.mul(-alpha) + parent2.mul(3 * alpha),
                         parent1.mul(3 * alpha) + parent2.mul(-alpha)]
            fitnesses = [sol.fitness for sol in solutions]
            solutions.pop(fitnesses.index(max(fitnesses)))
            offspring1, offspring2 = solutions[0], solutions[1]

        # Blend
        elif self.crossover_method == 3:
            mat1 = np.zeros((self.problem_size.M, self.problem_size.N))
            mat2 = np.zeros((self.problem_size.M, self.problem_size.N))
            alpha = self.crossover_method_value

            for i in range(self.problem_size.M):
                for j in range(self.problem_size.N):
                    smaller = parent1[i, j] if parent1[i, j] < parent2[i, j] else parent1[i, j]
                    bigger = parent2[i, j] if parent1[i, j] < parent2[i, j] else parent2[i, j]
                    low = max([0, smaller - alpha * (bigger - smaller)])
                    high = min([1, bigger + alpha * (bigger - smaller)])
                    mat1[i, j] = np.random.uniform(low=low, high=high)
                    mat2[i, j] = np.random.uniform(low=low, high=high)

            offspring1 = Solution(self.problem_size, self.problem_parameters, self.algorithm_parameters, mat=mat1)
            offspring2 = Solution(self.problem_size, self.problem_parameters, self.algorithm_parameters, mat=mat2)

        # Simulated binary
        else:
            n = self.crossover_method_value
            u = np.random.uniform()
            beta = (2 * u)**(1 / (1 + n)) if u <= 0.5 else (1 / (2 - 2 * u))**(1 / (1 + n))
            offspring1 = (parent1.mul(1 + beta) + parent2.mul(1 - beta)).mul(0.5)
            offspring2 = (parent1.mul(1 - beta) + parent2.mul(1 + beta)).mul(0.5)

        # Check the constraints
        if offspring1.is_feasible() and offspring2.is_feasible():
            return offspring1, offspring2

        else:
            return parent1, parent2

    def genetic_algorithm(self) -> List[Solution]:
        """
        Method to perform the genetic algorithm
        :return: Best solutions throughout the history
        """

        history: List[Solution] = []

        for n_generation in range(self.n_generations):
            new_generation: List[Solution] = []
            self.sorted_population = sorted(self.population, key=lambda sol: sol.fitness, reverse=True)
            history.append(self.sorted_population[0])

            for _ in range(0, self.population_size, 2):

                # Selection and crossover
                offspring1, offspring2 = self.crossover()

                # Mutation
                offspring1.mutate(n_generation), offspring2.mutate(n_generation)
                new_generation.append(offspring1), new_generation.append(offspring2)

            self.population = new_generation

        return history

    def save_problem_parameters(self) -> None:
        """
        Function to save the problem parameters
        """

        d = {"c": self.problem_parameters.c, "d": self.problem_parameters.d, "V": self.problem_parameters.V}

        for name, array in d.items():
            pd.DataFrame(array).to_csv(path_or_buf=f"{name}.csv", index=False)
