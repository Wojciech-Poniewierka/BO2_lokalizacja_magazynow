#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN MODULES
import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.ttk as ttk

from typing import Tuple, List, Optional

# PROJECT MODULES
from data import ProblemSize, ProblemParameters, AlgorithmParameters
from solution import Solution


# CLASSES
class Solver:
    """
    Class to represent the solver
    """

    def __init__(self, problem_size: ProblemSize, problem_parameters: ProblemParameters,
                 algorithm_parameters: AlgorithmParameters,
                 progress_with_frame: Tuple[ttk.Progressbar, tk.Frame]) -> None:
        """
        Constructor
        :param problem_size: Problem size
        :param problem_parameters: Problem parameters
        :param algorithm_parameters: Algorithm parameters
        :param progress_with_frame: Progress bar with its frame
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
        self.crossover_method_value = algorithm_parameters.methods_values[2]

        # Progress bar
        self.progress, self.progress_frame = progress_with_frame
        self.progress["value"] = 0

        # Number of parent in the sorting grouping strategy
        self.n_parent: int = 0

        # First generation
        self.population = [Solution(problem_size, problem_parameters, algorithm_parameters)
                           for _ in range(self.population_size)]
        self.sorted_population = self.population

    def spin(self) -> Tuple[Solution, Solution]:
        """
        Method to select the parents with one of those methods: roulette wheel, linear rank, non-linear rank
        :return: Selected parents
        """

        # Create the wheel
        wheel = np.zeros(self.population_size + 1)

        # Roulette wheel
        if self.selection_method == 0:
            for i in range(1, self.population_size + 1):
                wheel[i] = self.sorted_population[i - 1].fitness + wheel[i - 1]

        # Linear
        if self.selection_method == 3:
            eta = self.selection_method_value

            for i in range(1, self.population_size + 1):
                wheel[i] = eta * (self.population_size - 1) + 2 * (1 - eta) * (i - 1) + wheel[i - 1]

        else:
            q = self.selection_method_value

            for i in range(1, self.population_size + 1):
                wheel[i] = q * (1 - q)**(i - 1) + wheel[i - 1]

        # Draw 2 different random solutions
        parents: List[Solution, Solution] = [self.sorted_population[0], self.sorted_population[0]]
        idx: List[int, int] = [0, 0]

        for i in range(2):
            r = np.random.uniform(low=0, high=wheel[-1])

            for j in range(self.population_size):
                if wheel[j] < r < wheel[j + 1]:
                    parents[i], idx[i] = self.sorted_population[j], j
                    break

        if idx[0] != idx[1]:
            return parents[0], parents[1]

        elif sum(idx) == 0:
            return self.sorted_population[0], self.sorted_population[1]

        else:
            return self.sorted_population[idx[0] - 1], self.sorted_population[idx[0]]

    def select(self) -> Tuple[Solution, Solution]:
        """
        Method to select the parents from the population
        :return: Parents
        """

        # Roulette wheel, linear rank or non-linear rank
        if self.selection_method in (0, 3, 4):
            return self.spin()

        # Sorting grouping strategy
        elif self.selection_method == 1:
            parent1 = self.sorted_population[self.n_parent]
            idx = (self.n_parent + self.selection_method_value) % ((self.population_size + 1)//2) + self.population_size//2
            parent2 = self.sorted_population[idx]
            self.n_parent = (self.n_parent + 1) % (self.population_size//2)

            return parent1, parent2

        # Tournament
        elif self.selection_method == 2:
            parents: List[Solution] = []

            for _ in range(2):
                participants: List[Solution] = []
                idx: List[int] = []

                while len(participants) < self.selection_method_value:
                    i = np.random.randint(self.population_size)

                    if i not in idx:
                        participants.append(self.sorted_population[i])
                        idx.append(i)

                parents.append(sorted(participants, key=lambda p: p.fitness, reverse=True)[0])

            return parents[0], parents[1]

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

            r = np.random.randint(self.problem_size.N)
            mat1[:, :r] = parent1.X[:, :r]
            mat1[:, r:] = parent2.X[:, r:]
            mat2[:, :r] = parent2.X[:, :r]
            mat2[:, r:] = parent1.X[:, r:]

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
            eta = self.crossover_method_value

            # for i in range(self.problem_size.M):
            #     for j in range(self.problem_size.N):
            #         gamma = 1 + 2 * min(parent1[i, j], 1 - parent2[i, j]) / (parent1[i, j] - parent2[i, j])
            #         alpha = 2 - gamma**(-eta - 1)
            #         u = np.random.uniform()
            #
            #         if u <= 1 / alpha:
            #             beta = (u * alpha)**(1 / (eta + 1))
            #
            #         else:
            #             beta = 1 / (2 - u * alpha)**(1 / (eta + 1))

            u = np.random.uniform()
            beta = (2 * u)**(1 / (1 + eta)) if u <= 0.5 else (1 / (2 - 2 * u))**(1 / (1 + eta))
            offspring1 = (parent1.mul(1 + beta) + parent2.mul(1 - beta)).mul(0.5)
            offspring2 = (parent1.mul(1 - beta) + parent2.mul(1 + beta)).mul(0.5)

        # Check the constraints
        if offspring1.is_correct() and offspring2.is_correct():
            return offspring1, offspring2

        else:
            return parent1, parent2

    def genetic_algorithm(self) -> Tuple[List[Solution], List[Optional[Solution]]]:
        """
        Method to perform the genetic algorithm
        :return: Best solutions and best feasible solutions throughout the history
        """

        best_solutions: List[Solution] = []
        best_feasible_solutions: List[Optional[Solution]] = []
        progress_ticks: int = 0

        for n_generation in range(self.n_generations):
            if n_generation > self.n_generations / 100 * progress_ticks - 1:
                progress_ticks += 10
                self.progress["value"] = progress_ticks
                self.progress_frame.update_idletasks()

            new_generation: List[Solution] = []
            self.sorted_population = sorted(self.population, key=lambda sol: sol.fitness, reverse=True)
            best_solutions.append(self.sorted_population[0])

            for solution in self.sorted_population:
                if solution.is_feasible():
                    best_feasible_solutions.append(solution)
                    break

            else:
                best_feasible_solutions.append(None)

            for _ in range(0, self.population_size, 2):

                # Selection and crossover
                offspring1, offspring2 = self.crossover()

                # Mutation
                new_generation.append(offspring1.mutate(n_generation))
                new_generation.append(offspring2.mutate(n_generation))

            self.population = new_generation
            print(sum([(np.abs(solution.X.sum(axis=0) - 1) < 0.01 * np.ones((1, self.problem_size.N))).all()
                       for solution in self.population]), end="\t")

            # self.population = sorted(self.population + new_generation, key=lambda sol: sol.fitness, reverse=True)[:self.population_size]

        print()

        self.progress["value"] = 100

        return best_solutions, best_feasible_solutions

    def save_problem_parameters(self) -> None:
        """
        Function to save the problem parameters
        """

        d = {"c": self.problem_parameters.c, "d": self.problem_parameters.d, "V": self.problem_parameters.V}

        for name, array in d.items():
            pd.DataFrame(array).to_csv(path_or_buf=f"{name}.csv", index=False)
