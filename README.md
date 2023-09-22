# Introduction to AI - Assignment I

This repository contains code for the second assignment of the `Introduction to Artificial Intelligence` course at Addis Ababa Institute of technology. The objectives of the assignment include getting acquinted to Local Search, and being able to solve problems using it which would otherwise have been hard, even impossible to solve.

This project will run experiments to solve the problems using different local search algorithms. To run them use the following command.

## Usage:

`python runner.py problem <problem_name> algorithm <algorithm_name> [OPTIONS]`

## OPTIONAL COMMANDS

`--file  <file_path>` Specify the file path for the problem (default: knapsackdata.txt for knapsack, tspdata.txt for tsp)

`--num-of-items <number_of_items>` Specify the number of items (default: 10 for knapsack, 16 for tsp)

`--experiment` Display a graph of the runtime comparision

`--all` Run all algorithms at once

`--verbose` Enable verbose mode for detailed live output

`--num-of-generations <num_generations>` Set the number of generations for genetic algorithm(default: 2000 generations)

`--population-size <population_size>` Set the population size for genetic algorithm (default: 100)

`--mutation-rate <mutation_rate>` Set the mutation rate for genetic algorithm (default: 0.1)

`--crossover-rate <crossover_rate>` Set the crossover rate for genetic algorithm (default: 0.8)

`--temperature <initial_temperature>` Set the initial temperature for simulated annealing (default: 100.0)

`--cooling-rate <cooling_rate>` Set the cooling rate for simulated annealing (default: 0.95)

`--help` See the usage of the algorithm
