
from util import *

class Knapsack:
    def __init__(self, file, number_of_items= None):
        self.number_of_items = number_of_items or 20
        self.knapsack_weight_limit, self.items =  knapsack_file_parser(file, self.number_of_items)
        self.curr_state = generate_initial_knapsack_state(self.knapsack_weight_limit, self.items)
        self.initial_state = self.curr_state[:]
        self.best_state = self.curr_state[:]
        self.best_state_fitness = get_items_weight_and_value(self.best_state, self.items)[1]


    def hillclimbing(self, verbose = False):

        while True:
            best_neighbor_state = generate_best_immediate_neighbour(self.knapsack_weight_limit, self.curr_state, self.items)
            current_state_value = get_items_weight_and_value(self.curr_state, self.items)[1]
            best_neighbor_value = get_items_weight_and_value(best_neighbor_state, self.items)[1]

            if best_neighbor_value > current_state_value:
                self.curr_state = best_neighbor_state
                self.best_state = best_neighbor_state
                self.best_state_fitness = best_neighbor_value

            else:
                break

            if verbose:
                print("    Fitness:", self.best_state_fitness, end="\r")

    def simulated_annealing(self, initial_temperature= None, cooling_rate= None,verbose = False):
        temperature = initial_temperature or 20
        COOLING_RATE = cooling_rate or 0.9999

        # start With a high temperature
        while temperature > 0.01:
            # generate a random neighbor
            next_state = generate_random_neighbour(self.curr_state)

            # if the next state is not valid, if it exceeds the knapsack's weight limit, don't even consider it
            if get_items_weight_and_value(next_state, self.items)[0] > self.knapsack_weight_limit:
                continue

            curr_state_fitness = get_items_weight_and_value(self.curr_state, self.items)[1]
            next_state_fitness = get_items_weight_and_value(next_state, self.items)[1]

            if curr_state_fitness > self.best_state_fitness:
                self.best_state = self.curr_state[:]
                self.best_state_fitness = curr_state_fitness

            acceptance_prob = acceptance_probability(curr_state_fitness, next_state_fitness, temperature)

            if acceptance_prob > random():
                self.curr_state = next_state

            if verbose:
                print("    Temp: " + "{:.2f}".format(temperature), " ||  Fitness:", curr_state_fitness, end="\r")

            temperature *= COOLING_RATE
        
        self.best_state_fitness = get_items_weight_and_value(self.best_state, self.items)[1]


    def genetic_algorithm(self, population_size, max_generations = None, crossover_rate= None, mutation_rate= None, verbose = False):
        POPULATION_SIZE = population_size or 100
        MAX_GENERATIONS = max_generations or 1000
        CROSSOVER_RATE = crossover_rate or 0.8
        MUTATION_RATE = mutation_rate or 0.2

        # generate a random population
        curr_population = [generate_initial_knapsack_state(self.knapsack_weight_limit, self.items) for _ in range(POPULATION_SIZE)]
        # generation 0 sorted by value
        initial_population = sorted(curr_population, key=lambda x: get_items_weight_and_value(x, self.items)[1], reverse=True)
        best_state = initial_population[0]
        
        for generation in range(MAX_GENERATIONS):
            # generate the next generation
            next_generation = []
            for _ in range(POPULATION_SIZE//2):
                parent_1 = tournament_selection_knapsack(curr_population, self.items)
                parent_2 = tournament_selection_knapsack(curr_population, self.items)
                child_1, child_2 = crossover_knapsack(parent_1.copy(), parent_2.copy(), CROSSOVER_RATE)
                mutate_knapsack(child_1, MUTATION_RATE)
                mutate_knapsack(child_2, MUTATION_RATE)
                next_generation.append(child_1)
                next_generation.append(child_2)

            # sort the next generation by value
            sorted_next_generation = sorted(next_generation, key=lambda x: get_items_weight_and_value(x, self.items)[1], reverse=True)
            combined_curr_population = elitism_merge_knapsack(curr_population, sorted_next_generation, POPULATION_SIZE, self.items, self.knapsack_weight_limit)
            best_state = combined_curr_population[0]
            curr_population = combined_curr_population

            if verbose:
                print(f'    Gen: {generation}, {" " * (4-len(str(generation)))}  ||  Best total value: {get_items_weight_and_value(best_state, self.items)[1]}', end='\r')

        return initial_population[0], best_state


    def clear(self):
        self.curr_state = generate_initial_knapsack_state(self.knapsack_weight_limit, self.items)
        self.initial_state = self.curr_state
        self.best_state = self.curr_state[:]
        self.best_state_fitness = get_items_weight_and_value(self.best_state, self.items)[1]

        