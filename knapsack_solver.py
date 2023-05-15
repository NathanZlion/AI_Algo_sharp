
from util import *

class Knapsack:
    def __init__(self, file):
        self.knapsack_weight_limit, self.items =  knapsack_file_parser(file)
        # select a random state to begin with
        self.curr_state = generate_initial_knapsack_state(self.knapsack_weight_limit, self.items)
        self.initial_state = self.curr_state


    def hillclimbing(self):

        while True:
            best_neighbor_state = generate_best_immediate_neighbour(self.knapsack_weight_limit, self.curr_state, self.items)
            current_state_value = get_items_weight_and_value(self.curr_state, self.items)[1]
            best_neighbor_value = get_items_weight_and_value(best_neighbor_state, self.items)[1]
            if best_neighbor_value > current_state_value:
                self.curr_state = best_neighbor_state
            else:
                # break if a maxima has been reached
                break

        print_analytics_knapsack(self.initial_state, self.curr_state, self.items, self.knapsack_weight_limit)


    def simulated_annealing(self):
        temperature = 20
        COOLING_RATE = 0.9999

        # start With a high temperature
        while temperature > 0.01:
            # generate a random neighbor
            next_state = generate_random_neighbour(self.curr_state)

            # if the next state is not valid, if it exceeds the knapsack's weight limit, don't even consider it
            if get_items_weight_and_value(next_state, self.items)[0] > self.knapsack_weight_limit:
                continue

            curr_state_fitness = get_items_weight_and_value(self.curr_state, self.items)[1]
            next_state_fitness = get_items_weight_and_value(next_state, self.items)[1]

            acceptance_prob = acceptance_probability(curr_state_fitness, next_state_fitness, temperature)

            if acceptance_prob > random():
                self.curr_state = next_state

            # print("Temp: " + "{:.2f}".format(temperature), " ||  Fitness:", curr_state_fitness)
                        # uncomment the above line ^^^^ to see the process of the algorithm

            temperature *= COOLING_RATE

        print_analytics_knapsack(self.initial_state, self.curr_state, self.items, self.knapsack_weight_limit)


    def genetic_algorithm(self):
        POPULATION_SIZE = 100
        MAX_GENERATIONS = 1000
        CROSSOVER_RATE = 0.8
        MUTATION_RATE = 0.2
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

            # print(f'Gen: {generation}, {" " * (4-len(str(generation)))} {get_items_weight_and_value(best_state, self.items)[1]}')
                        # uncomment the above line ^^^^ to see the process of the algorithm

        print_analytics_knapsack(initial_population[0], best_state, self.items, self.knapsack_weight_limit)

