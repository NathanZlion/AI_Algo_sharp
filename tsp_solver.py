
from random import choice, random, sample
from Modules.graph import Graph
from util import *

class Tsp_problem:

    def __init__(self, file, number_of_items= None):
        self.number_of_items = number_of_items or 16
        self.city_map = tsp_file_parser(file)
        self.city_graph = self.__build_city_graph()
        self.cities = sample(list(self.city_graph.get_nodes().keys()), self.number_of_items)
        self.start_city = choice(self.cities)
        self.between_cities_distance = self.__get_distance_between_cities()
        self.curr_state = generate_initial_tsp_state(self.cities)
        self.curr_state_fitness = get_fitness_tsp(self.curr_state, self.between_cities_distance)
        self.initial_state = self.curr_state[:]
        self.best_state = self.curr_state
        self.best_state_fitness = self.curr_state_fitness


    def hillclimbing(self, verbose = False):
        """
        Hill Climbing algorithm for TSP
        """
        while True:
            best_neighbor = generate_best_immediate_neighbour_tsp(self.best_state, self.between_cities_distance)
            if get_fitness_tsp(best_neighbor, self.between_cities_distance) >= self.best_state_fitness:
                break
            self.best_state = best_neighbor
            self.best_state_fitness = get_fitness_tsp(best_neighbor, self.between_cities_distance)

            if verbose:
                print("    Fitness:", get_fitness_tsp(self.best_state, self.between_cities_distance), end="\r")

    def simulated_annealing(self, initial_temperature= None, cooling_rate= None,verbose = False):
        temperature = initial_temperature or 20
        COOLING_RATE = cooling_rate or 0.9999


        # start With a high temperature
        while temperature > 0.01:
            # generate neighbor randomly
            neighbor_state = generate_random_neighbor_tsp(self.best_state)
            neighbor_state_fitness = get_fitness_tsp(neighbor_state, self.between_cities_distance)

            # if the next state is not valid, if it exceeds the knapsack's weight limit, don't even consider it
            if neighbor_state_fitness < self.curr_state_fitness:
                self.curr_state = neighbor_state[:]
                self.curr_state_fitness = get_fitness_tsp(self.curr_state, self.between_cities_distance)

            else:
                acceptance_prob = acceptance_probability(int(neighbor_state_fitness), self.curr_state_fitness, temperature)
                if random() > acceptance_prob:
                    self.curr_state = neighbor_state[:]
                    self.curr_state_fitness = get_fitness_tsp(self.curr_state, self.between_cities_distance)

            if self.curr_state_fitness < self.best_state_fitness:
                self.best_state = self.curr_state[:]
                self.best_state_fitness = self.curr_state_fitness

            if verbose:
                print(" ~   Temp: " + "{:.2f}".format(temperature), " ||  Best Path Distance:", get_fitness_tsp(self.best_state, self.between_cities_distance) , "_" , end="\r")

            temperature *= COOLING_RATE


    def genetic_algorithm(self, population_size= None, max_generations= None, crossover_rate= None, mutation_rate= None, verbose = False):
        POPULATION_SIZE = population_size or 100
        MAX_GENERATIONS = max_generations or 2000
        CROSSOVER_RATE = crossover_rate or 0.8
        MUTATION_RATE = mutation_rate or 0.2

        # generate a random population
        curr_population = [generate_initial_tsp_state(self.cities) for _ in range(POPULATION_SIZE)]
        # generation 0 sorted by value
        curr_population = sorted(curr_population, key=lambda x: get_fitness_tsp(x, self.between_cities_distance))
        initial_state = curr_population[0]
        best_state = curr_population[0]
        best_state_fitness = get_fitness_tsp(best_state, self.between_cities_distance)
        
        for generation in range(MAX_GENERATIONS+1):
            # generate the next generation
            next_generation = []
            for _ in range(POPULATION_SIZE//2):
                parent_1 = tournament_selection_tsp(curr_population, self.between_cities_distance)
                parent_2 = tournament_selection_tsp(curr_population, self.between_cities_distance)

                child_1, child_2 = crossover_tsp(parent_1.copy(), parent_2.copy(), CROSSOVER_RATE)
                mutate_tsp(child_1, MUTATION_RATE)
                mutate_tsp(child_2, MUTATION_RATE)
                next_generation.append(child_1)
                next_generation.append(child_2)

            # sort the next generation by value
            sorted_next_generation = sorted(next_generation, key=lambda x: get_fitness_tsp(x, self.between_cities_distance))
            combined_curr_population = elitism_merge_tsp(curr_population, sorted_next_generation, POPULATION_SIZE, self.between_cities_distance)
            current_best_state = combined_curr_population[0]
            current_best_state_fitness = get_fitness_tsp(current_best_state, self.between_cities_distance)

            if current_best_state_fitness < best_state_fitness:
                best_state = current_best_state[:]
                best_state_fitness = current_best_state_fitness

            curr_population = combined_curr_population

            if verbose:
                print(f'    Gen: {generation}, {" " * (4-len(str(generation)))} ||  Best Path Distance: {get_fitness_tsp(best_state, self.between_cities_distance)}',  end="\r")

        return initial_state, best_state

    def clear(self):
        self.cities = sample(list(self.city_graph.get_nodes().keys()), self.number_of_items)
        self.start_city = choice(self.cities)
        self.between_cities_distance = self.__get_distance_between_cities()
        self.curr_state = generate_initial_tsp_state(self.cities)
        self.curr_state_fitness = get_fitness_tsp(self.curr_state, self.between_cities_distance)
        self.initial_state = self.curr_state[:]
        self.best_state = self.curr_state
        self.best_state_fitness = self.curr_state_fitness

    def __build_city_graph(self):
        """
        Build a graph of cities and their distances
        """
        city_graph = Graph()
        for city1, city2, distance in self.city_map:
            city_graph.add_edge(city1, city2, distance)        

        return city_graph
    

    def __get_distance_between_cities(self):
        """
        Get the distance between two cities
        """

        between_cities_distance = {}

        for city1 in self.cities:
            for city2 in self.cities:
                if city1 == city2:
                    between_cities_distance[(city1, city2)] = 0
                    between_cities_distance[(city2, city1)] = 0
                    continue

                path_cost = get_path_cost(self.city_graph, uniform_cost_search(self.city_graph, city1, city2))
                between_cities_distance[(city1, city2)] = path_cost
                between_cities_distance[(city2, city1)] = path_cost
        
        return between_cities_distance
