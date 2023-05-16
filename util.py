
# This module contains utility functions for the project.

from collections import defaultdict
import heapq
from math import e
from random import choice as rand_choice, randint, random, shuffle
from Modules.graph import Graph

###################################################
#     UTILITY FUNCTIONS FOR KNAPSACK PROBLEM      #
###################################################
def knapsack_file_parser(filename: str, number_of_items = 10) -> tuple:
    """
    Sample file format:
    ```
    -- knapsackdata.txt --
    50
    Item, weight, value, n_items
    Phone, 0.19, 1000, 5
    Laptop, 1.1, 700, 2
    ```

    returns a tuple of the form `(knapsack_weight_limit, list_of_items)`
    """
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            knapsack_weight_limit = int(lines[0])
            items = []
            for line in lines[2:]:
                if number_of_items == 0:
                    break
                number_of_items -= 1
                item = line.split(",")
                for _ in range(int(item[3])):
                    # itemname, itemweight, itemvalue
                    items.append((item[0], float(item[1]), float(item[2])))

            return knapsack_weight_limit, items

    except FileNotFoundError:
        print(f'❌ Error: File `{filename}` not found')
        exit()


def generate_initial_knapsack_state(knapsack_weight_limit: int, items: list) -> list[int]:
    """
    Generates a random initial solution. 
    """
    total_weight = 0
    state = [0 for _ in range(len(items))]
    # generate a random solution as [0, 1, 0, 1, 1, 0, 0, 1, 1, 0]
    # Item,weight,value, n_items
    # Phone,0.19,1000, 5
    for index, (item_name, item_weight, item_value) in enumerate(items):
        if total_weight + item_weight <= knapsack_weight_limit:
            if random() > 0.5:
                state[index] = 1
                total_weight += item_weight

    return state

def get_items_weight_and_value(state: list, items: list) -> tuple:
    """
    Returns the total weight and value of the items in the knapsack
    """
    total_weight = 0
    total_value = 0
    for index, (item_name, item_weight, item_value) in enumerate(items):
        if state[index] == 1:
            total_weight += item_weight
            total_value += item_value
    
    return total_weight, total_value



def generate_best_immediate_neighbour(knapsack_weight_limit: int | float, initial_state: list, items: list) -> list:
    """generates neighbours and returns the best neighbour"""
    best_neighbour = initial_state
    best_neighbour_value = get_items_weight_and_value(initial_state, items)[1]

    for index, _ in enumerate(initial_state):
        neighbour = initial_state.copy()
        neighbour[index] = 1 - neighbour[index]  # flip the bits
        neighbour_weight, neighbour_value = get_items_weight_and_value(neighbour, items)

        # if the new state doesn't exceed the weight limit and has better value
        if neighbour_weight <= knapsack_weight_limit and neighbour_value > best_neighbour_value:
            best_neighbour = neighbour
            best_neighbour_value = neighbour_value

    return best_neighbour


def generate_random_neighbour(initial_state: list) -> list:
    """generates a random neighbour and returns it."""

    # flip each bit randomly
    next_random_state = initial_state.copy()

    for index, _ in enumerate(initial_state):
        if random() > 0.5:
            next_random_state[index] = 1 - next_random_state[index]
    
    return next_random_state


def acceptance_probability(current_energy:int, next_energy:int, temperature) -> float:
    """
    returns the probability of acceptance of a solution. Uses the formula:-
    ```
    e^(current.value - next.value)/ T
    ```
    """
    if next_energy > current_energy:
        return 1.0

    energy_difference = (next_energy - current_energy)//10000
    return pow(e, energy_difference/ temperature)


def print_analytics_knapsack(intitial_state : list, final_state: list, items: list, knapsack_weight_limit: int):
    """
    Prints the analytics of the algorithm, The initial and final state of the knapsack.
    The items in it. The total weight and value of the items in the knapsack.
    """
    print("\n######################\n")
    print(f'Maximum weight: {knapsack_weight_limit}')
    print("\n####### Initial State ######\n") 
    __print_items_in_knapsack(intitial_state, items)
    
    print("\n####### Final State ######\n")
    __print_items_in_knapsack(final_state, items)
    print("\n######################\n")


def __print_items_in_knapsack(state: list, items: list):
    """
    Prints the items in the knapsack
    """

    items_count = defaultdict(int)
    for index, (item_name, item_weight, item_value) in enumerate(items):
        if state[index] == 1:
            items_count[item_name] += 1
    
    print("Items in the knapsack:")
    for item_name, count in items_count.items():
        print(f"{item_name} x {count}", end=", ")
    
    total_total, total_value = get_items_weight_and_value(state, items)
    print()
    print(f"Total weight: {total_total}")
    print(f"Total value: {total_value}")



def crossover_knapsack(parent_1, parent_2, CROSSOVER_RATE) -> tuple:

    if random() > CROSSOVER_RATE:
        return parent_1, parent_2

    # select a random index
    index = randint(0, len(parent_1) - 1)
    # return the child
    return parent_1[:index] + parent_2[index:], parent_1[index:] + parent_2[:index]


def tournament_selection_knapsack(population: list, items:list) -> list:
    """
    Selects the best individual from a random sample of the population.
    """

    parent_1 = rand_choice(population)
    parent_2 = rand_choice(population)
    if get_items_weight_and_value(parent_1, items)[1] > get_items_weight_and_value(parent_2, items)[1]:
        return parent_1

    return parent_2


def mutate_knapsack(state, mutation_rate=0.2):
    """Mutates the current state with a chance of `mutation_rate`"""
    for index in range(len(state)):
        if random() < mutation_rate:
            state[index] = 1 - state[index]


def elitism_merge_knapsack(curr_population, next_population, population_size, items, knapsack_weight_limit):
    """
    merges the current population with the next population and returns the merged population of the two.
    It drops those states in the population whose weight exceed the weight limit of the knapsack.
    """
    merged_population = []
    ptr1 = 0
    ptr2 = 0

    while ptr1 < len(curr_population) and ptr2 < len(next_population) and len(merged_population) < population_size:
        curr_weight, curr_value = get_items_weight_and_value(curr_population[ptr1], items)
        next_weight, next_value = get_items_weight_and_value(next_population[ptr2], items)
        # check for wieght limit exceeded the knapsack limit
        if curr_weight > knapsack_weight_limit:
            ptr1 += 1
            continue

        if next_weight > knapsack_weight_limit:
            ptr2 += 1
            continue

        if curr_value < next_value:
            merged_population.append(next_population[ptr2])
            ptr2 += 1

        else:
            merged_population.append(curr_population[ptr1])
            ptr1 += 1

    
    while len(merged_population) < population_size:
        if ptr1 < len(curr_population):
            merged_population.append(curr_population[ptr1]) 
            ptr1 += 1

        elif ptr2 < len(next_population):
            merged_population.append(next_population[ptr2])
            ptr2 += 1

    return merged_population


############################################################################
#                   UTILITY FUNCTIONS FOR TSP ROMANIA PROBLEM              #
############################################################################
def  tsp_file_parser(filename: str):
    """
    Sample file format:
    ```
    -- romaniacity.txt --
    Oradea, Zerind, 71
    Oradea, Sibiu, 151
    Zerind, Arad, 75
    Arad, Timisoara, 118
    Timisoara, Lugoj, 111
    Lugoj, Mehadia, 70
    ...
    returns a tuple of the form `(knapsack_weight_limit, list_of_items)`
    """
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            city_map = []
            for line in lines:
                city1, city2, distance = line.split(",")
                city_map.append((city1.strip(), city2.strip(), int(distance)))

            return city_map

    except FileNotFoundError:
        print(f'❌ Error: File `{filename}` not found')
        exit()



def generate_initial_tsp_state(cities):
    """
    Generates a random initial solution by visiting each city exactly once, following the available roads.
    """
    shuffles_city = cities[:]
    shuffle(shuffles_city)

    return shuffles_city


def print_analytics_tsp(city_graph, initial_state, curr_state, city_map, cities, start_city):

    print("===============================================================\n")
    __format_final_path(cities, start_city)
    print(f'{len(cities)} Cities to visit:- ')
    print(", ".join(cities))
    __format_final_path(initial_state, start_city)
    __format_final_path(curr_state, start_city)

    print("===============================================================\n")
    print("Initial state: \n")
    # print the initial state's representation
    __print_path_taken(city_graph, initial_state, city_map)

    print("===============================================================")
    print("\nFinal state: \n")
    # print the final state's representation
    __print_path_taken(city_graph, curr_state, city_map)
    print("===============================================================")


def __print_path_taken(city_graph : Graph,curr_state: list[str], between_cities_distance: dict[tuple[str, str], int|float]):
    """
    Prints the path taken by the salesman
    """

    total_distance = 0
    total_number_of_cities_crossed = 0
    for index in range(len(curr_state)):
        _expanded_path = uniform_cost_search(city_graph, curr_state[index], curr_state[(index + 1) % len(curr_state)])
        for idx in range(len(_expanded_path)-1):
            print(f"{_expanded_path[idx]} > ", end="")
            total_number_of_cities_crossed += 1

        total_distance += between_cities_distance[(curr_state[index], curr_state[(index + 1) % len(curr_state)])]
    print(curr_state[0])
    print()
    print(f">>> Total distance: {total_distance} miles")
    print(f'>>> Number of cities Passed Through: {total_number_of_cities_crossed} cities')


def __format_final_path(cities: list, start_city: str):
    idx = cities.index(start_city)
    first_part = [cities[index] for index in range(idx)]

    for _ in range(idx):
        cities.pop(0)
    
    for city in first_part:
        cities.append(city)

def generate_best_immediate_neighbour_tsp(curr_state, city_map):
    # swap 2 cities and return the best neighbour
    best_neighbour = curr_state
    best_neighbour_distance = get_fitness_tsp(curr_state, city_map)

    for index1, _ in enumerate(curr_state):
        for index2 in range(index1 + 1, len(curr_state)):
            neighbour = curr_state[:]
            neighbour[index1], neighbour[index2] = neighbour[index2], neighbour[index1]
            neighbour_distance = get_fitness_tsp(neighbour, city_map)
            if neighbour_distance < best_neighbour_distance:
                best_neighbour = neighbour
                best_neighbour_distance = neighbour_distance

    return best_neighbour


def generate_random_neighbor_tsp(curr_state):
    neighbor = curr_state[:]

    for index1, _ in enumerate(curr_state):
        for index2 in range(index1 + 1, len(curr_state)):
            if random() > 0.5:
                neighbor[index1], neighbor[index2] = neighbor[index2], neighbor[index1]
    
    return neighbor



def get_fitness_tsp(curr_state, between_cities_distance):
    """
    Returns the total distance covered by the salesman following path `curr_state`
    """
    total_distance = 0

    for index in range(len(curr_state)):
        total_distance += between_cities_distance[(curr_state[index], curr_state[(index + 1) % len(curr_state)])]

    return total_distance


def uniform_cost_search(graph: Graph, start: str, goal:str) -> list[str]:
    """
    UCS: `Uniform cost first search` for a graph and returns the path between the start\
    and goal. Returns `empty list` if there is no valid path. Explores all paths.
    """
    heap : list[tuple[int, str, list[str]]] = [(0, start, [])]
    explored = set()

    while heap:
        # get the least cost path so far.
        (curr_cost, node, path) = heapq.heappop(heap)

        if node not in explored:
            explored.add(node)
            path = path + [node]

            if node == goal:
                return path

            neighbors = graph.get_node(node).get_neighbors()

            for neighbor in neighbors:
                if neighbor.name not in explored:
                    cost_of_path = curr_cost + graph.get_node(node).get_weight(neighbor)
                    heapq.heappush(heap, (cost_of_path, neighbor.name, path.copy())) # type: ignore

    return []


def tournament_selection_tsp(curr_population, between_cities_distance):
    parent1 = rand_choice(curr_population)
    parent2 = rand_choice(curr_population)

    if get_fitness_tsp(parent1, between_cities_distance) > get_fitness_tsp(parent1, between_cities_distance):
        return parent2
    
    return parent1


def mutate_tsp(child, MUTATION_RATE):
    """randomly selects two cities in the tour and swaps their positions"""

    if random() < MUTATION_RATE:
        index1 = randint(0, len(child) - 1)
        index2 = randint(0, len(child) - 1)
        child[index1], child[index2] = child[index2], child[index1]


def crossover_tsp(parent_1:list[str], parent_2:list[str], CROSSOVER_RATE):
    """ Crossover operator for TSP. """

    if random() > CROSSOVER_RATE:
        return parent_1, parent_2
    
    # select a random subset of cities from one parent and maintains their order in the offspring
    index1 = randint(0, len(parent_1) - 1)
    index2 = randint(0, len(parent_1) - 1)

    if index1 > index2:
        index1, index2 = index2, index1

    child1 = parent_1[index1:index2]
    child2 = parent_2[index1:index2]

    # Then, it fills the remaining positions in the offspring with cities from the other parent,
    # respecting the order of appearance and excluding duplicates.
    for city in parent_2:
        if city not in child1:
            child1.append(city)

    for city in parent_1:
        if city not in child2:
            child2.append(city)

    return child1, child2


def elitism_merge_tsp(curr_population, next_population, population_size, between_cities_distance):
    """Merges the current population with the next population and returns the merged population of 
    size `population_size` sorted by the total distance covered. The best state is at index 0.
    """
    merged_population = []
    ptr1 = 0
    ptr2 = 0

    for _ in range(population_size):
        if get_fitness_tsp(curr_population[ptr1], between_cities_distance) < get_fitness_tsp(next_population[ptr2], between_cities_distance):
            merged_population.append(curr_population[ptr1])
            ptr1 += 1

        else:
            merged_population.append(next_population[ptr2])
            ptr2 += 1

    return merged_population


def get_path_cost(graph: Graph, path: list[str]) -> int|float:
    """Returns the cost of the path followed."""

    return sum([graph.get_cost(path[index], path[index+1]) \
                for index in range(len(path)-1)])
