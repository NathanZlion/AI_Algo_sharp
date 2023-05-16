
import sys
from matplotlib import pyplot as plt
from knapsack_solver import Knapsack
from tsp_solver import Tsp_problem
from util import print_analytics_knapsack, print_analytics_tsp
from time import perf_counter

def main():
    # Get the command line arguments
    arguments = sys.argv[1:]

    if "help" in arguments or "-h" in arguments or "--help" in arguments:
        show_error_message()
        return

    # Check if the required arguments are provided
    if len(arguments) < 2 or ("algorithm" not in arguments and "-all" not in arguments) or "problem" not in arguments:
        show_error_message()
        return

    
    # the problem tsp or knapsack
    problem_index = arguments.index("problem") + 1
    problem = arguments[problem_index]

    # Parse the arguments
    algorithm_index = arguments.index("algorithm") + 1 if "algorithm" in arguments else -1
    algorithm = arguments[algorithm_index] if algorithm_index != -1 else None

    file_index = arguments.index("--file") + 1 if "--file" in arguments else -1
    file_path = arguments[file_index] if file_index != -1 else None

    experiment_mode = "--experiment" in arguments
    verbose = "--verbose" in arguments

    __generation_index = arguments.index("--num-of-generations") + 1 if "--num-of-generations" in arguments else -1
    generations = int(arguments[__generation_index]) if __generation_index != -1 else None

    __number_items_index = arguments.index("--num-of-items") + 1 if "--num-of-items" in arguments else -1
    number_of_items = int(arguments[__number_items_index]) if __number_items_index != -1 else None

    __population_size_index = arguments.index("--population-size") + 1 if "--population-size" in arguments else -1
    population_size = int(arguments[__population_size_index]) if __population_size_index != -1 else None

    __mutation_rate_index = arguments.index("--mutation-rate") + 1 if "--mutation-rate" in arguments else -1
    mutation_rate = float(arguments[__mutation_rate_index]) if __mutation_rate_index != -1 else None

    __temperature_index = arguments.index("--initial-temperature") + 1 if "--initial-temperature" in arguments else -1
    temperature = float(arguments[__temperature_index]) if __temperature_index != -1 else None

    __cooling_rate_index = arguments.index("--cooling-rate") + 1 if "--cooling-rate" in arguments else -1
    cooling_rate = float(arguments[__cooling_rate_index]) if __cooling_rate_index != -1 else None

    __crossover_rate_index = arguments.index("--crossover-rate") + 1 if "--crossover-rate" in arguments else -1
    crossover_rate = float(arguments[__crossover_rate_index]) if __crossover_rate_index != -1 else None


    # checking for file sanity
    if file_path:
        # check if the file exists
        try:
            with open(file_path) as _:
                pass
        except FileNotFoundError:
            # if file doesnt exist go with the default file
            if problem == "knapsack":
                print(f"❌ Error: File `{file_path}` not found. Using default file `knapsackdata.txt`")
                file_path = "knapsackdata.txt"
            elif problem == "tsp":
                print(f"❌ Error: File `{file_path}` not found. Using default file `tspdata.txt`")
                file_path = "tspdata.txt"

    else:
        if problem == "knapsack":
            print(f"❌ Error: File not provided. Using default file `knapsackdata.txt` for knapsack problem." )
            file_path = "knapsackdata.txt"
        elif problem == "tsp":
            print(f"❌ Error: File not provided. Using default file `tspdata.txt`" )
            file_path = "tspdata.txt"


    problem_class = Knapsack(file_path, number_of_items= number_of_items) if problem == "knapsack" else Tsp_problem(file_path, number_of_items= number_of_items)

    if not experiment_mode:
        if problem == "knapsack":
            # Call the appropriate algorithm
            if algorithm == "ga":
                initial_population, best_state = problem_class.genetic_algorithm(population_size= population_size, max_generations= generations, \
                                                crossover_rate= crossover_rate, mutation_rate= mutation_rate, verbose= verbose)
                print_analytics_knapsack(initial_population, best_state, problem_class.items, problem_class.knapsack_weight_limit) # type: ignore

            elif algorithm == "sa":
                problem_class.simulated_annealing(initial_temperature = temperature, cooling_rate= cooling_rate, verbose= verbose) 
                print_analytics_knapsack(problem_class.initial_state, problem_class.best_state, problem_class.items, problem_class.knapsack_weight_limit) # type: ignore

            elif algorithm == "hc":
                problem_class.hillclimbing(verbose= verbose)
                print_analytics_knapsack(problem_class.initial_state, problem_class.curr_state, problem_class.items, problem_class.knapsack_weight_limit) # type: ignore

            else:
                print(f"❌ Error: Unknown algorithm `{algorithm}`")
                show_error_message()
                return

        elif problem == "tsp":
            # Call the appropriate algorithm
            if algorithm == "ga":
                initial_state, best_state = problem_class.genetic_algorithm(population_size= population_size, max_generations= generations, \
                                                crossover_rate= crossover_rate, mutation_rate= mutation_rate, verbose= verbose)
                print_analytics_tsp(problem_class.city_graph, initial_state, best_state, problem_class.between_cities_distance, problem_class.cities, problem_class.start_city)     # type: ignore

            elif algorithm == "sa":
                problem_class.simulated_annealing(initial_temperature= temperature, cooling_rate= cooling_rate, verbose= verbose) 
                print_analytics_tsp(problem_class.city_graph, problem_class.initial_state, problem_class.best_state, problem_class.between_cities_distance, problem_class.cities, problem_class.start_city) # type: ignore

            elif algorithm == "hc":
                problem_class.hillclimbing(verbose= verbose)
                print_analytics_tsp(problem_class.city_graph, problem_class.curr_state, problem_class.best_state, problem_class.between_cities_distance, problem_class.cities, problem_class.start_city) # type: ignore

            else:
                print(f"❌ Error: Unknown algorithm `{algorithm}` for problem {problem}")
                show_error_message()
                return

    # for experimental mode
    else:
        if problem == "knapsack":
            test_cases = [10, 15, 20]

            if algorithm == "hc":
                run_hillclimbing(problem_class, verbose, test_cases)

            elif algorithm == "sa":
                run_simulated_annealing(problem_class, test_cases = test_cases, verbose = verbose, initial_temperature=temperature, cooling_rate=cooling_rate)

            elif algorithm == "ga":
                run_genetic_algorithm(problem_class, verbose = verbose, test_cases = test_cases, max_generations=generations, mutation_rate=mutation_rate, crossover_rate=crossover_rate, population_size=None)
            
            else:
                run_hillclimbing(problem_class, verbose, test_cases)
                run_simulated_annealing(problem_class, test_cases = test_cases, verbose = verbose, initial_temperature=temperature, cooling_rate=cooling_rate)
                run_genetic_algorithm(problem_class, verbose = verbose, test_cases = test_cases, max_generations=generations, mutation_rate=mutation_rate, crossover_rate=crossover_rate, population_size=None)

        elif problem == "tsp":
            test_cases = [8, 16, 20]

            if algorithm == "hc":
                run_hillclimbing(problem_class, verbose, test_cases)

            elif algorithm == "sa":
                run_simulated_annealing(problem_class, test_cases = test_cases, verbose = verbose, initial_temperature=temperature, cooling_rate=cooling_rate)

            elif algorithm == "ga":
                run_genetic_algorithm(problem_class, verbose = verbose, test_cases = test_cases, max_generations=generations, mutation_rate=mutation_rate, crossover_rate=crossover_rate, population_size=None)

            else:
                run_hillclimbing(problem_class, verbose, test_cases)
                run_simulated_annealing(problem_class, test_cases = test_cases, verbose = verbose, initial_temperature=temperature, cooling_rate=cooling_rate)
                run_genetic_algorithm(problem_class, verbose = verbose, test_cases = test_cases, max_generations=generations, mutation_rate=mutation_rate, crossover_rate=crossover_rate, population_size=None)
            exit()



def run_hillclimbing(problem_class, verbose, test_cases):
    NUMBER_OF_TESTS = 5
    number_of_items = []
    runtime = []
    fitness_value = []

    for test_case in test_cases:
        problem_class.number_of_items = test_case
        print(f"Running Hill Climbing with {test_case} items : ")

        performance = []
        fitness = []

        for _ in range(NUMBER_OF_TESTS):
            start = perf_counter()
            problem_class.hillclimbing(verbose = verbose)
            end = perf_counter()
            performance.append(end - start)
            fitness.append(problem_class.best_state_fitness)
            problem_class.clear()
        
        print(" ")
        number_of_items.append(test_case)
        runtime.append(sum(performance) / len(performance))
        fitness_value.append(sum(fitness) / len(fitness))

    plot_graph(number_of_items, runtime, fitness_value, "Hill Climbing")

    

def run_simulated_annealing(problem_class, test_cases, initial_temperature, cooling_rate, verbose):
    NUMBER_OF_TESTS = 5
    print(f"---This might take a while, running each experiment {NUMBER_OF_TESTS} times and averaging ---")
    number_of_items = []
    runtime = []
    fitness_value = []

    for test_case in test_cases:
        problem_class.number_of_items = test_case
        print(f"Running Simulated Annealing with {test_case} items : ")

        performance = []
        fitness = []

        for _ in range(NUMBER_OF_TESTS):
            start = perf_counter()
            problem_class.simulated_annealing(initial_temperature = initial_temperature, cooling_rate = cooling_rate, verbose = verbose)
            end = perf_counter()
            performance.append(end - start)
            fitness.append(problem_class.best_state_fitness)
            problem_class.clear()
        
        number_of_items.append(test_case)
        runtime.append(sum(performance) / len(performance))
        fitness_value.append(sum(fitness) / len(fitness))

    plot_graph(number_of_items, runtime, fitness_value, "Simulated Annealing")

# population_size= None, max_generations= None, crossover_rate= None, mutation_rate= None, verbose = False
def run_genetic_algorithm(problem_class, population_size, max_generations, crossover_rate, mutation_rate, verbose, test_cases):
    NUMBER_OF_TESTS = 5
    print(f"---This might take a while, running each experiment {NUMBER_OF_TESTS} times and averaging---")
    number_of_items = []
    runtime = []
    fitness_value = []

    for test_case in test_cases:
        problem_class.number_of_items = test_case
        print(f"Running Genetic Algorithm problem with {test_case} items")

        performance = []
        fitness = []

        for _ in range(NUMBER_OF_TESTS):
            start = perf_counter()
            problem_class.genetic_algorithm(population_size = population_size, max_generations = max_generations,\
                                            crossover_rate = crossover_rate, mutation_rate = mutation_rate, verbose = verbose)
            end = perf_counter()
            performance.append(end - start)
            fitness.append(problem_class.best_state_fitness)
            problem_class.clear()
        

        number_of_items.append(test_case)
        runtime.append(sum(performance) / len(performance))
        fitness_value.append(sum(fitness) / len(fitness))

    plot_graph(number_of_items, runtime, fitness_value, "Genetic Algorithm")

    
def show_error_message():
    message = """Usage:
    python runner.py problem <problem_name> algorithm <algorithm_name> [OPTIONS]

Options:
    --file <file_path>                  Specify the file path for the problem (default: knapsackdata.txt for knapsack, tspdata.txt for tsp)
    --num-of-items <number_of_items>    Specify the number of items (default: 10 for knapsack, 16 for tsp)
    --experiment                        Display a graph of the runtime comparision 
    --verbose                           Enable verbose mode for detailed output
    --num-of-generations <num_generations>       Set the number of generations for genetic algorithm(default: 1000)
    --population-size <population_size> Set the population size for genetic algorithm (default: 100)
    --mutation-rate <mutation_rate>     Set the mutation rate for genetic algorithm (default: 0.1)
    --crossover-rate <crossover_rate>   Set the crossover rate for genetic algorithm (default: 0.8)
    --initial-temperature <initial_temperature> Set the initial temperature for simulated annealing (default: 100.0)
    --cooling-rate <cooling_rate>       Set the cooling rate for simulated annealing (default: 0.95)
    --help                              See the usage of the algorithm
    """
    
    print(message)


def plot_graph(number_of_items, runtime, fitness_value, algorithm):
    plt.figure(figsize=(10, 6))
    plt.bar(number_of_items, runtime)

    # Add labels and runtime on top of each bar
    for i in range(len(number_of_items)):
        plt.text(number_of_items[i], runtime[i], f'Runtime: {runtime[i]*1000:.2f}ms', ha='center', va='top')
        plt.text(number_of_items[i], runtime[i], f'Fitness :  {fitness_value[i]:.2f}', ha='center', va='bottom')

    # Set the x-axis and y-axis labels
    plt.xlabel('Number of Items')
    plt.ylabel('Runtime (seconds)')

    # Set the title
    plt.title(f'{algorithm} Runtime vs Number of Items')

    # Display the plot
    plt.show()


if __name__ == "__main__":
    main()
