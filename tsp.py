

import sys
from tsp_solver import Tsp_problem


### Travelling sales man problem


def main():
    """handles the command line arguments and calls the appropriate algorithm made for TSP"""
    algorithm = None
    file = None

    # Check if the user has provided the correct number of arguments
    if len(sys.argv) != 5:
        print(f'❌ Error: Wrong command \n Usage :  `py tsp.py --algorithm ga --file tspdata.txt`')
        exit()

    if sys.argv[1].startswith("--algorithm"):
        algorithm = sys.argv[2]
        file = sys.argv[4]

    elif sys.argv[1].startswith("--file"):
        algorithm = sys.argv[4]
        file = sys.argv[2]

    else:
        print("Usage :  `py tsp.py --algorithm ga --file tspdata.txt")
        exit()

    TSP = Tsp_problem(file, number_of_cities = 20)

    if algorithm == "ga":
        # Call the genetic algorithm
        TSP.genetic_algorithm()
    
    elif algorithm == "hc":
        TSP.hillclimbing()

    elif algorithm == "sa":
        TSP.simulated_annealing()
    # Otherwise, print an error message
    else:
        print(f'❌ Error: unknown algorithm `{algorithm}`')
        exit()

if __name__ == "__main__":
    main()
