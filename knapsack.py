import sys
from knapsack_solver import Knapsack

def main():
    """handles the command line arguments and calls the appropriate algorithm made for knapsack"""
    algorithm = None
    file = None

    # Check if the user has provided the correct number of arguments
    if len(sys.argv) != 5:
        print(f'❌ Error: Wrong command \n Usage :  `py knapsack.py --algorithm ga --file testfile.txt`')
        exit()

    if sys.argv[1].startswith("--algorithm"):
        algorithm = sys.argv[2]
        file = sys.argv[4]

    elif sys.argv[1].startswith("--file"):
        algorithm = sys.argv[4]
        file = sys.argv[2]

    else:
        print("Usage :  py knapsack.py --algorithm ga --file testfile.txt")
        exit()
    
    knapsack_ = Knapsack(file)

    if algorithm == "ga":
        knapsack_.genetic_algorithm()
    
    elif algorithm == "hc":
        knapsack_.hillclimbing()

    elif algorithm == "sa":
        knapsack_.simulated_annealing()

    else:
        print(f'❌ Error: unknown algorithm `{algorithm}`')
        exit()

if __name__ == "__main__":
    main()
