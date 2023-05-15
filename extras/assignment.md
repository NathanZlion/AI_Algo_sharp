Assignment II
---

---
# 1. PEAS (30 pts)

    a. Let each member of the group Identify an AI system and characterize it based on the PEAS formulation. For each system, have a single-paragraph description. Then compare them based on their PEAS specification.  (Read Part I, Chapter 1, and Chapter 2).
    b. Come up with project ideas. Each student should come up with an idea. Have a paragraph that describes an idea. Don't worry about the difficulty or whether you will be covering the topic. Based on the ideas, I will be brainstorming with you and you will implement one of them with feasible specifications.

`Important:` The systems in (a) are just existing commercial or any system that you know. You will have to browse the internet to learn more about them. The ideas in (b) are the system you want to implement.

---
# 2. Knapsack (30 pts)

    The knapsack problem is a problem in combinatorial optimization: Given a set of items, each with a weight and a value, determine the number of each item included in a collection so that the total weight is less than or equal to a given limit and the total value is as large as possible.

    Write three algorithms (Genetic Algorithm, Hill Climbing, and Simulated Annealing) to solve the knapsack problem. Your algorithm should take a file on the command line in the following fashion:
	
```
python knapsack.py --algorithm ga --file my-file.txt
```

    The input file should have content in the following style
```
    50
    Item,weight,value, n_items
    Phone,0.19,1000, 5
    Laptop,1.1,700, 2
```

    The first line in the content is the maximum weight in kilograms that your knapsack can handle. The second line is the headers of the succeeding lines and your algorithm should ignore it. The third and onwards should have a comma-separated list of an item’s name, its weight in kilogram, and the item's value in USD. The list should contain 10, 15, and 20 items. It might be tiresome to write 20 items, hence, write some randomized program that generates such a list for you. 

# 3. The traveling salesperson problem (TSP) (30 pts)


    Write three algorithms (Genetic Algorithm, Hill Climbing, and Simulated Annealing) to solve the TSP problem. Your algorithm should take a file on the command line in the following fashion:

```	
	python tsp.py --algorithm ga --file my-file.txt
```
    The input file should contain the Romanian city list that you used in Assignment I.

**Note**: Depending on the team size, distribute the tasks in questions 2 and 3 evenly. In the Question 1 case, one student should work on one idea.

`
Your deliverables should be the code for each question and a report. The report should have a brief explanation of the benchmark of the speed of your algorithms compared to each other and their performance compared with each other. In the first question case, compare your algorithms in 10, 15, and 20 list cases. In question 2 case, compare your algorithms based on 8, 16, and 20 cities cases. It's fine to randomly generate data for the first case. Finally, briefly discuss what you observed. You can use a table or a graph to present your findings.`
