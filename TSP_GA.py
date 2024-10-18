import math
import random
import time
import queue
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import streamlit as st



""" TERMS AND THINGS TO KNOW:
    * individual = path
    * population = paths
    * child = new_path
    * chromosome = path (city_order)
    * gene = a city (in a path)
    * path = all cities once + start_city = end_city
    * Possible solutions = 100!
    * GA SMALL POP = [50 - 100]
    * GA LARGE POP = [200 - 500]
    * MUTATION in nature: [0.01 - 0.1%]
    * LOW / HIGH MUTATION in GA: [0.1 - 1%] / [1 - 5%]
    * LOW / HIGH MUTATION in TSP: [0.5 - 1%] / [1 - 5%]
    * ... or LATER / EARLIER generations
    * CROSSOVER in nature: [100%]
    * LOW / HIGH CROSSOVER in GA: [50 - 70%] / [70 - 100%]
    * LOW / HIGH CROSSOVER in TSP: [60 - 75%] / [80 - 95%]
    * ... or LATER / EARLIER generations
    * ... HIGHER X = preferred in TSP
    * MAX GENERATIONS for TSP: [200 - 1000]
"""


    
class TSPSolver_GA:

    """ CLASS CONSTRUCTOR: 
        * Requires .tsp file and has an option to provide a particular starting 
          city from which to find the shortest path. Optional parameter, start_city. If a start_city is not supplied by the user, a default value is used. This default value represents the first coordinate in the provided .tsp file. A goal_city can also be provided as the end goal or a default value has been supplied. Lastly, an optional assist parameter is available, defaulting to True. When this has not been changed to False, the constructor automatically calls run_BFS_algorithm and run_DFS_algorithms which allows for a streamlined and intuitive use of the algorithm, especially when the class is instantiated directly in a driver file.
    """
    def __init__(self, tsp_file, data_set, pop_size=200, max_gen=200, c_prob_high=0.95, m_prob_high=0.05, solution_type='dict', algorithm='GENETIC ALGORITHM', assist=True, run=False):

        # BASIC TSP FILE VARIABLES:
        self.tsp_file = tsp_file
        self.city_coords = {} # dict of city index + city coordinates
        # self.start_city = start_city

        # GENETIC ALGORITHM VARIABLES:
        self.current_population = self.generate_random_pop # dict of chromosomes (paths) + length?
        # Let's try list first as is typical...
        self.next_generation = [] # list of children to replace current population (if fitness score is better otherwise just add them to the population and kill off the worst ones equivalent to the num children added?)
        self.population_size = pop_size # Num potential solution paths in a population
        self.generation_count = 0
        self.max_generations = max_gen
        self.data_set = data_set
        # self.distance_matrix = []

        # CROSSOVER
        self.cross_prob_HIGH = c_prob_high 
        self.cross_prob_LOW = c_prob_high * 0.75
        

        # MUTATION
        self.mutate_prob_HIGH = m_prob_high
        self.mutate_prob_LOW = m_prob_high * 0.2


        self.data_sets = {
            'D1_single_swap': (self.single_pt_crossover, self.swap_mutation), 
            'D2_single_invert': (self.single_pt_crossover, self.inversion_mutation), 
            'D3_order_swap': (self.order_crossover, self.swap_mutation), 
            'D4_order_invert': (self.order_crossover, self.inversion_mutation) 
        }


        # USED FOR PRINTING RESULTS:
        self.solution_type = solution_type # LIST of DICTS: {city_coords}
        self.algorithm = algorithm # FUTURE: Used to toggle algorithms?
        


        if assist:
            self.parse_tsp_file() # FUTURE: Add end-to-end performance measure?
        if run:
            self.run_algorithm()


    """ PARSE_TSP_FILE:
        * This instance method opens a .tsp file (provided by the user as an 
          argument to the class constructor), reads each line and places  
          them into a list of strings. 
        * The parser then looks for the title to the section holding the 
          coordinates of each city. Once the title is found, the title line is skipped and the next line, which holds the first set of coordinates for a particular city, is broken up into 3 parts: city index, x-coord and y-coord. These are then used to build the instance dictionary, city_coords.
        * Each city's index and corresponding coordinates are added iteratively to the city_coords 
          dictionary as a key / value pair, 'CITY_INDEX': (x, y), until the 'EOF', end of file, is reached.
    """
    def parse_tsp_file(self): 
        # print("PARSE_TSP_FILE")
        # OPEN THE TSP FILE:
        with open(self.tsp_file, 'r') as file:
            lines = file.readlines() # returns a list of strings from .tsp file
            section_found = False
        # LOOK FOR THE NODE COORD SECTION: 
        for line in lines:
            if line.startswith("NODE_COORD_SECTION"):
                section_found = True
                # ONCE SECTION FOUND, JUMP TO NEXT LINE
                continue
            # ONCE THE END OF FILE REACHED BREAK OUT OF LOOP 
            # - (RETURN CONTROL TO NEXT LINE IN MAIN FUNCTION SCOPE)
            if line.startswith("EOF"): # not needed??
                # print("Break statement reached") 
                break
            # ONCE SECTION FOUND (AFTER SKIP) - BREAK THE LINE DOWN INTO PARTS
            if section_found:
                line_parts = line.strip().split() # split string / remove white space
                # CREATE A DICTIONARY: 'CITY_INDEX': (x, y) <-- coordinated tuple = value
                city_index = int(line_parts[0])
                x = float(line_parts[1])
                y = float(line_parts[2])
                self.city_coords[city_index] = (x, y)
        # self.print_this('dict', self.city_coords)

    """ _EUCLIDEAN_DISTANCE: 
        * Helper method used to calculate and return the distance between 2 
          cities, using the euclidean formula.
    """
    def _euclidean_distance(self, city1, city2):
        # print("_EUCLIDEAN_DISTANCE")
        # print(city1, city2)
        return math.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)

        return total_distance
    
    
    """ CALC_TOTAL_DISTANCE:
    This method takes a path as an argument and calculates the total distance of the entire path using the euclidean distance formula. A for loop using the range and length functions iterates through each ith element in the path provided. The coordinates of each consecutive pair of cities in the path are obtained from the city coordinates dictionary built during data parsing. The euclidean distance is calculated between the two consecutive cities and added to the local total_distance variable. Once the last 2 cities of the path have been reached, the method returns the total distance of the path.
    """
    def calc_total_distance(self, path):
        # print("CALC_TOTAL_DISTANCE")
        total_distance = 0
        for i in range(len(path) - 1):
            city1_coords = self.city_coords[path[i]]
            city2_coords = self.city_coords[path[i + 1]]
            distance = self._euclidean_distance(city1_coords, city2_coords)
            total_distance += distance
        return total_distance
    

    
    """ RANDOM_POP_INITIALIZER: 
        Generate initial population of paths.
        PYDOCS: random.sample() = return k length list of elements chosen from the population sequence. Used for random sampling without replacement.
    
    """
    def generate_random_pop(self):
        # print("RANDOM_POP_INITIALIZER")
        city_indices = list(self.city_coords.keys())
        for _ in range(self.population_size): # _ = "throwaway var", only used to maintain syntax
            route = random.sample(city_indices, len(city_indices)) # POP=city_indices, k=len 
            circuit = route + [route[0]] # Outer brackets: list of 1 element for concatenation
            self.current_population.append(circuit)     
        # print(len(self.current_population))       
        return self.current_population # variable assignment convenience if needed

    
    """ FITNESS_FUNCTION:
        Computes the fitness score of a single individual (chromosome) to evaluate its quality as a solution.
        TSP = the shorter the path the higher the fitness score

    """
    def calc_fitness_score(self, path):
        # print("calc_fitness_score")
        # print(path)

        # path_length = 100
        path_length = self.calc_total_distance(path)
        # print("PATH LENGTH", path_length)
        # print("INVERTED PATH LENGTH", 1 / path_length)
        fitness_score1 = 10000000 * (1 / path_length) # Scale floating pt value
        # fitness_score2 = 1000 * (1 / (path_length + 1e-6)) # Scale + avoid division by zero
        # fitness_score3 = 1000 * (1/ (path_length ** 2)) # Exaggerate diff btwn high / low scores.
        # ... [3] stronger selection pressure, speeds up convergence, may converge prematurely.
        # print("SCALED FITNESS SCORE1", fitness_score1)
        # print("SCALED FITNESS SCORE2", fitness_score2)
        # print("SCALED FITNESS SCORE3", fitness_score3)
        # print(fitness_score1)
        return fitness_score1

    def _convert_to_probabilities(self, scores_list):
        # print("_convert_to_probabilities")
        # FIND THE SUM OF ALL SCORES:
        total_scores = sum(scores_list)
        # print(total_scores)
        # print("TOTAL SCORES", total_scores)
        # print("SCORES LIST", scores_list)
        probabilities = []
        # print(total_scores)
        # for score in scores_list:
        #     probability = score / total_scores
        #     probabilities.append(probability)
        #     print(probability)

        # NORMALIZE WITH LIST COMPREHENSION:
        probabilities = [score / total_scores for score in scores_list]
        # print(sum(probabilities))
        return probabilities


    """ RANDOM_SELECTION:
        Applies the fitness function, stores individuals and scores locally in a data structure such as a list of tuples, computes selection probabilities, and randomly selects an individual with a bias towards higher fitness scores. 
    
    """
    def rand_parent_select(self):
        # print("rand_parent_select")
        fitness_scores = []
        f_probabilities = []
        for path in self.current_population:
            f_score = self.calc_fitness_score(path)
            fitness_scores.append(f_score)
            # print("F_SCORE:", f_score)
        # CONVERT FITNESS SCORES TO PROBABILITIES:
        f_probabilities = self._convert_to_probabilities(fitness_scores)

        # curr_pop_index = list(range(len(self.current_population)))
        # print(curr_pop_index)
        # SELECT PARENTS          
        p1_index, p2_index = np.random.choice(len(self.current_population), 2, replace=False,  p=f_probabilities)
        # print("P1", p1_index, f"{(f_probabilities[p1_index] * 100):.3f}")
        # print("P2", p2_index, f"{(f_probabilities[p2_index] * 100):.3f}")
        # print("P1 INDEX", p1_index)
        # print("POP SIZE", len(self.current_population))
        # print("LENGTH PROBS", len(f_probabilities))
        # print(self.current_population)
        parent1 = self.current_population[p1_index]
        parent2 = self.current_population[p2_index]
        # parent2 = parent1
        # # MAKE SURE PARENT1 AND PARENT2 ARE NOT THE SAME INDIVIDUAL:
        # while parent2 == parent1:
        #     p2_index = np.random.choice(len(self.current_population), p=f_probabilities)
            
        return parent1, parent2


    

    
    """ SINGLE_PT_CROSSOVER:
    parent = ordered list of cities
    """
    def single_pt_crossover(self, parent1, parent2):
        # print("SINGLE_PT_CROSSOVER")
        p1 = parent1
        p2 = parent2
        p_size = len(parent1)

        # RAND SELECT SINGLE CROSSOVER POINT:
        crossover_point = random.randint(1, p_size-1)

        # COMBINE P1 / P2 USING LIST COMPREHENSIONS:
        child1 = p1[:crossover_point] + [gene for gene in p2 if gene not in p1[:crossover_point]]
        # IN WORDS: Add the cities/genes from p1 up to but not including the cross_pt as the first
        # ... portion of child1 and then iterate through all cities of p2 and add only cities that 
        # ... are unique to the first portion of child1 (taken from p1 up to cp)
        child2 = p2[:crossover_point] + [gene for gene in p1 if gene not in p2[:crossover_point]]

        return child1, child2
    
    """ ORDER_CROSSOVER:
    
    """   
    def order_crossover(self, parent1, parent2):
        p_size = len(parent1)

        child1, child2 = [None] * p_size, [None] * p_size
        
        cp1, cp2 = sorted(random.sample(range(p_size), 2))

        child1[cp1:cp2] = parent1[cp1:cp2]
        child2[cp1:cp2] = parent2[cp1:cp2]

        child1 = [gene for gene in parent2 if gene not in parent1[cp1:cp2]] + parent1[cp1:cp2] + [gene for gene in parent2 if gene not in child1[:cp2]]

        child2 = [gene for gene in parent1 if gene not in parent2[cp1:cp2]] + parent2[cp1:cp2] + [gene for gene in parent2 if gene not in child2[:cp2]]

        # print(child1)

        # print(child2)
       
        return child1, child2
    
    """ MUTATE1:
        Introduces random mutation to maintain diversity.
    
    """
    def swap_mutation(self, child1, child2):
        # print("SWAP_MUTATION")
        chosen = random.choice([0, 1])
        mutant = child1[:] if chosen == 0 else child2[:] # Syntax ensures shallow copy for mutant
        mutant_indices = list(range(len(mutant)))
        gene1, gene2 = random.sample(mutant_indices, 2)
        mutant[gene1], mutant[gene2] = mutant[gene2], mutant[gene1]
        if chosen == 0:
            return mutant, child2
        else:
            return child1, mutant     

            
    def inversion_mutation(self, child1, child2):
        # print("INVERSION MUTATION")
        chosen = random.choice([0, 1])        
        mutant = child1[:] if chosen == 0 else child2[:] # Syntax ensures deep copy for mutant
        mutant_indices = list(range(len(mutant)))
        cp1, cp2 = sorted(random.sample(mutant_indices, 2))
        mutant[cp1 : cp2] = mutant[cp1 : cp2][::-1]
        if chosen == 0:
            return mutant, child2
        else:
            return child1, mutant
   
    


    """ REPRODUCE:
        Combines two parents to produce a child through crossover.

    """
    def reproduce(self, parent1, parent2):
        # print("------REPRODUCE-----")
        p1 = parent1
        p2 = parent2
        crossover, mutation = self.data_sets[self.data_set]
        # print("DATA SET", self.data_sets[self.data_set])

        if (self.generation_count < (self.max_generations * 0.5)) and (random.random() <= self.cross_prob_HIGH):
            # print("[[CROSSOVER = HIGH]]")
            child1, child2 = crossover(p1, p2)
        elif random.random() <= self.cross_prob_LOW:
            # print("[[CROSSOVER = LOW]]")
            child1, child2 = crossover(p1, p2)
        else:
            # print("NO CROSSOVER")
            child1, child2 = p1[:], p2[:]

        # APPLY MUTATION ON THE CHILDREN PRODUCED ABOVE:    
        if (self.generation_count < (self.max_generations * 0.5)) and (random.random() <= self.mutate_prob_HIGH):
            # print("[[MUTATION = HIGH]]")
            child1, child2 = mutation(p1, p2)
        elif random.random() <= self.mutate_prob_LOW:
            # print("[[MUTATION = LOW]]")
            child1, child2 = mutation(p1, p2)
        # else:
        #     print("NO MUTATION")
  
        return child1, child2



    # def population_replacement(self, child1, child2):
    #     partial_replace = ''
    #     elite_replace = ''
    #     full_replace = ''
    


    """ GENETIC_ALGORITHM:
        Manages the entire process of selection, reproduction, mutation, and population update until the termination criteria are met.
    
    """
    def genetic_algorithm(self):
        # print("GENETIC_ALGORITHM")
        # print("DATA SET PASSED", data_set)
        # [1] 
        self.current_population = []
        children = []
        next_generation = []
        # final_fit_score = 0
        # final_path = []
        
        # print("CURR_POP LENGTH", len(self.current_population))
        # pop_size = len(self.current_population)
        
        # while self.generation_count != self.max_generations: 
            # next_generation = []
        # num_pairs = math.ceil(self.population_size / 2)                 
    # for _ in range(num_pairs):
        parent1, parent2 = self.rand_parent_select()
        child1, child2 = self.reproduce(parent1, parent2)
            # print(child1, child2)
        # best_child = max([child1, child2], key=self.calc_fitness_score)
            # print("BEST CHILD", best_child)
        children.append(child1)
        children.append(child2)
        self.current_population = self.pick_elite_next_gen(children)
        # ANIMATION - separate this out?
        # print("NXT GEN LEN: ", len(next_generation))
        # print("CURR POP LEN: ", len(self.current_population))
        # self.generation_count += 1
        # next_generation = []
        # children = []
        
        best_path = min(self.current_population, key=self.calc_total_distance)
        # solution_path2 = max(self.current_population, key=self.calc_fitness_score)
        # for path in self.current_population:
        #     f_score = self.calc_fitness_score(path)
        #     if f_score > final_fit_score:
        #         final_fit_score = f_score
        #         p_index = path

        # solution_path = self.current_population[p_index]

        best_distance = self.calc_total_distance(best_path) 



        return best_distance, best_path

    def pick_elite_next_gen(self, children):
        # print("PICK ELITE")
        combined_population = []
        combined_population = self.current_population + children
        # print("CHILDS LEN:", len(children))
        # print(len(combined_population))
        combined_population.sort(key=self.calc_fitness_score, reverse=True)
        next_gen = combined_population[:self.population_size]
        return next_gen
  
    """ RUN_ALGORITHM:
        * Instance method that runs all components of the class that make up 
          the complete algorithm in the manner they are intended to work together. 
        * Additionally, the runtime of the algorithm is calculated.    
    """
    
    def run_algorithm(self):
        fitness_progress = []
        shortest_paths = []

        start_time = time.perf_counter()
        for generation in range(self.max_generations):
            best_distance, best_path = self.genetic_algorithm()
            fitness_progress.append(best_distance)
            shortest_paths.append(best_path)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        solution = {
            "DATA SET" : self.data_set,
            "SOLUTION PATH": best_path,
            "SOLUTION DISTANCE": best_distance,
            "POPULATION SIZE": self.population_size,
            "MAX GENERATIONS": self.max_generations,
            "CROSSOVER RATES HIGH": self.cross_prob_HIGH,
            "MUTATION RATES HIGH": self.mutate_prob_HIGH,
            }
        # self.plot_cities()
        # self.plot_tour(solution['SOLUTION'])
        # results[data_set] = {
        #     "ELAPSED TIME": elapsed_time,
        #     "SOLUTION": solution
        # }
        self._print_results(elapsed_time, solution, self.solution_type, self.algorithm)
    


    def plot_cities(self):
        x_coords = [coord[0] for coord in self.city_coords.values()]
        y_coords = [coord[1] for coord in self.city_coords.values()]

        plt.figure(figsize=(12,8))
        plt.scatter(x_coords, y_coords, color='blue', label='Cities')

        for city_index, (x, y) in self.city_coords.items():
            plt.text(x + 1.5, y + 1.5, str(city_index), fontsize=10, ha='right', va='bottom')

        plt.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.title('TSP Cities')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        plt.grid(True)
        plt.show()

    def plot_tour(self, path):
        x_coords = [coord[0] for coord in self.city_coords.values()]
        y_coords = [coord[1] for coord in self.city_coords.values()]

        plt.figure(figsize=(12,8))
        plt.scatter(x_coords, y_coords, color='blue', label='Cities')

        for city_index, (x, y) in self.city_coords.items():
            plt.text(x + 1.5, y + 1.5, str(city_index), fontsize=10, ha='right', va='bottom')
        
        tour_x = [self.city_coords[city][0] for city in path]
        tour_y = [self.city_coords[city][1] for city in path]

        plt.plot(tour_x, tour_y, color='red', linestyle='-', linewidth=2, marker='o', label='Tour Path', markerfacecolor='blue')
        plt.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.title('TSP Cities')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        plt.grid(True)
        plt.show()
        
    def create_gui():
        root = tk.Tk()
        root.title("TSP Solver - Closest Edge Insertion Heuristic")


    """ _PRINT_RESULTS:
        * Prints all results for the algorithm.
    """
    def _print_results(self, total_time, solution, type_str, algorithm):
        print(f"\n\n\n[{self.tsp_file}] & {algorithm}")
        print("_" * 100)
        print(f"TOTAL RUNTIME: {total_time: .6f} seconds || {total_time / 60: .6f} minutes\n")
        self.print_this(type_str, solution)
        print("_" * 100)

    """ PRINT_THIS:
        * Method used for debugging and printing stuff.
    """
    def print_this(self, type, object):
        if type == 'iter':
            for element in object:
                print(element)
        if type == 'dict':
            for key, value in object.items():
                print(f"{key}: {value}")
        if type == 'enum': 
            for index, value in object:
                print(f"{index}: {value}")
        if type == 'pq':
            temp_pq = queue.PriorityQueue() 
            # ... mutable objects are passed by reference to functions in Python
            elements = []
            while not object.empty():
                item = object.get() # pop off an element
                elements.append(item)
                temp_pq.put(item) # put it back in the queue
            print("PRIORITY QUEUE CONTAINS:", elements)
            while not temp_pq.empty():
                object.put(temp_pq.get())




if __name__ == "__main__":

    
    data_sets = [
        'D1_single_swap', 
        'D2_single_invert', 
        'D3_order_swap', 
        'D4_order_invert', 
    ]

    for data_set in data_sets:
        solve = TSPSolver_GA('Random100.tsp', data_set, pop_size=10, max_gen=10, run=True)
    # solve.parse_tsp_file
    # solve.print_this('dict', solve.city_coords)
    # pop_sizes = [5, 10, 15, 20, 25]
    # for size in pop_sizes:
    #     solve = TSPSolver_GA('Random100.tsp', data_sets[0], pop_size=5, max_gen=10)

    # solve = TSPSolver_GA('Random100.tsp', data_sets[0], pop_size=5, max_gen=10)

    # solve = TSPSolver_GA('Random100.tsp', data_sets[0], pop_size=10, max_gen=10)

    # solve = TSPSolver_GA('Random100.tsp', data_sets[0], pop_size=20, max_gen=20)
    

    # # print(f"------------------------------ {data_set} ---------------------------------------")






    # CODE GRAVEYARD
            # print("ORDER_CROSSOVER")
        # p1 = parent1
        # p2 = parent2
        # p_size = len(parent1)
        # print("P_SIZE", p_size)

        # # RAND SELECT TWO CROSS_PT INDICES IN RANGE OF PARENT SIZE:
        # cp1, cp2 = sorted(random.sample(range(p_size), 2))
        # # ... sort ascending for start : end splicing
        # # ... k = 2 values in range

        # # CHILD1: CREATE WITH EMPTY GENES TO BE FILLED BY PARENTS:
        # child1 = [None] * p_size
        # print("CHILD SIZE", len(child1))
        # # FILL A PORTION OF THE CHILD WITH AN ORDER SPLICE FROM P1:
        # child1[cp1 : cp2] = p1[cp1 : cp2]
        # # FILL THE REST OF THE C1 GENES FROM P2 ORDERED SEGMENTS OF UNIQUE GENES:
        # p_gene = 0
        # for i in range(p_size + 1):
        #     print("i", i)
        #     if child1[i] is None:
        #         print(i, child1[i])
        #         while p2[p_gene] in child1: # skip genes already in c1 (from p1)
        #             print("PGENE", p_gene)
        #             if p_gene < len(child1) - 1:
        #                 p_gene += 1 # skip it
        #             # if p_gene >= p_size:
        #             #     p_gene = 0
        #         child1[i] = p2[p_gene] # add unique p2 genes
        #         if p_gene < len(child1) - 1:
        #             p_gene += 1 # skip i
        
        # # # CHILD2: REPEAT - SWAPPING START PARENT:
        # # child2 = [None] * p_size

        # # child2[cp1 : cp2] = p2[cp1 : cp2]
        # # p_gene = 0
        # # for i in range(p_size):
        # #     if child2[i] is None:
        # #         while p1[p_gene] not in child2:
        # #             child2[i] = p1[p_gene]
        # #             p_gene += 1

        
        # print(child1)
        # return child1

        #         child1[:cp1] = parent2[:cp1]
        

        # child1[:cp1] = p2[] gene for gene in p2 if gene not in p1[cp1:cp2]

        # current_index = 0
        # for gene in parent2:
        #     if gene not in child1:
        #         if current_index >= p_size:
        #             current_index = 0
        #         child1[current_index] = gene
        #         current_index += 1

        # for gene in parent1:
        #     if gene not in child2:
        #         if current_index >= p_size:
        #             current_index = 0
        #         child2[current_index] = gene
        #         current_index += 1


            # def run_algorithm3(self):
    #     print("run_algorithm")

    #     for data_set in self.data_sets.keys():
    #         print(data_set)
    #         start_time = time.perf_counter()
    #         solution = self.genetic_algorithm(data_set)
    #         end_time = time.perf_counter()
    #         elapsed_time = end_time - start_time
    #         # results[data_set] = {
    #         #     "ELAPSED TIME": elapsed_time,
    #         #     "SOLUTION": solution
    #         # }
    #         self._print_results(elapsed_time, solution, self.solution_type, self.algorithm)
    #         # self.plot_cities()
    #         # self.plot_tour(solution['SOLUTION PATH'])
    #     # print(len(solution["SOLUTION PATH"]))
    
    # def run_algorithm2(self, data_set):
    #     start_time = time.perf_counter()
    #     solution = self.genetic_algorithm(data_set)
    #     end_time = time.perf_counter()
    #     elapsed_time = end_time - start_time
    #     # results[data_set] = {
    #     #     "ELAPSED TIME": elapsed_time,
    #     #     "SOLUTION": solution
    #     # }
    #     self._print_results(elapsed_time, solution, self.solution_type, self.algorithm)

        # """ CROSSOVER_SELECT
    # Determines the crossover_point that will be used by REPRODUCE.
    
    # """
    # def crossover_select(self, parent1, parent2):
    #     print("CROSSOVER_SELECT")

    #     crossovers = []
    #     crossover_method = random.choice(crossovers)
    #     return crossover_method(parent1, parent2)
        # # EASIEST TO CODE
        # single_pt = ''
        # uniform = ''

        # # BEST FOR TSP:
        # order = ''
        # edge_recombination = ''

        # # EASIER TO CODE THAN ERX:
        # partial_map = ''

        
    # def create_distance_matrix(self):
    #     city_indices = list(self.city_coords.keys())
    #     size = len(city_indices)
    #     self.distance_matrix = [[0] * size for _ in range(size)]
    #     for i in range(size):
    #         for j in range(size):
    #             if i != j:
    #                 self.distance_matrix[i][j] = self._euclidean_distance(self.city_coords[city_indices[i]], self.city_coords[city_indices[j]])
    
    # def calc_total_distance(self, path):
    #     total_distance = 0
    #     for i in range(len(path) - 1):
    #         total_distance += self.distance_matrix[path[i] - 1][path[i + 1] - 1]