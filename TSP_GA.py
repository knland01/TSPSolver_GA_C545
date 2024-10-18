import math
import random
import time
import queue
import numpy as np


""" THINGS TO KNOW:
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
        * Requires .tsp file and a data set to use. The data set represents which crossover and   
          mutation techniques to use during reproduction. has an option to provide a particular starting. Additional parameters are provided with default values that can also be specified to fine tune the genetic algorithm including: population size, maximum number of generations, the first half (higher) probability of crossover,and the first half (higher) probability of mutation. The solution type and algorithm parameters are for printing results. The assist parameter set to true will automatically parse the tsp file during class construction and when the run parameter is set to true it will auto-run the algorithm during class construction.
    """
    def __init__(self, tsp_file, data_set, pop_size=30, max_gen=30, c_prob_high=0.95, m_prob_high=0.05, solution_type='dict', algorithm='GENETIC ALGORITHM', assist=True, run=False):

        # BASIC TSP FILE VARIABLES:
        self.tsp_file = tsp_file
        self.city_coords = {} # dict of city index + city coordinates

        # GENETIC ALGORITHM VARIABLES:
        self.current_population = []
        self.next_generation = [] 
        self.population_size = pop_size # Num potential solution paths in a population
        self.generation_count = 0
        self.max_generations = max_gen
        self.data_set = data_set

        # CROSSOVER
        self.cross_prob_HIGH = c_prob_high 
        self.cross_prob_LOW = c_prob_high * 0.85
        

        # MUTATION
        self.mutate_prob_HIGH = m_prob_high
        self.mutate_prob_LOW = m_prob_high * 0.75

        self.data_sets = {
            'D1_single_swap': (self.single_pt_crossover, self.swap_mutation), 
            'D2_single_invert': (self.single_pt_crossover, self.inversion_mutation), 
            'D3_order_swap': (self.order_crossover, self.swap_mutation), 
            'D4_order_invert': (self.order_crossover, self.inversion_mutation) 
        }

        # USED FOR PRINTING RESULTS:
        self.solution_type = solution_type 
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
                break
            # ONCE SECTION FOUND (AFTER SKIP) - BREAK THE LINE DOWN INTO PARTS
            if section_found:
                line_parts = line.strip().split() # split string / remove white space
                # CREATE A DICTIONARY: 'CITY_INDEX': (x, y) <-- coordinated tuple = value
                city_index = int(line_parts[0])
                x = float(line_parts[1])
                y = float(line_parts[2])
                self.city_coords[city_index] = (x, y)


    """ _EUCLIDEAN_DISTANCE: 
        * Helper method used to calculate and return the distance between 2 
          cities, using the euclidean formula.
    """
    def _euclidean_distance(self, city1, city2):
        return math.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)
    
    
    """ CALC_TOTAL_DISTANCE:
    This method takes a path as an argument and calculates the total distance of the entire path using the euclidean distance formula. A for loop using the range and length functions iterates through each ith element in the path provided. The coordinates of each consecutive pair of cities in the path are obtained from the city coordinates dictionary built during data parsing. The euclidean distance is calculated between the two consecutive cities and added to the local total_distance variable. Lastly, the distance between the last city and the first city is added to complete the distance of the tour as a circuit. Once the last 2 cities of the path have been reached, the method returns the total distance of the path.
    """
    def calc_total_distance(self, path):
        total_distance = 0
        for i in range(len(path) - 1):
            city1_coords = self.city_coords[path[i]]
            city2_coords = self.city_coords[path[i + 1]]
            distance = self._euclidean_distance(city1_coords, city2_coords)
            total_distance += distance
        # ADD LAST CITY BACK TO FIRST CITY DISTANCE TO COMPLETE CIRCUIT:
        first_city = self.city_coords[path[0]]
        last_city = self.city_coords[path[-1]]
        total_distance += self._euclidean_distance(last_city, first_city)
        return total_distance
    

    
    """ RANDOM_POP_INITIALIZER: 
        Generate initial population of paths.
        PYDOCS: random.sample() = return k length list of elements chosen from the population sequence. Used for random sampling without replacement.
    
    """
    def generate_random_pop(self):
        city_indices = list(self.city_coords.keys())
        for _ in range(self.population_size): # _ = "throwaway var", only used to maintain syntax
            route = random.sample(city_indices, len(city_indices)) # POP=city_indices, k=len 
            self.current_population.append(route) # Last city is implied in TSP GA          
        return self.current_population # variable assignment convenience if needed

    
    """ FITNESS_FUNCTION:
        Computes the fitness score of a single individual (chromosome) to evaluate its quality as a solution. In the case of TSP, the fitness score is the shortest distance. To create a fitness score that favors higher values, the inverse of the distance is created and then scaled for readability. 
    """
    def calc_fitness_score(self, path):
        path_length = self.calc_total_distance(path)
        fitness_score1 = 10000000 * (1 / path_length) # Scale floating pt value
        # fitness_score2 = 1000 * (1 / (path_length + 1e-6)) # Scale + avoid division by zero
        # fitness_score3 = 1000 * (1/ (path_length ** 2)) # Exaggerate diff btwn high / low scores.
        return fitness_score1

    
    """_CONVERT_TO_PROBABILITIES:
        Converts the fitness scores into probabilities so that they can be used to weight individuals within the population for random but biased parent selection to foster the next generation.
    """
    def _convert_to_probabilities(self, scores_list):
        total_scores = sum(scores_list)
        probabilities = []
        # NORMALIZE WITH LIST COMPREHENSION:
        probabilities = [score / total_scores for score in scores_list]
        return probabilities


    """ RAND_PARENT_SELECTION:
        Applies the fitness function, stores individuals and scores locally, calls the _convert_probabilities helper function to convert fitness scores to probabilities, and randomly selects an individual with a bias towards higher fitness scores, using the numpy module. 
    """

    def rand_parent_select(self):
        fitness_scores = []
        f_probabilities = []
        for path in self.current_population:
            f_score = self.calc_fitness_score(path)
            fitness_scores.append(f_score)
        # CONVERT FITNESS SCORES TO PROBABILITIES:
        f_probabilities = self._convert_to_probabilities(fitness_scores)     
        p1_index, p2_index = np.random.choice(len(self.current_population), 2, replace=False,  p=f_probabilities)
        parent1 = self.current_population[p1_index]
        parent2 = self.current_population[p2_index] 
        return parent1, parent2

    
    """ SINGLE_PT_CROSSOVER:
        Crossover technique used in reproduction between two parents and produces two children. A crossover point is randomly selected and then the children are produced. Each child is given the genes (cities) from one parent up to but excluding the crossover point from that parent's chromosome (path) and the genes from the other parent after, but including, the crossover point. The two children (new paths) are returned from the function. 
    """
    def single_pt_crossover(self, parent1, parent2):
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
        Crossover technique used in reproduction between two parents and produces two children. Two children are created with the same number of gene slots as the parents, however the genes at this point are assigned the value None. Two crossoverpoints are randomly selected for the earlier and later crossover points. The genes from the parents are filled into these segments (as determined by the crossover points) in order. The first parent fills in the genes btwn the crossover points and the second parent fills in the genes on the outside of the crossover points. All genes are checked for uniqueness against the genes already placed. Only unique genes are placed.
    """   
    def order_crossover(self, parent1, parent2):
        p_size = len(parent1)
        child1, child2 = [None] * p_size, [None] * p_size
        cp1, cp2 = sorted(random.sample(range(p_size), 2))
        child1[cp1:cp2] = parent1[cp1:cp2]
        child2[cp1:cp2] = parent2[cp1:cp2]
        child1 = [gene for gene in parent2 if gene not in parent1[cp1:cp2]] + parent1[cp1:cp2] + [gene for gene in parent2 if gene not in child1[:cp2]]
        child2 = [gene for gene in parent1 if gene not in parent2[cp1:cp2]] + parent2[cp1:cp2] + [gene for gene in parent2 if gene not in child2[:cp2]]
        return child1, child2
    
    """ SWAP MUTATION:
        Mutation technique applied to children produced by reproduction. If mutation occurs it is only applied to one of the two children produced by two parents. The child chosen is random. Two genes are chosen at random and swapped.
    
    """
    def swap_mutation(self, child1, child2):
        chosen = random.choice([0, 1])
        mutant = child1[:] if chosen == 0 else child2[:] # Syntax ensures shallow copy for mutant
        mutant_indices = list(range(len(mutant)))
        gene1, gene2 = random.sample(mutant_indices, 2)
        mutant[gene1], mutant[gene2] = mutant[gene2], mutant[gene1]
        if chosen == 0:
            return mutant, child2
        else:
            return child1, mutant     

    """ INVERSION_MUTATION:
        Mutation technique applied to children produced by reproduction. If mutation occurs it is only applied to one of the two children produced by two parents. The child chosen is random. Two points are chosen at random and the genes between these points are inverted.
    
    """ 
    def inversion_mutation(self, child1, child2):
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
        Combines two parents to produce a child based on the data set and probabilities issued during class construction / instantiation. The crossover and mutation functions are called accordingly. Higher probabilities for both crossover and mutation are applied to the first half of the max generations and lower probabilities are applied for the second half of the generation cycles.

    """
    def reproduce(self, parent1, parent2):
        p1 = parent1
        p2 = parent2
        crossover, mutation = self.data_sets[self.data_set]
        if (self.generation_count < (self.max_generations * 0.5)) and (random.random() <= self.cross_prob_HIGH):
            child1, child2 = crossover(p1, p2)
        elif random.random() <= self.cross_prob_LOW:
            child1, child2 = crossover(p1, p2)
        else:
            child1, child2 = p1[:], p2[:]
        # APPLY MUTATION ON THE CHILDREN PRODUCED ABOVE:    
        if (self.generation_count < (self.max_generations * 0.5)) and (random.random() <= self.mutate_prob_HIGH):
            child1, child2 = mutation(p1, p2)
        elif random.random() <= self.mutate_prob_LOW:
            child1, child2 = mutation(p1, p2)
        return child1, child2


    """ GENETIC_ALGORITHM:
        Manages the random parent selection, reproduction, the list of children for the next generation, the elite selection of individuals for the next generation, and finds the best path and calculates its distance of the current population.
    
    """
    def genetic_algorithm(self):
        children = []            
        for _ in range(self.population_size):
            parent1, parent2 = self.rand_parent_select()
            child1, child2 = self.reproduce(parent1, parent2)
            children.append(child1)
            children.append(child2)
        self.pick_elite_next_gen(children)
        best_path = min(self.current_population, key=self.calc_total_distance)
        best_distance = self.calc_total_distance(best_path) 
        return best_distance, best_path

    """ PICK_ELITE_NEXT_GEN:
        Combines the newly produced children and the parent population and picks the number of individuals equal to the population size with the highest fitness scores as the elite members of the next generation. (FUTURE: This should be a parameter that can be toggled, adding tournament selection or other as another option to explore)
    """
    def pick_elite_next_gen(self, children):
        combined_population = []
        combined_population = self.current_population + children
        combined_population.sort(key=self.calc_fitness_score, reverse=True)
        self.current_population = combined_population[:self.population_size]
        return self.current_population
  
    """ RUN_ALGORITHM:
        * Instance method that runs all components of the class that make up 
          the complete algorithm in the manner they are intended to work together. 
        * Additionally, the runtime of the algorithm is calculated.    
    """
    
    def run_algorithm(self):
        fitness_progress = []
        shortest_paths = []
        start_time = time.perf_counter()
        self.current_population = self.generate_random_pop()
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
        self._print_results(elapsed_time, solution, self.solution_type, self.algorithm)
    


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
        solve = TSPSolver_GA('Random100.tsp', data_set, pop_size=50, max_gen=50, run=True)




# CODE GRAVEYARD
    # circuit = route + [route[0]] # Outer brackets: list of 1 element for concatenation

    # import matplotlib.pyplot as plt
# import tkinter as tk
# from tkinter import ttk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # def plot_cities(self):
    #     x_coords = [coord[0] for coord in self.city_coords.values()]
    #     y_coords = [coord[1] for coord in self.city_coords.values()]

    #     plt.figure(figsize=(12,8))
    #     plt.scatter(x_coords, y_coords, color='blue', label='Cities')

    #     for city_index, (x, y) in self.city_coords.items():
    #         plt.text(x + 1.5, y + 1.5, str(city_index), fontsize=10, ha='right', va='bottom')

    #     plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    #     plt.title('TSP Cities')
    #     plt.xlabel('X Coordinate')
    #     plt.ylabel('Y Coordinate')

    #     plt.grid(True)
    #     plt.show()

    # def plot_tour(self, path):
    #     x_coords = [coord[0] for coord in self.city_coords.values()]
    #     y_coords = [coord[1] for coord in self.city_coords.values()]

    #     plt.figure(figsize=(12,8))
    #     plt.scatter(x_coords, y_coords, color='blue', label='Cities')

    #     for city_index, (x, y) in self.city_coords.items():
    #         plt.text(x + 1.5, y + 1.5, str(city_index), fontsize=10, ha='right', va='bottom')
        
    #     tour_x = [self.city_coords[city][0] for city in path]
    #     tour_y = [self.city_coords[city][1] for city in path]

    #     plt.plot(tour_x, tour_y, color='red', linestyle='-', linewidth=2, marker='o', label='Tour Path', markerfacecolor='blue')
    #     plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    #     plt.title('TSP Cities')
    #     plt.xlabel('X Coordinate')
    #     plt.ylabel('Y Coordinate')

    #     plt.grid(True)
    #     plt.show()



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

        # solve.parse_tsp_file
    # solve.print_this('dict', solve.city_coords)
    # pop_sizes = [5, 10, 15, 20, 25]
    # for size in pop_sizes:
    #     solve = TSPSolver_GA('Random100.tsp', data_sets[0], pop_size=5, max_gen=10)

    # solve = TSPSolver_GA('Random100.tsp', data_sets[0], pop_size=5, max_gen=10)

    # solve = TSPSolver_GA('Random100.tsp', data_sets[0], pop_size=10, max_gen=10)

    # solve = TSPSolver_GA('Random100.tsp', data_sets[0], pop_size=20, max_gen=20)
    

    # # print(f"------------------------------ {data_set} ---------------------------------------")