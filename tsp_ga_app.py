import streamlit as st
import matplotlib.pyplot as plt
from TSP_GA import TSPSolver_GA  # Ensure this points to your class file
import time

# Title and Instructions
st.title("TSP Solver Using Genetic Algorithm")
st.write("This app demonstrates the evolution of a solution to the Traveling Salesman Problem using a Genetic Algorithm.")

# Collect GA Parameters from User
tsp_file = st.text_input("Enter TSP file path:", "Random100.tsp")
data_set = st.selectbox("Select Data Set", ["D1_single_swap", "D2_single_invert", "D3_order_swap", "D4_order_invert"])
pop_size = st.slider("Population Size", 5, 500, 5)
max_gen = st.slider("Maximum Generations", 5, 500, 5)
c_prob_high = st.slider("FIRST HALF: Crossover Probability", 0.5, 1.0, 0.95)
st.write(f"SECOND HALF: Crossover Probability = {c_prob_high * 0.75:.2f}")
m_prob_high = st.slider("FIRST HALF: Mutation Probability", 0.01, 0.1, 0.05)
st.write(f"SECOND HALF: Mutation Probability = {m_prob_high * 0.20:.2f}")
solution_type = st.selectbox("Solution Type", ["dict", "list"])
algorithm = st.selectbox("Algorithm", ["GENETIC ALGORITHM", "BRUTE FORCE", "GREEDY: CLOSEST EDGE", "DEPTH FIRST SEARCH", "BREADTH FIRST SEARCH"])
assist = True
# algorithm = st.text_input("Algorithm", "GENETIC ALGORITHM") 
# ... eventually make this a toggle for the different tsp algorithms?
# assist = st.checkbox("Assist", True)

# ANIMATION OPTIONS:
animation_speed = st.slider("Animation Speed (seconds per generation)", 0.1, 2.0, 0.5)
dynamic_plot = st.empty()

# Initialize the TSPSolver_GA instance
solver = TSPSolver_GA(
    tsp_file=tsp_file,
    data_set=data_set,
    pop_size=pop_size,
    max_gen=max_gen,
    c_prob_high=c_prob_high,
    m_prob_high=m_prob_high,
    solution_type=solution_type,
    algorithm=algorithm,
    assist=assist
)

# Run the algorithm and visualize progress
if st.button("Run GA"):
    fitness_progress = []
    shortest_paths = []
    generation = 0

    # Run through each generation and visualize evolution
    for generation in range(solver.max_generations):
        solver.genetic_algorithm()  # Run a single generation
        best_path = min(solver.current_population, key=solver.calc_total_distance)
        best_distance = solver.calc_total_distance(best_path)
        fitness_progress.append(best_distance)
        shortest_paths.append(best_path)

        # Display current generation and best path details
        st.metric("Generation", generation + 1)
        st.metric("Best Distance", f"{best_distance:.2f}")
        
        # NOTE: PLOT WITH MATPLOTLIB
        # [1] Figure object = 
        # [2] Axis object = array of axes
        fig, ax = plt.subplots() 
        # [3] Obtain x, y coordinates from the coordinates dictionary
        tour_x = [solver.city_coords[city][0] for city in best_path]
        tour_y = [solver.city_coords[city][1] for city in best_path]
        # [4] Add x, y coords to the axis object
        # ... 'r-' = draw red line | 'o' = circular markers
        ax.plot(tour_x, tour_y, 'r-', marker='o', label="Best Path")
        ax.set_title(f"Best Path - Generation {generation + 1}")
        ax.legend()
        
        dynamic_plot.pyplot(fig)
        st.pyplot(fig) # static plot
        time.sleep(animation_speed)

    # LINE CHART DISPLAYING FITNESS OF SOLUTION ACROSS GENERATIONS
    # st.line_chart(fitness_progress)
    st.write("### BEST DISTANCE OVER GENERATIONS")
    fig, ax = plt.subplots()
    ax.plot(range(len(fitness_progress)), fitness_progress, label="Best Distance")
    ax.set_xlabel("Generation")
    ax.set_ylabal("Best Distance")
    ax.set_title("Evolution of Best Distance across Generations")

    if solver.max_generations <= 50:
        tick_interval = 2
    else:
        tick_interval = 20
    ax.set_xticks(range(0, len(fitness_progress), tick_interval))


    # Display final results
    st.write("### Final Solution:")
    final_best_distance = min(fitness_progress)
    final_best_path = shortest_paths[fitness_progress.index(final_best_distance)]
    st.write(f"**Final Best Distance:** {final_best_distance:.2f}")
    
    # Plot the final best path found
    fig, ax = plt.subplots()
    tour_x = [solver.city_coords[city][0] for city in final_best_path]
    tour_y = [solver.city_coords[city][1] for city in final_best_path]
    ax.plot(tour_x, tour_y, 'g-', marker='o', label="Final Best Path")
    ax.set_title("Final Best Path Found")
    ax.legend()
    st.pyplot(fig)