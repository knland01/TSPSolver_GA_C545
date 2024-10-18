import streamlit as st
import matplotlib.pyplot as plt
from TSP_GA import TSPSolver_GA 
import time

# TITLE + INSTRUCTIONS:
st.title("Solving TSP w/ Genetic Algorithm")
st.write("This app demonstrates the evolution of a solution to the Traveling Salesman Problem using a Genetic Algorithm.")

# USER DEFINED GA PARAMS:
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


# TSPSolver_GA INSTANCE: 
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

# DYNAMIC COMPONENTS:
animation_speed = st.slider("Animation Speed (seconds per generation)", 0.0001, 3.0, 0.5)
dynamic_plot_placeholder = st.empty()
# max_gens = solver.max_generations

# Run the algorithm and visualize progress
if st.button("Run GA"):
    fitness_progress = []
    shortest_paths = []
    generation = 0


    # RUN THROUGH EACH GENERATION
    for generation in range(solver.max_generations):
        solution_dict = solver.genetic_algorithm()  # Run a single generation
        best_path = solution_dict['SOLUTION']
        best_distance = solution_dict['TOTAL DISTANCE']
        # best_path = min(solver.current_population, key=solver.calc_total_distance)
        # best_distance = solver.calc_total_distance(best_path)
        fitness_progress.append(best_distance)
        shortest_paths.append(best_path)     

        
        # PLOT WITH MATPLOTLIB
        fig, ax = plt.subplots() 
        tour_x = [solver.city_coords[city][0] for city in best_path]
        tour_y = [solver.city_coords[city][1] for city in best_path]
        ax.plot(tour_x, tour_y, 'r-', marker='o', label="Best Path")
        ax.set_title(f"Best Path ({best_distance:.2f}) - Generation {generation + 1}")
        ax.legend()

        dynamic_plot_placeholder.pyplot(fig)
        time.sleep(animation_speed)
        plt.close(fig)

    
    # EVOLUTION OF SOLUTIONS OVER GENERATIONS:
    fig1, ax = plt.subplots()
    ax.plot(range(len(fitness_progress)), fitness_progress, label="Best Distance")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Distance")
    ax.set_title("Evolution of Best Distance across Generations")
    if solver.max_generations <= 50:
        tick_interval = 2
    if solver.max_generations >= 100:
        tick_interval = 50
    else:
        tick_interval = 20
    ax.set_xticks(range(0, len(fitness_progress), tick_interval))
    st.pyplot(fig1)




    # FINAL SOLUTION
    st.write("### Final Solution:")
    final_best_distance = min(fitness_progress)
    final_best_path = shortest_paths[fitness_progress.index(final_best_distance)]
    st.write(f"**Final Best Distance:** {final_best_distance:.2f}")

    # PLOT & DISPLAY FINAL SOLUTION:
    fig2, ax = plt.subplots()
    tour_x = [solver.city_coords[city][0] for city in final_best_path]
    tour_y = [solver.city_coords[city][1] for city in final_best_path]
    ax.plot(tour_x, tour_y, 'g-', marker='o', label="Final Best Path")
    ax.set_title("Final Best Path Found")
    ax.legend()
    st.pyplot(fig2)

    # EXPLICITLY CLOSE THE FIGURES TO FREE MEMORY:
    plt.close(fig1)
    plt.close(fig2)

    


# CODE GRAVEYARD:

# if 'generation_slider' not in st.session_state:
#     st.session_state['generation_slider'] = 1

# generation_plots = []
# dynamic_slider_placeholder = st.empty()

        # generation_plots.append((best_path, best_distance))

    # with dynamic_slider_placeholder:
    
# if generation_plots:
#     selected_generation = st.slider("Select Generation", 1, max_gens, st.session_state['generation_slider'], key="unique_generation_slider") 
#     st.session_state["generation_slider"] = selected_generation

#     # st.pyplot(generation_plots[selected_generation - 1]) 

#     best_path, best_distance = generation_plots[selected_generation - 1]
#     fig, ax = plt.subplots() 
#     # [3] Obtain x, y coordinates from the coordinates dictionary
#     tour_x = [solver.city_coords[city][0] for city in best_path]
#     tour_y = [solver.city_coords[city][1] for city in best_path]
#     # [4] Add x, y coords to the axis object
#     # ... 'r-' = draw red line | 'o' = circular markers
#     ax.plot(tour_x, tour_y, 'r-', marker='o', label="Best Path")
#     ax.set_title(f"Generation {selected_generation} - Best Distance {best_distance:.2f}")
#     ax.legend()
#     st.pyplot(fig)


    # st.line_chart(fitness_progress)
            # Display current generation and best path details
        # st.metric("Generation", generation + 1)
        # st.metric("Best Distance", f"{best_distance:.2f}")

        # algorithm = st.text_input("Algorithm", "GENETIC ALGORITHM") 
# ... eventually make this a toggle for the different tsp algorithms?
# assist = st.checkbox("Assist", True)