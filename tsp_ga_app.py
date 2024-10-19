import streamlit as st
import matplotlib.pyplot as plt
from TSP_GA import TSPSolver_GA 
import time

# TITLE + INSTRUCTIONS:
st.title("Solving TSP w/ Genetic Algorithm")
st.write("CREATED BY: Karis Land\n")
st.write("This app demonstrates the evolution of a solution to the Traveling Salesman Problem using a Genetic Algorithm.")


# USER DEFINED GA PARAMS:
tsp_file = st.text_input("Enter TSP file path:", "Random100.tsp")
data_set = st.selectbox("Select Data Set", ["D1_single_swap", "D2_single_invert", "D3_order_swap", "D4_order_invert"])
pop_size = st.slider("Population Size", 5, 500, 5)
max_gen = st.slider("Maximum Generations", 5, 1000, 5)
c_prob_high = st.slider("FIRST HALF: Crossover Probability", 0.5, 1.0, 0.95)
m_prob_high = st.slider("FIRST HALF: Mutation Probability", 0.01, 0.5, 0.05)
st.write(f"SECOND HALF: Crossover Probability = {c_prob_high * 0.85:.2f}")
st.write(f"SECOND HALF: Mutation Probability = {m_prob_high * 0.75:.2f}")
# solution_type = st.selectbox("Solution Type", ["dict", "list"])
algorithm = st.selectbox("Algorithm", ["GENETIC ALGORITHM"])
# algorithm = st.selectbox("Algorithm", ["GENETIC ALGORITHM", "BRUTE FORCE", "GREEDY: CLOSEST EDGE", "DEPTH FIRST SEARCH", "BREADTH FIRST SEARCH"])
parse = True

# DYNAMIC COMPONENTS:
animation_speed = st.slider("Animation Speed (seconds per generation)", 0.01, 2.0, 0.01)
dynamic_plot_placeholder = st.empty()
time_placeholder = st.empty()

# TSPSolver_GA INSTANCE: 
solver = TSPSolver_GA(
    tsp_file=tsp_file,
    data_set=data_set,
    start_city=1,
    pop_size=pop_size,
    max_gen=max_gen,
    c_prob_high=c_prob_high,
    m_prob_high=m_prob_high,
    algorithm=algorithm,
    parse=parse
)

min_city = min(solver.city_coords.keys())
max_city = max(solver.city_coords.keys())
solver.start_city = st.slider("Start City", min_city, max_city)

# Run the algorithm and visualize progress
if st.button("Run GA"):
    solver.generate_random_pop()
    fitness_progress = []
    shortest_paths = []
    generation = 0
    cumulative_time = 0


    # RUN ALGORITHM FOR EACH GENERATION:
    for generation in range(solver.max_generations):
        start_time = time.perf_counter()
        best_distance, best_path = solver.genetic_algorithm()  # Run a single generation
        fitness_progress.append(best_distance)
        shortest_paths.append(best_path)     
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cumulative_time += elapsed_time
        
        # PLOT WITH MATPLOTLIB:
        fig, ax = plt.subplots() 
        tour_x = [solver.city_coords[city][0] for city in best_path]
        tour_y = [solver.city_coords[city][1] for city in best_path]
        ax.plot(tour_x, tour_y, 'r-', marker='o', label="Best Path")

        # LABEL THE START CITY:
        start_x = solver.city_coords[best_path[0]][0]
        start_y = solver.city_coords[best_path[0]][1]
        ax.scatter(start_x, start_y, s=100, c='blue', label="Start City", edgecolor='black', zorder=5)

        # ADD FIGURE LABELS:
        ax.set_title(f"GA Solution Path ({best_distance:.2f}) - Gen ({generation + 1})")
        if data_set == "D1_single_swap":
            xlabel = "Single Point Crossover | Swap Mutation"
        elif data_set == "D2_single_invert":
            xlabel = "Single Point Crossover | Inversion Mutation"
        elif data_set == "D3_order_swap":
            xlabel = "Order Crossover | Swap Mutation"
        elif data_set == "D4_order_invert":
            xlabel = "Order Crossover | Inversion Mutation"
        ax.set_xlabel(xlabel)
        ax.legend(loc='upper left', bbox_to_anchor=(1,1))

        # DISPLAY FIGURE IN SAME SPOT (ANIMATION): 
        dynamic_plot_placeholder.pyplot(fig)

        # DISPLAY ALGORITHM RUNTIME:
        time_placeholder.write(f"**ELAPSED TIME:** {cumulative_time: .2f} seconds || {cumulative_time / 60: .2f} minutes\n")

        time.sleep(animation_speed)
        plt.close(fig)


    # EVOLUTION OF SOLUTIONS OVER GENERATIONS:
    fig1, ax = plt.subplots()
    ax.plot(range(len(fitness_progress)), fitness_progress, label="Best Distance")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Distance")
    ax.set_title("Evolution of Best Distance across Generations")
    if solver.max_generations <= 50:
        tick_interval = 5
    if solver.max_generations >= 100:
        tick_interval = 100
    else:
        tick_interval = 20
    ax.set_xticks(range(0, len(fitness_progress), tick_interval))
    st.pyplot(fig1)

    # FINAL SOLUTION
    st.write("### FINAL SOLUTION:")
    final_best_distance = min(fitness_progress)
    final_best_path = shortest_paths[fitness_progress.index(final_best_distance)]
    st.write(f"**Final Best Distance:** {final_best_distance:.2f}")

    # PLOT & DISPLAY FINAL SOLUTION:
    fig2, ax = plt.subplots()
    tour_x = [solver.city_coords[city][0] for city in final_best_path]
    tour_y = [solver.city_coords[city][1] for city in final_best_path]
    ax.plot(tour_x, tour_y, 'g-', marker='o', label="Final Best Path")

    # LABEL THE START CITY:
    start_x = solver.city_coords[best_path[0]][0]
    start_y = solver.city_coords[best_path[0]][1]
    ax.scatter(start_x, start_y, s=100, c='blue', label="Start City", edgecolor='black', zorder=5)

    # LABELS:
    ax.set_title("Final Best Path Found")
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    st.pyplot(fig2)
    st.write(f"**TOTAL RUNTIME:** {cumulative_time: .2f} seconds || {cumulative_time / 60: .2f} minutes\n")
    st.write(f"SOLUTION PATH: {best_path}")
    # EXPLICITLY CLOSE THE FIGURES TO FREE MEMORY:
    plt.close(fig1)
    plt.close(fig2)

    


# CODE GRAVEYARD:

        # COMPLETE THE CIRCUIT:
        # ax.plot([tour_x[-1], tour_x[0]], [tour_y[-1], tour_y[0]], '-r', marker='o')

# x_coords = [city[0] for city in solver.city_coords.values()]
# y_coords = [city[1] for city in solver.city_coords.values()]

# fig, ax = plt.subplots() 
# ax.plot(x_coords, y_coords, 'r', marker='o', label="Cities")

# for city_index, (x, y) in solver.city_coords.items():
#     plt.text(x + 1.5, y + 1.5, str(city_index), fontsize=10, ha='right', va='bottom')

# ax.legend(loc='upper left', bbox_to_anchor=(1,1))
# ax.set_title('TSP Cities')
# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Y Coordinate')
# st.pyplot(fig)
# plt.close(fig)

# SHOW SCATTER PLOT OF ORIGINAL TSP FILE:

# x_coords = [city[0] for city in solver.city_coords.values()]
# y_coords = [city[1] for city in solver.city_coords.values()]

# fig, ax = plt.subplots(figsize=(12,8), dpi=100) 
# ax.scatter(x_coords, y_coords, color='blue', label='Cities')

# for city_index, (x, y) in solver.city_coords.items():
#     plt.text(x + 1.5, y + 1.5, str(city_index), fontsize=10, ha='right', va='bottom')

# ax.legend(loc='upper left', bbox_to_anchor=(1,1))
# ax.set_title('TSP Cities')
# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Y Coordinate')
# st.pyplot(fig)
# plt.close(fig)

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