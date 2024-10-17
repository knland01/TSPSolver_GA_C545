import streamlit as st
import matplotlib.pyplot as plt
from TSP_GA import TSPSolver_GA  # Ensure this points to your class file

# Title and Instructions
st.title("TSP Solver Using Genetic Algorithm")
st.write("This app demonstrates the evolution of a solution to the Traveling Salesman Problem using a Genetic Algorithm.")

# Collect GA Parameters from User
tsp_file = st.text_input("Enter TSP file path:", "Random100.tsp")
data_set = st.selectbox("Select Data Set", ["D1_single_swap", "D2_single_invert", "D3_order_swap", "D4_order_invert"])
pop_size = st.slider("Population Size", 50, 500, 200)
max_gen = st.slider("Maximum Generations", 100, 1000, 200)
c_prob_high = st.slider("High Crossover Probability", 0.5, 1.0, 0.95)
m_prob_high = st.slider("High Mutation Probability", 0.01, 0.1, 0.05)
solution_type = st.selectbox("Solution Type", ["dict", "list"])
algorithm = st.text_input("Algorithm", "GENETIC ALGORITHM")
assist = st.checkbox("Assist", True)

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
        st.metric("Best Distance", best_distance)
        
        # Update path plot with best path in the current generation
        fig, ax = plt.subplots()
        tour_x = [solver.city_coords[city][0] for city in best_path]
        tour_y = [solver.city_coords[city][1] for city in best_path]
        ax.plot(tour_x, tour_y, 'r-', marker='o', label="Best Path")
        ax.set_title(f"Best Path - Generation {generation + 1}")
        ax.legend()
        st.pyplot(fig)

    # Display overall fitness evolution
    st.line_chart(fitness_progress)

    # Display final results
    st.write("### Final Solution:")
    final_best_distance = min(fitness_progress)
    final_best_path = shortest_paths[fitness_progress.index(final_best_distance)]
    st.write(f"**Final Best Distance:** {final_best_distance}")
    
    # Plot the final best path found
    fig, ax = plt.subplots()
    tour_x = [solver.city_coords[city][0] for city in final_best_path]
    tour_y = [solver.city_coords[city][1] for city in final_best_path]
    ax.plot(tour_x, tour_y, 'g-', marker='o', label="Final Best Path")
    ax.set_title("Final Best Path Found")
    ax.legend()
    st.pyplot(fig)