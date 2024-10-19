import pandas as pd
import matplotlib.pyplot as plt
from TSP_GA import TSPSolver_GA

if __name__ == "__main__":

    data_sets = [
        'D1_single_swap', 
        'D2_single_invert', 
        'D3_order_swap', 
        'D4_order_invert', 
    ]

    data_set_results = {
        'Dataset': ['D1_single_swap', 'D2_single_invert', 'D3_order_swap', 'D4_order_invert'], 
        'Crossover Type': ['Single Point', 'Single Point', 'Order', 'Order'],
        'Mutation Type': ['Swap', 'Inversion', 'Swap', 'Inversion'],
        'Average Runtime': [0, 0, 0, 0],
        'Average Best Distance': [0, 0, 0, 0],
        }
    
    p_size = 50
    m_gen = 600
    c_prob = 100
    m_prob = 0.2
    num_trials = 2

    for idx, data_set in enumerate(data_sets):
        sum_runtime = 0
        average_runtime = 0
        sum_distance = 0
        ave_best_distance = 0   
        for i in range(num_trials):   
            solver = TSPSolver_GA('Random100.tsp', data_set, pop_size=p_size, max_gen=m_gen, c_prob_high=c_prob, m_prob_high=m_prob, parse=True, run=True, print_results=False)
            sum_runtime += solver.runtime
            sum_distance += solver.best_distance
        average_runtime = sum_runtime/num_trials
        ave_best_distance = sum_distance/num_trials
        data_set_results['Average Runtime'][idx] = average_runtime
        data_set_results['Average Best Distance'][idx] = ave_best_distance

    df = pd.DataFrame(data_set_results)
   
    data_set_params = {
        'NUM TRIALS': num_trials,        
        'POP SIZE': solver.population_size,
        'GENERATIONS': solver.max_generations,        
        'Crossover HIGH Prob': solver.cross_prob_HIGH,
        'Mutation HIGH Prob': solver.mutate_prob_HIGH,
        'START CITY': solver.start_city,
        
    }

    print("-" * 100)
    print("PERFORMANCE STATISTICS")
    solver.print_this('dict', data_set_params)
    print('_' * 100)
    print(df)
    
    datasets = data_set_results['Dataset']
    average_runtime = data_set_results['Average Runtime']
    average_best_distance = data_set_results['Average Best Distance']

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Create bar plot for average runtime
    ax1.bar(datasets, average_runtime, color='skyblue', label='Average Runtime (s)')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Average Runtime (s)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create another axis for the average best distance (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(datasets, average_best_distance, color='orange', marker='o', label='Average Best Distance')
    ax2.set_ylabel('Average Best Distance', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add titles and legends
    fig.suptitle('Average Runtime and Best Distance by Dataset', fontsize=14)
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.05), borderaxespad=0.)
    ax2.legend(loc='upper right', bbox_to_anchor=(1,1.05), borderaxespad=0.)

    params_text = '\n'.join([f"{key}: {value}" for key, value in data_set_params.items()])
    plt.figtext(0.02, 0.05, params_text, wrap=True, horizontalalignment='left', fontsize=10)
    # plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.subplots_adjust(bottom=0.25)
    plt.show()
