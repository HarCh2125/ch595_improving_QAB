# Implementing a Monte Carlo search for the optimal values of 
# the paramters Ms, sg, and pds

import random
from script_signal import SIGNAL_PATH, denoise

# Monte Carlo parameters
num_samples = 100  # Number of random samples to test
bounds = {
    'Ms': (200, 1000),    # Range for Ms
    'pds': (20.0, 30.0),    # Range for pds
    'sg': (5, 20)         # Range for sg
}

# Function to sample random parameters within bounds
def random_parameters(bounds):
    Ms = random.randint(bounds['Ms'][0] // 100, bounds['Ms'][1] // 100) * 100  # Ms in multiples of 100
    pds = round(random.uniform(bounds['pds'][0], bounds['pds'][1]), 1)         # pds with 1 decimal precision
    sg = random.randint(bounds['sg'][0], bounds['sg'][1])                      # sg as an integer
    return Ms, pds, sg

# Specifying the output file path
results_file_path = 'results/monte_carlo/logs/monte_carlo_results_6.txt'
plot_path = 'results/monte_carlo/signal_plots_6'

# Run Monte Carlo search
def monte_carlo_search(num_samples, bounds):
    best_params = None
    best_fitness = -float('inf')
    
    with open(results_file_path, 'a') as f:
        for i in range(num_samples):
            # Generate random parameters
            Ms, pds, sg = random_parameters(bounds)
            
            # Calculate fitness (PSNR) using denoise function
            _, psnr_value = denoise(SIGNAL_PATH, Ms, pds, sg, plot_path = plot_path, results_path = results_file_path)
            
            # Update best parameters if current PSNR is higher
            if psnr_value > best_fitness:
                best_fitness = psnr_value
                best_params = (Ms, pds, sg)
            
            # Write intermediate results to the results file
            f.write(f"Sample {i+1}/{num_samples}: Ms={Ms}, pds={pds}, sg={sg}, PSNR={psnr_value}")
        
        f.write("\nBest parameters found:")
        f.write(f"Ms={best_params[0]}, pds={best_params[1]}, sg={best_params[2]}, PSNR={best_fitness}")
        return best_params, best_fitness

# Run Monte Carlo search
best_params, best_fitness = monte_carlo_search(num_samples, bounds)
with open(results_file_path, 'a') as f:
    f.write(f"Optimal Parameters: Ms={best_params[0]}, pds={best_params[1]}, sg={best_params[2]}, PSNR={best_fitness}")
