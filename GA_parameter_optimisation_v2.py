import random
from concurrent.futures import ProcessPoolExecutor
from script_signal import SIGNAL_PATH, denoise
import json

# Define important paths
json_path = 'results/genetic_algorithms/logs/population_data_10.json'
results_path = 'results/genetic_algorithms/logs/genetic_algorithm_results_10.txt'

# Define the fitness function
def fitness_function(solution):
    Ms, pds, sg = solution
    _, psnr_value = denoise(SIGNAL_PATH, Ms, pds, sg, plot_path = 'results/genetic_algorithms/signal_plots_10', results_path = results_path)
    return psnr_value

# Initialize population
def initialize_population(pop_size, bounds):
    return [
        [
            random.randint(bounds[0][0] // 100, bounds[0][1] // 100) * 100,  # Ms
            round(random.uniform(bounds[1][0], bounds[1][1]), 1),            # pds
            random.randint(bounds[2][0], bounds[2][1])                       # sg
        ]
        for _ in range(pop_size)
    ]

# Selection
def select(population, fitnesses, num_parents):
    parents = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    return [parent[0] for parent in parents[:num_parents]]

# Crossover
def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(parents, 2)
        crossover_point = random.randint(1, len(parent1) - 1)
        offspring.append(parent1[:crossover_point] + parent2[crossover_point:])
    return offspring

# Mutation
def mutate(offspring, mutation_rate, bounds):
    for individual in offspring:
        if random.random() < mutation_rate:
            mutation_index = random.randint(0, len(individual) - 1)
            if mutation_index == 0:  # Ms
                individual[mutation_index] = random.randint(bounds[0][0] // 100, bounds[0][1] // 100) * 100
            elif mutation_index == 1:  # pds
                individual[mutation_index] = round(random.uniform(bounds[1][0], bounds[1][1]), 1)
            elif mutation_index == 2:  # sg
                individual[mutation_index] = random.randint(bounds[2][0], bounds[2][1])
    return offspring

# Genetic Algorithm with early stopping
def genetic_algorithm(pop_size, bounds, num_generations, num_parents, mutation_rate, max_no_improvement):
    population = initialize_population(pop_size, bounds)
    all_generations = []
    best_fitness = -float('inf')
    no_improvement_counter = 0

    with open(results_path, 'a') as f:
        with ProcessPoolExecutor() as executor:
            for generation in range(num_generations):
                # Evaluate fitness in parallel
                fitnesses = list(executor.map(fitness_function, population))
                
                # Get best fitness of this generation
                max_gen_fitness = max(fitnesses)
                if max_gen_fitness > best_fitness:
                    best_fitness = max_gen_fitness
                    no_improvement_counter = 0  # Reset counter if improvement
                else:
                    no_improvement_counter += 1
                
                # Early stopping if no improvement over specified generations
                if no_improvement_counter >= max_no_improvement:
                    f.write(f"Stopping early at generation {generation} due to lack of improvement.")
                    break

                parents = select(population, fitnesses, num_parents)
                offspring_size = pop_size - num_parents
                offspring = crossover(parents, offspring_size)
                offspring = mutate(offspring, mutation_rate, bounds)
                population = parents + offspring

                # Store population for each generation
                all_generations.append(population)
                f.write(f"Generation {generation}: Best Fitness = {best_fitness}")

        best_solution = max(population, key=fitness_function)
        return best_solution, all_generations

# Parameters
pop_size = 5  # Reduced to improve runtime
bounds = [(500, 1000), (20.0, 30.0), (5, 20)]  # Adjusted for realistic ranges
num_generations = 10  # Reduced to improve runtime
num_parents = 3
mutation_rate = 0.45
max_no_improvement = 5  # Early stopping if no improvement for 3 generations

# Run Genetic Algorithm
best_solution, all_generations = genetic_algorithm(pop_size, bounds, num_generations, num_parents, mutation_rate, max_no_improvement)
with open(results_path, 'a') as f:
    f.write(f"Best Solution: {best_solution}")

# Save the population data to a file
with open(json_path, 'a') as f:
    json.dump(all_generations, f)