import random
from concurrent.futures import ThreadPoolExecutor
from script_signal import SIGNAL_PATH, denoise
import json

# Define the fitness function
def fitness_function(solution):
    Ms, pds, sg = solution
    # Use the denoise function to calculate the fitness
    psnr_value = denoise(SIGNAL_PATH, Ms, pds, sg)
    return psnr_value

# Initialize population
def initialize_population(pop_size, bounds):
    population = []
    for _ in range(pop_size):
        individual = [
            random.randint(bounds[0][0] // 100, bounds[0][1] // 100) * 100,  # Ms
            round(random.uniform(bounds[1][0], bounds[1][1]), 1),  # pds
            random.randint(bounds[2][0], bounds[2][1])  # sg
        ]
        population.append(individual)
    return population

# Selection
def select(population, fitnesses, num_parents):
    parents = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    return [parent[0] for parent in parents[:num_parents]]

# Crossover
def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        offspring.append(child)
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

# Genetic Algorithm
def genetic_algorithm(pop_size, bounds, num_generations, num_parents, mutation_rate):
    population = initialize_population(pop_size, bounds)
    all_generations = []  # To store population values for each generation
    with ThreadPoolExecutor() as executor:
        for generation in range(num_generations):
            # Evaluate fitness in parallel
            fitnesses = list(executor.map(fitness_function, population))
            parents = select(population, fitnesses, num_parents)
            offspring_size = pop_size - num_parents
            offspring = crossover(parents, offspring_size)
            offspring = mutate(offspring, mutation_rate, bounds)
            population = parents + offspring
            best_solution = max(population, key=fitness_function)
            print(f"Generation {generation}: Best Solution = {best_solution}, Fitness = {fitness_function(best_solution)}")
            all_generations.append(population)  # Record the population for this generation
    return best_solution, all_generations

# Parameters
pop_size = 20
bounds = [(100, 1000), (0.1, 10.0), (5, 20)]  # Some bounds for Ms, pds, sg
num_generations = 50
num_parents = 10
mutation_rate = 0.1

# Run Genetic Algorithm
best_solution, all_generations = genetic_algorithm(pop_size, bounds, num_generations, num_parents, mutation_rate)
print(f"Best Solution: {best_solution}")

# Save the population data to a file
with open('population_data.json', 'w') as f:
    json.dump(all_generations, f)