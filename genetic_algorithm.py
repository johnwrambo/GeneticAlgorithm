import random
import matplotlib
from datetime import datetime
import time
import pandas as pd
import numpy as np
from random import randint
from random import choices
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')  # Use TkAgg instead of InterAgg

# initial conditions
note = "without cache"
population = 60
number_of_generations = 1000
weight_limit = 800  # Knapsack capacity

knapsack_key = pd.DataFrame([
    [300, 1200],  # High value, moderate weight
    [150, 500],  # Good value-to-weight ratio
    [85, 350],  # Light item, decent value
    [105, 433],  # Moderate weight, solid value
    [30, 192],  # Very light, low value
    [400, 900],  # Heavy but valuable
    [700, 1500],  # Very heavy, but high value
    [250, 750],  # Balanced weight-to-value ratio
    [500, 800],  # Medium-heavy, fair value
    [50, 300]  # Lightweight, reasonable value
], columns=["weight", "value"])


# initialize

def populate(pop_size: int):
    """Decorator iterates the function into a list of the length of the input."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            generation = [func(*args, **kwargs) for i in range(pop_size)]
            return generation

        return wrapper

    return decorator


def log_time_results(pop, gen, weight, note):
    """Decorator that logs the execution time and run time of the function."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            current_date_and_time = datetime.now()
            start = time.time()
            results = func(*args, **kwargs)
            end = time.time()
            total_time = end - start
            with open("log.txt", "a") as file:  # "w" mode creates or overwrites the file
                file.write(
                    f"{current_date_and_time}-: {results} pop:{pop} gens:{gen} weight:{weight} - run time: {total_time:.2f}->{note}\n")
            print(f"Total run time: {total_time}")
            return results

        return wrapper

    return decorator


@populate(population)
def generate_population(input_dataframe):
    genome_length = len(input_dataframe)
    return choices([0, 1], k=genome_length)


def evaluate_genome(dataframe_key, constraint, current_generation):
    """takes as input the initial dataframe with the problem conditions, the constraint,
    such as the weight limit, and a list of genomes. The function evaluates each gene in the genome and assigns weight
    and value, and then determines if the genome is fit or not based on if it fits the constraint"""
    fit_generation = {"genome": [], "weight": [], "value": [], "is fit": []}
    for genome in current_generation:
        genome_logic = np.array(genome).astype(bool)
        weight = dataframe_key[genome_logic]["weight"].sum()
        value = dataframe_key[genome_logic]["value"].sum()
        if weight > constraint or value == 0:
            is_fit = 0
        else:
            is_fit = 1
        fit_generation["genome"].append(genome)
        fit_generation["weight"].append(weight)
        fit_generation["value"].append(value)
        fit_generation["is fit"].append(is_fit)

    return pd.DataFrame(fit_generation)


def check_fitness(dataframe_key, current_generation, constraint, filtered: bool = True):
    """takes as input the initial dataframe with the problem conditions, the constraint,
    such as the weight limit, and a list of genomes to be evaluated"""
    filtered_df = evaluate_genome(dataframe_key, constraint, current_generation)
    filtered_df = filtered_df[filtered_df["is fit"] == 1]
    filtered_df = filtered_df.sort_values("value", ascending=False)
    if filtered:
        filtered_df = filtered_df.drop_duplicates("genome")
        filtered_df = filtered_df.reset_index(drop=True)
    # max_value = filtered_df.value.values.max()
    return filtered_df


# @populate(population)
def single_point_crossover(parents):
    """Takes as input a value pair of parents genomes and outputs two new offspring genomes"""
    genome_1, genome_2 = parents
    if len(genome_1) != len(genome_2):
        raise ValueError("Length of genomes do not match")
    genome_length = len(genome_1)
    slice_location = random.randint(1, genome_length - 1)
    slice_a_1 = genome_1[:slice_location]
    slice_a_2 = genome_1[slice_location:]
    slice_b_1 = genome_2[:slice_location]
    slice_b_2 = genome_2[slice_location:]
    genome_1_new = slice_a_1 + slice_b_2
    genome_2_new = slice_b_1 + slice_a_2
    return genome_1_new, genome_2_new


def mutate(genome):
    """Takes as input one genome and randomly mutates the genome"""
    genome_length = len(genome)
    mutate_roll = random.randint(1, 100)
    if mutate_roll < 3:
        mutate_index = random.randint(0, genome_length - 1)
        if genome[mutate_index] == 0:
            genome[mutate_index] = 1
            return genome
        else:
            genome[mutate_index] = 0
            return genome
    else:
        return genome


def fitness_function(dataframe):
    probability = [0] * (len(dataframe))
    probability_total = [0] * (len(dataframe))
    sum_values = dataframe.value.values.sum()
    for i in dataframe.index:
        probability[i] = round(float(dataframe.value[i] / sum_values))
        probability_total[i] = round(float(sum(probability)))
    r1 = 1
    r2 = 1
    wheel = [0]
    wheel = wheel + probability_total
    r1_select = 1
    r2_select = 1
    # while r1 == r2 or r1_select == r2_select:
    r1 = randint(1, 100) / 100
    r2 = randint(1, 100) / 100
    for n, i in enumerate(probability_total):
        if wheel[n] < r1 <= probability_total[n]:
            r1_select = n
        if wheel[n] < r2 <= probability_total[n]:
            r2_select = n

    filtered_df = dataframe.iloc[[r1_select, r2_select]]
    # print(f"selected parents: \n{filtered_df}")
    filtered_df = filtered_df.sort_values("value", ascending=False).reset_index(drop=True)
    # print(filtered_df)
    return filtered_df


def evolve(parents, population_size):
    population_size = int((population_size - 2) / 2)
    generation = [parents[0]] + [parents[1]]
    for i in range(population_size):
        off_spring = single_point_crossover(parents)
        generation += [off_spring[0]] + [off_spring[1]]

    return generation


@log_time_results(population, number_of_generations, weight_limit, note)
def genetic_algorithm(input_dataframe, population_size, generations, weight_limit, options: str = None):
    first_generation = generate_population(input_dataframe)
    current_generation = first_generation
    solution_vec = []
    for _ in range(generations):
        unfit_removed = check_fitness(knapsack_key, current_generation, weight_limit)
        # print(f"unfit removed: \n{unfit_removed}")
        parents_df = fitness_function(unfit_removed)
        parents = parents_df.genome.tolist()
        solution_row = parents_df.loc[parents_df["value"].idxmax()]
        solution_vec.append(solution_row.genome)

        current_generation = evolve(parents, population_size)
        for n, i in enumerate(current_generation):
            current_generation[n] = mutate(current_generation[n])
    # final_generation = check_fitness(knapsack_key, current_generation, weight_limit, filtered=True)
    solution_df = check_fitness(knapsack_key, solution_vec, weight_limit, filtered=True)
    print(solution_df.sort_values("value", ascending=False))
    solution_vec = solution_df.value.tolist()
    max_solution = solution_df.loc[solution_df["value"].idxmax()]
    weight = max_solution.weight
    max_value = solution_df.value.values.max()
    sol = [max_value.tolist(), weight.tolist()]
    if options == "plot":
        x_vec = range(0, len(solution_vec))
        plt.plot(x_vec, solution_vec)
        return sol, [x_vec, solution_vec]
    return sol


results_ga = genetic_algorithm(knapsack_key, population, number_of_generations, weight_limit)
print(f"results: {results_ga}")
