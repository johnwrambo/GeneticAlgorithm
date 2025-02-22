import random
import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg instead of InterAgg
import pandas as pd
import numpy as np
from random import randint
from random import choices
import matplotlib.pyplot as plt

# initial conditions

population = 25

knapsack = pd.DataFrame([[300, 1200],
                         [150, 160],
                         [85, 350],
                         [105, 433],
                         [30, 192]],
                        columns=["weight", "value"])

gene_length = len(knapsack)


# initialize

def populate(pop_size):
    def decorator(func):
        def wrapper(*args, **kwargs):
            generation = [func(*args, **kwargs) for i in range(pop_size)]
            return generation

        return wrapper

    return decorator


@populate(population)
def generate_population(length):
    return choices([0, 1], k=length)


generation = generate_population(gene_length)
generation


def evaluate_genome(dataframe, genome):
    genome = np.array(genome).astype(bool)
    weight = dataframe[genome]["weight"].sum()
    value = dataframe[genome]["value"].sum()
    return weight, value

def fitness(dataframe, batch, weight_limit):
    fit_generation = {"genome": [], "weight": [], "value": [], "fitness": []}
    for genome in batch:
        [weight, value] = evaluate_genome(dataframe, genome)
        if weight > weight_limit or weight == 0:
            fitness_score = 0
        else:
            fitness_score = 1
        fit_generation["genome"].append(genome)
        fit_generation["weight"].append(weight)
        fit_generation["value"].append(value)
        fit_generation["fitness"].append(fitness_score)

    return pd.DataFrame(fit_generation)


generation = fitness(knapsack, generation, 450)
# print(batch[batch[fitness]==1])

filtered_df = generation[generation["fitness"] == 1]
filtered_df.sort_values("value", ascending=False)

plt.plot(filtered_df["value"], filtered_df["weight"])

print(filtered_df)
