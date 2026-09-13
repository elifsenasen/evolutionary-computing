
from rich.console import Console
from rich.traceback import install
from ariel.ec.genotypes.tree.operators import random_tree
import numpy as np
import random


from ariel.ec import (
    EA,
    Crossover,
    EAOperation,
    Individual,
    IntegerMutator,
    IntegersGenerator,
    Population,
    config,
)

install()
console = Console()

max_moduls = 20
population_size = 75

def create_individual(max_moduls: int):
    genome = random_tree(max_moduls)
    individual = Individual()
    individual.genotype = genome.to_dict()
    individual.tags["ps"] = False # for chosen parents then we change this to true.
    return individual

def create_population():
    population = Population([create_individual(max_moduls) for _ in range(population_size)])
    return population

def ea(seed:int):
    np.random.seed(seed)
    random.seed(seed)

    population = create_population()

    ops = [
        # EAOperation(parent_selection),
        # EAOperation(crossover),
        # EAOperation(mutate),
        # EAOperation(evaluate),
        # EAOperation(survivor_selection),
    ]

    algorithm=EA(population, ops, num_steps=100) #generation size
    algorithm.run()
    return algorithm


def main():
    SEEDS = [42, 123, 456, 789, 1011]
    for seed in SEEDS:
        ea(seed)
 


if __name__ == "__main__":
    main()