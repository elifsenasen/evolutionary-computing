from fitness import evaluate, log_generation
from rich.console import Console
from rich.traceback import install
from ariel.ec.genotypes.tree.operators import random_tree
import numpy as np
import random
import fitness


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

max_modules = 20
population_size = 75

def create_individual(max_modules: int):
    genome = random_tree(max_modules)
    individual = Individual()
    individual.genotype = genome.to_dict()
    individual.tags["ps"] = False # for chosen parents then we change this to true.
    return individual

def create_population():
    population = Population([create_individual(max_modules) for _ in range(population_size)])
    return population

def ea(seed:int):
    np.random.seed(seed)
    random.seed(seed)
    fitness.CURRENT_SEED = seed

    population = create_population()

    ops = [
        # EAOperation(parent_selection),
        # EAOperation(crossover),
        # EAOperation(mutate),
        EAOperation(evaluate),
        EAOperation(log_generation),
        # EAOperation(survivor_selection),
    ]

    algorithm=EA(population, ops, num_steps=100) #generation size
    algorithm.run()
    return algorithm


def main():
    SEEDS = [42, 123, 456, 789, 1011]
    for seed in SEEDS:
        ea(seed)
 
    print(f"Logged {len(fitness.LOG)} generation records")
    print(fitness.LOG[:3])

    fitness.save_log_csv()

    targets = fitness.load_targets()
    baseline_results = fitness.random_search_baseline(
        num_evaluations=500,
        targets=targets,
    )
    fitness.save_baseline_csv(baseline_results)
    print("Saved fitness_log.csv and baseline_log.csv")

if __name__ == "__main__":
    main()