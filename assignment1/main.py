from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
import fitness
from fitness import evaluate, log_generation
from parent_selection import parent_selection
from crossover import crossover
from mutation import mutate, survivor_selection
from rich.console import Console
from rich.traceback import install
from ariel.ec.genotypes.tree.operators import random_tree
from ariel.ec import (
    EA,
    EAOperation,
    Individual,
    Population,
)

install()
console = Console()

# parameters
max_modules = 20
population_size = 75
num_generations = 100

SEEDS = [42, 123, 456, 789, 1011]



#output directory
HERE = Path(__file__).parent
DATA = HERE / "__data__"
DATA.mkdir(parents=True, exist_ok=True)


def create_individual(max_modules: int):
    genome = random_tree(max_modules)

    individual = Individual()
    individual.genotype = genome.to_dict()

    individual.tags["ps"] = False
    return individual


def create_population():
    population = Population([create_individual(max_modules) for _ in range(population_size)])
    return population


def ea(seed: int, mutation_rate: float):
    np.random.seed(seed)
    random.seed(seed)
    fitness.CURRENT_SEED = seed
    fitness.evolution_count = 0
    fitness.best_individual = None

    population = create_population()
    population = evaluate(population) #for first generation

    ops = [
        EAOperation(parent_selection),
        EAOperation(crossover),
        EAOperation(mutate, mutation_rate=mutation_rate),
        EAOperation(evaluate),
        EAOperation(survivor_selection, target_size=population_size),
        EAOperation(log_generation),
        EAOperation(fitness.save_best_individual),# Save best fitness/genotype 

    ]

    algorithm=EA(population, ops, num_steps=num_generations, is_maximisation=False)
    algorithm.run()

    evaluation_count = fitness.evolution_count

    return evaluation_count



def run_variant(variant_name: str, mutation_rate: float):

    best_per_run = []
    evaluation_counts = []

    for seed in SEEDS:

        # Remember where this run starts in the global log.
        start = len(fitness.LOG)
        evaluation_count = ea(seed=seed, mutation_rate=mutation_rate)
        evaluation_counts.append(evaluation_count)

        for generation, row in enumerate(fitness.LOG[start:], start=1):
            row["variant"] = variant_name
            row["mutation_rate"] = mutation_rate
            row["generation"] = generation

        
        best_per_run.append(
            (
                fitness.best_individual["fitness"],
                fitness.best_individual["genotype"],
            )
        )

    console.log(
        f"{variant_name}: "
        f"{len(SEEDS)} runs x "
        f"{num_generations} generations logged"
    )

    return best_per_run, evaluation_counts


def plot_fitness_curves(path: Path, baseline_runs: list[list[float]]):

    per_variant = defaultdict(lambda: defaultdict(list))

    for i in fitness.LOG:
        variant = i["variant"]
        generation = i["generation"]
        per_variant[variant][generation].append(i["best"])

    plt.figure(figsize=(8, 5))

    display_names = {
        "EA1_mutation_0.1": "EA, mutation rate = 0.1",
        "EA2_mutation_0.3": "EA, mutation rate = 0.3",
    }

    # Plot the two EA variants
    for variant_name, generations_data in per_variant.items():

        all_means = []
        all_stds = []
        generations = sorted(generations_data.keys())
        for i in generations:
            fitness_values = generations_data[i]
            fitness_means = np.mean(fitness_values)
            fitness_stds = np.std(fitness_values)
            all_means.append(fitness_means)
            all_stds.append(fitness_stds)

        means = np.array(all_means)

        stds = np.array(all_stds)

        plt.plot(
            generations,
            means,
            linewidth=2,
            label=display_names.get(
                variant_name,
                variant_name,
            ),
        )

        plt.fill_between(
            generations,
            means - stds,
            means + stds,
            alpha=0.20,
        )
    baseline_by_generation = []
    for run in baseline_runs:
        sampled_run = []
        for generation in range(1, num_generations + 1):

            evaluation_index = int(np.ceil(generation * len(run) / num_generations)) - 1
            sampled_run.append(run[evaluation_index])
        baseline_by_generation.append(sampled_run)
    baseline_array = np.array(
        baseline_by_generation
    )

    baseline_mean = np.mean(
        baseline_array,
        axis=0,
    )

    baseline_std = np.std(
        baseline_array,
        axis=0,
    )

    generations = np.arange(
        1,
        num_generations + 1,
    )

    plt.plot(
        generations,
        baseline_mean,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Random search",
    )

    plt.fill_between(
        generations,
        baseline_mean - baseline_std,
        baseline_mean + baseline_std,
        color="gray",
        alpha=0.20,
    )

    plt.xlabel("Generation")
    plt.ylabel("Best-so-far fitness (lower is better)")

    plt.title(
        "EA variants compared with random search"
    )

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():

    fitness.LOG.clear()
    # 2 different mutation rates for research question
    best_ea1, budgets_ea1 = run_variant(
        "EA1_mutation_0.1",
        0.1
    )

    best_ea2, budgets_ea2 = run_variant(
        "EA2_mutation_0.3",
        0.3
    )
    if (budgets_ea1 == budgets_ea2):
        console.log("Both variants used the same evaluation budget.")

    # Rrandom search baseline
    targets = fitness.load_targets()
    baseline_runs = []
    for i in range(len(SEEDS)):

        np.random.seed(SEEDS[i])
        random.seed(SEEDS[i])
        evaluation_budget = budgets_ea1[i]
        baseline = fitness.random_search_baseline(
            num_evaluations=evaluation_budget,
            targets=targets,
            max_modules=max_modules,
        )
        baseline_runs.append(baseline)

    # fitness plot
    plot_fitness_curves(DATA / "fitness_vs_generation.png", baseline_runs)

    # find best body
    best_variant = None
    best_genotype = None
    best_fitness = float("inf")

    for fitness_value, genotype in best_ea1:
        if fitness_value < best_fitness:
            best_fitness = fitness_value
            best_genotype = genotype
            best_variant = "EA1_mutation_0.1"

    for fitness_value, genotype in best_ea2:
        if fitness_value < best_fitness:
            best_fitness = fitness_value
            best_genotype = genotype
            best_variant = "EA2_mutation_0.3"

    console.log(
        f"best overall: "
        f"{best_variant}, "
        f"fitness={best_fitness:.4f}"
    )

if __name__ == "__main__":
    main()