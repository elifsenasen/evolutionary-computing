from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco as mj
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
# from ariel.ec.genotypes.tree.tree_genome import TreeGenome
# from ariel.simulation.environments import SimpleFlatWorld
# from ariel.utils.renderers import single_frame_renderer
# from ariel.body_phenotypes.robogen_lite.constructor import (
#     construct_mjspec_from_graph,
# )
from ariel.ec import (
    EA,
    EAOperation,
    Individual,
    Population,
)


install()
console = Console()


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

max_modules = 20
population_size = 75
num_generations = 100

SEEDS = [42, 123, 456, 789, 1011]



# ============================================================
# OUTPUT DIRECTORY
# ============================================================

HERE = Path(__file__).parent
DATA = HERE / "__data__"
DATA.mkdir(parents=True, exist_ok=True)


# ============================================================
# POPULATION CREATION
# ============================================================

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
    population = evaluate(population) # for first generation

    ops = [
        EAOperation(parent_selection),
        EAOperation(crossover),
        EAOperation(mutate, mutation_rate=mutation_rate),
        EAOperation(evaluate),
        EAOperation(survivor_selection, target_size=population_size),
        EAOperation(log_generation),
        EAOperation(fitness.save_best_individual),# Save best fitness/genotype while Individuals are

    ]

    algorithm=EA(population, ops, num_steps=num_generations, is_maximisation=False)
    algorithm.run()

    evaluation_count = fitness.evolution_count

    return evaluation_count


# ============================================================
# RUN ONE VARIANT ACROSS ALL 5 SEEDS
# ============================================================

def run_variant(variant_name: str, mutation_rate: float) :

    best_per_run = []
    evaluation_counts = []

    for seed in SEEDS:

        # Remember where this run starts in the global log.
        start = len(fitness.LOG)

        evaluation_count = ea(seed=seed, mutation_rate=mutation_rate)
        evaluation_counts.append(evaluation_count)

        # Add experiment information to the rows produced
        # during this run.

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



# def save_full_log_csv(path: Path) -> None:

#     with open(path, "w") as f:

#         f.write(
#             "variant,mutation_rate,seed,"
#             "generation,best,mean,worst\n"
#         )

#         for row in fitness.LOG:

#             f.write(
#                 f"{row['variant']},"
#                 f"{row['mutation_rate']},"
#                 f"{row['seed']},"
#                 f"{row['generation']},"
#                 f"{row['best']},"
#                 f"{row['mean']},"
#                 f"{row['worst']}\n"
#             )


def plot_fitness_curves(path: Path, baseline_best: float) -> None:

    per_variant = defaultdict(
        lambda: defaultdict(list)
    )

    for row in fitness.LOG:

        per_variant[
            row["variant"]
        ][
            row["generation"]
        ].append(row["best"])

    plt.figure(figsize=(8, 5))

    for variant_name, gens in per_variant.items():

        generations = sorted(gens)

        means = np.array(
            [
                np.mean(gens[g])
                for g in generations
            ]
        )

        stds = np.array(
            [
                np.std(gens[g])
                for g in generations
            ]
        )

        plt.plot(
            generations,
            means,
            label=variant_name,
        )

        plt.fill_between(
            generations,
            means - stds,
            means + stds,
            alpha=0.2,
        )

    plt.axhline(
        y=baseline_best,
        linestyle="--",
        label="Random search final mean",
    )

    plt.xlabel("Generation")

    plt.ylabel(
        "Best fitness "
        "(tree edit distance + std, lower is better)"
    )

    plt.title(
        "Effect of mutation rate on evolved body fitness"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(path)
    plt.close()

    console.log(f"saved {path}")


# VISUALIZE BEST BODY

# def visualize_best(
#     genotype: dict,
#     file_name: str,
# ) -> None:

#     mj.set_mjcb_control(None)

#     genome = TreeGenome.from_dict(genotype)

#     body = genome.to_networkx()

#     world = SimpleFlatWorld()

#     robot = construct_mjspec_from_graph(body)

#     world.spawn(
#         robot.spec,
#         position=[0.0, 0.0, 0.1],
#         correct_collision_with_floor=True,
#     )

#     model = world.spec.compile()

#     data = mj.MjData(model)

#     mj.mj_resetData(model, data)
#     mj.mj_forward(model, data)

#     save_path = str(
#         DATA / f"{file_name}.png"
#     )

#     single_frame_renderer(
#         model,
#         data,
#         save=True,
#         save_path=save_path,
#     )

#     console.log(f"saved {save_path}")


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    # Avoid old data if main() is executed again
    fitness.LOG.clear()

    best_ea1, budgets_ea1 = run_variant(
        "EA1_mutation_0.1",
        0.1
    )

    best_ea2, budgets_ea2 = run_variant(
        "EA2_mutation_0.3",
        0.3
    )



    # --------------------------------------------------------
    # RANDOM SEARCH BASELINE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SAVE RANDOM SEARCH RESULTS
    # --------------------------------------------------------

    # baseline_path = DATA / "baseline_log.csv"

    # with open(baseline_path, "w") as f:

    #     f.write(
    #         "seed,evaluation,best_so_far\n"
    #     )
    #     for seed in range(len(SEEDS)):
    #         run = baseline_runs[seed]

    #         for i in range(len(run)):
    #             evaluation = i + 1
    #             value = run[i]

    #             f.write(
    #                 f"{SEEDS[seed]},"
    #                 f"{evaluation},"
    #                 f"{value}\n"
    #             )


    # console.log(
    #     "saved fitness_log.csv and baseline_log.csv"
    # )


    # --------------------------------------------------------
    # RANDOM SEARCH SUMMARY
    # --------------------------------------------------------

    baseline_final = []
    for run in baseline_runs:
        baseline_final.append(run[-1])

    baseline_mean = np.mean(baseline_final)


    # --------------------------------------------------------
    # FITNESS PLOT
    # --------------------------------------------------------

    plot_fitness_curves(
        DATA / "fitness_vs_generation.png",
        baseline_mean,
    )


    # --------------------------------------------------------
    # FIND BEST BODY ACROSS BOTH VARIANTS
    # --------------------------------------------------------

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


    # ------------------------zz--------------------------------
    # VISUALIZE BEST BODY
    # --------------------------------------------------------

    # visualize_best(
    #     best_genotype,
    #     file_name="best_body",
    # )


if __name__ == "__main__":
    main()