
from ariel.ec import Population
from pathlib import Path
import networkx as nx
import copy

from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from tree_edit_distance import mean_plus_std_tree_edit_distance
from ariel.ec.genotypes.tree.operators import random_tree

HERE = Path(__file__).parent
TARGET_DIR = HERE / "target_bodies"
best_individual = None
evolution_count=0



def decode_individual(individual):
    """Turn one individual's stored genotype dict into a phenotype graph."""
    genome = TreeGenome.from_dict(individual.genotype)
    return genome.to_networkx()


def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load all 5 target body graphs."""
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


def score_individual(individual, targets: list[nx.DiGraph]) -> float:
    """Decode an individual and score it against the target set. Lower is better."""
    phenotype = decode_individual(individual)
    return mean_plus_std_tree_edit_distance(phenotype, targets)


TARGETS = load_targets()


def evaluate(population: Population) -> Population:
    to_eval = [ind for ind in population if ind.alive and ind.requires_eval]
    for ind in to_eval:
        ind.fitness = score_individual(ind, TARGETS)
        global evolution_count
        evolution_count += 1
    return population

LOG: list[dict] = []
CURRENT_SEED: int | None = None


def log_generation(population: Population) -> Population:
    fitnesses = [ind.fitness for ind in population if ind.alive]
    LOG.append({
        "seed": CURRENT_SEED,
        "best": min(fitnesses),
        "mean": sum(fitnesses) / len(fitnesses),
        "worst": max(fitnesses),
    })
    return population

def random_search_baseline(
    num_evaluations: int,
    targets: list[nx.DiGraph],
    max_modules: int = 20,
) -> list[float]:
    best_so_far = []
    best = float("inf")
    for _ in range(num_evaluations):
        genome = random_tree(max_modules)
        body = genome.to_networkx()
        fitness_value = mean_plus_std_tree_edit_distance(body, targets)
        best = min(best, fitness_value)
        best_so_far.append(best)
    return best_so_far



def save_log_csv(path="fitness_log.csv"):
    with open(path, "w") as f:
        f.write("seed,best,mean,worst\n")
        for row in LOG:
            f.write(f"{row['seed']},{row['best']},{row['mean']},{row['worst']}\n")


def save_baseline_csv(baseline_results, path="baseline_log.csv"):
    with open(path, "w") as f:
        f.write("evaluation,best_so_far\n")
        for i, value in enumerate(baseline_results):
            f.write(f"{i+1},{value}\n")



def save_best_individual(population: Population) -> Population:
    alive = []
    global best_individual
    for ind in population:
        if ind.alive and ind.fitness is not None:
            alive.append(ind)


    best = min(alive, key=lambda ind: ind.fitness)

    best_individual = {
        "fitness": float(best.fitness),
        "genotype": copy.deepcopy(best.genotype),
    }

    return population