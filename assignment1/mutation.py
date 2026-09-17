import random
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec import Population
from ariel.ec.genotypes.tree.operators import (
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
)

MUTATION_OPERATORS = [
    mutate_replace_node,
    mutate_subtree_replacement,
    mutate_shrink,
    mutate_hoist,
]


def mutate(population: Population, mutation_rate: float) -> Population:

    candidates = population.where(lambda ind: ind.requires_eval )

    for ind in candidates:
        if random.random() < mutation_rate:    
            genome = TreeGenome.from_dict(ind.genotype)
            operator = random.choice(MUTATION_OPERATORS)
            operator(genome)  

            ind.genotype = genome.to_dict()  
            ind.requires_eval = True
            ind.tags["mutate"] = True


    return population


def survivor_selection(population: Population, target_size: int) -> Population:
    
    sorted_population = population.sort(sort="min", attribute="fitness_")
    survivors = sorted_population[:target_size]
    for ind in sorted_population:
        ind.alive = ind in survivors

    return sorted_population.alive