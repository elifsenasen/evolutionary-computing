from ariel.ec.genotypes.tree.operators import crossover_subtree
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec import Population
from ariel.ec.individual import Individual

def crossover(population: Population) -> Population:
    
    parents = population.where(lambda ind: bool(ind.tags.get("ps", False)))
    parent_list = list(parents)

    children: list[Individual] = []

    for i in range(0, len(parent_list) - 1, 2):
        parent_a = parent_list[i]
        parent_b = parent_list[i + 1]

        genome_a = TreeGenome.from_dict(parent_a.genotype)
        genome_b = TreeGenome.from_dict(parent_b.genotype)

        child_genome_a, child_genome_b = crossover_subtree(genome_a, genome_b)

        for child_genome in (child_genome_a, child_genome_b):
            child = Individual()
            child.genotype = child_genome.to_dict()
            child.tags = {"ps": False}
            children.append(child)

    population.extend(children)
    return population