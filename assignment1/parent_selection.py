from ariel.ec import Population

def parent_selection(population: Population) -> Population:
    
    sorted_pop = population.sort(sort="min")
    cutoff = len(sorted_pop) // 2
    for i, ind in enumerate(sorted_pop):
        ind.tags = {"ps": i < cutoff}
    return sorted_pop