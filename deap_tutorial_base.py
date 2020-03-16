# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:37:25 2020

@author: andrea
"""
from deap import base, creator
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

import random
from deap import tools

IND_SIZE = 10

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE) # la dimensione dell'individuo va definita a priori!


toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE) # la dimensione dell'individuo va definita a priori!







toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    return sum(individual),

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3) # selTournament prende i migliori k(nel caso non è specificato sono tutti)
# da un torneo di tournsize
toolbox.register("evaluate", evaluate)



pop = toolbox.population(n=50) # popolazione, numero di elementi che mi ritrovo dall'inizio alla fine 
CXPB, MUTPB, NGEN = 0.5, 0.2, 40 # probabilità dicrossover 0,5, prob di mutazione 20%,40 gen

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop)) # lista di fitness, lunga quanto il numero di individui nella popolazione 
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit # fit è una tupla di dimensione 1 

for g in range(NGEN): #si cicla sul numero di generazioni
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop)) # dalla popolazione faccio un torneo della lunghezza di toursize in modo casuale, attenzione che 
    # se avessi messo esplicitamente toursize=X mi avrebbe chiesto anche il valore di k 
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))# superfluo?

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]): #off da 0 a 49 di 2 in 2, e da 1 a 50 di 2 in due, le due liste sono quindi (0,1),(2,3) ecc
        if random.random() < CXPB: 
            toolbox.mate(child1, child2) 
            del child1.fitness.values # rimuove effettivamente i valori della fitness dal offspring 
            del child2.fitness.values

    for mutant in offspring: # adesso in offspring ci sono sia gli individui vecchi che quelli accoppiati contraddistinti dall'assenza di fitness.values
        
        if random.random() < MUTPB:
            toolbox.mutate(mutant) # muta chiunque, anche quelli già mutati
            del mutant.fitness.values # elimina anche qui la fitness, è possibile eliminare più volte un campo vuoto, basta che sa a chi ti stai riferendo

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring


def innt():
    return random.randint(0,4)

tools.initRepeat(list, innt,2)