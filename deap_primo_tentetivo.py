# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:37:25 2020

@author: andrea
"""
import numpy as np
from deap import base, creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # pesi=-1 per minimizzazione,deve essere comunque una tupla,
# se si usano più valori diventa una ottimizzazione multiobbiettivo 
creator.create("Individual", list, fitness=creator.FitnessMax) # un individuo è di tipo lista e come attributo ha la FitnessMin

import random
from deap import tools


toolbox = base.Toolbox()




def pruning_indexes(n,rate):
    import numpy as np # n is the length of the individual i.e the number of filters
    #rate is the rate of 1 where 1 means prune that filter and 0 means leave it
    tmp=[0 for i in range(n)]
    pruning_idx=random.sample(range(n),int(rate*n))
    for i in pruning_idx:
        tmp[i]=1
    
    return tmp


toolbox.register("indices", pruning_indexes, 10, 0.7)



toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.indices)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)




def evaluate(individual):
    
    
    
    
    
    return sum(individual),

toolbox.register("mate", tools.cxTwoPoint)




toolbox.register("mutate", tools.mutFlipBit, indpb=0.1) # OK

toolbox.register("select_b", tools.selBest)
toolbox.register("select", tools.selTournament, tournsize=3) # selTournament prende i migliori k(nel caso non è specificato sono tutti)
# da un torneo di tournsize
toolbox.register("evaluate", evaluate)

CXPB, MUTPB, NGEN = 0.5, 0.2, 40 # probabilità dicrossover 0,5, prob di mutazione 20%,40 gen
pop = toolbox.population(n=50) # popolazione, numero di elementi che mi ritrovo dall'inizio alla fine 

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop)) # lista di fitness, lunga quanto il numero di individui nella popolazione 

for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit # fit è una tupla di dimensione 1 



for g in range(NGEN): #si cicla sul numero di generazioni
    # Select the next generation individuals

    best=toolbox.select_b(pop, 2) #select the best 2 

    # offspring = toolbox.select(pop, len(pop)) # dalla popolazione faccio un torneo della lunghezza di toursize in modo casuale, attenzione che 
    offspring=pop
    # se avessi messo esplicitamente toursize=X mi avrebbe chiesto anche il valore di k 
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))# superfluo?
    for i in best:
        offspring.remove(i)
        
    
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
    for i in best:
            
        offspring.append(i)
        # The population is entirely replaced by the offspring
        pop[:] = offspring







# child1=offspring[0]
# child2=offspring[1]
# child1copia=child1
# child2copia=child2
# toolbox.mate(child1, child2) 
# toolbox.mutate(child1)












# import random

# from deap import base
# from deap import creator
# from deap import tools

# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)

# IND_SIZE=10  
# toolbox = base.Toolbox()
# toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)



