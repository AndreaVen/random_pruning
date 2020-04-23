# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:06:47 2020

@author: andrea
Versione parallelizzata dell'algoritmo genetico, funzionamento teoricamente identico 

"""




from __future__ import print_function
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import load_model 
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
# from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
import time
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping #, TensorBoard
import tensorflow as tf
import sklearn
import sklearn.metrics
import os 
from cifar100_reduced_dataset import *
import random
MATRIX=[]
load_flag=1 # set to 0 to prune, set to 1 to load the result 
debug=1 # abilita o disabilita i print per debug
from cifar100VGG import cifar100vgg # import the architecture and all the modules 
import random
from deap import tools     
import itertools
# from numexpr.utils import set_vml_num_threads
import time
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1),
    device_count = {'GPU': 1}
)

# set_vml_num_threads(16)
from datetime import datetime

server=0
if server==1:
    operation_path='/home/labrizzi/Scaricati/random_pruning-master/'
    path_to_files='/home/labrizzi/Scaricati/random_pruning-master/matrix_/'
    
else :
    path_to_files='R:\\mygithub\\random_pruning\\matrix_\\'
    operation_path='G:\\pruning_genetico'
    
x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced=cifar100_reduced_dataset(path_to_files)

data={}
if __name__ == '__main__':
    P=0.4
    MATRIX={}
    from deap import base, creator
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # pesi=-1 per minimizzazione,deve essere comunque una tupla,
    # se si usano più valori diventa una ottimizzazione multiobbiettivo 
    creator.create("Individual", list, fitness=creator.FitnessMax) # un individuo è di tipo lista e come attributo ha la FitnessMin
    toolbox = base.Toolbox()
    def pruning_indexes(n,rate):
        import numpy as np # n is the length of the individual i.e the number of filters
        #rate is the rate of 1 where 1 means prune that filter and 0 means leave it
        tmp=[0 for i in range(n)]
        pruning_idx=random.sample(range(n),int(rate*n))
        for i in pruning_idx:
            tmp[i]=1
        
        return tmp
    
    def my_crossover(genitore_1,genitore_2):
        n_filters=[64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]# vero solo per questa specifica architettura
        sumlist=[0 for i in n_filters] # vettore lungo quanto il numero di strati 
        for idx in range(len(n_filters)):
            sumlist[idx]=sum(n_filters[0:idx+1]) 
        sumlist.insert(0,0)   
        figlio_1=toolbox.individual()
        figlio_2=toolbox.individual() # sono nuovi individui con una loro campo fitness
        figlio_1[::]=genitore_1[::]# è necessario copiarli senza riferimento
        figlio_2[::]=genitore_2[::]
        for idx,i in enumerate(n_filters):
            if random.random()<0.5: # lancio una moneta, se esce testa scambio quel relativo strato, altrimenti passo allo strato successivo
            
                tmp=figlio_1[sumlist[idx]:sumlist[idx]+n_filters[idx]]
                figlio_1[sumlist[idx]:sumlist[idx]+n_filters[idx]]=figlio_2[sumlist[idx]:sumlist[idx]+n_filters[idx]]
                figlio_2[sumlist[idx]:sumlist[idx]+n_filters[idx]]=tmp[::]  
        return figlio_1,figlio_2
    
    toolbox.register("my_crossover",my_crossover)
    toolbox.register("indices", pruning_indexes, 4224, P)
    toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
 
    def evaluate(individual,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,MATRIX,gen,percorso):

        
        try:
            total_score=MATRIX[str(individual)]['total_score']
            
            MATRIX[str(individual)]['gen'].append(gen)
        except:
            
        

    
            score_val,score_test=model.train(x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,3,operation_path)
    
            total_score=0.7*score_val+0.3*((sum(individual))/len(individual))
            # total_score=5
            # score_val=2
            # score_test=
            
            MATRIX[str(individual)]={'score_val':[score_val],'score_test':[score_test],'gen':[gen]}

            # gen_matrix.append([mega_list,score,sum(individual),total_score])
           
        return total_score,MATRIX
    
    toolbox.register("evaluate", evaluate)
    # a,MATRIX=toolbox.evaluate(pop[1],x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,MATRIX)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate_big", tools.mutFlipBit, indpb=0.1) # OK
    
    toolbox.register("mutate_small", tools.mutFlipBit, indpb=0.01) # OK
    
    
    
    
    toolbox.register("select_b", tools.selBest) #OK    # da un torneo di tournsize
    toolbox.register("select_w",tools.selWorst)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select_tour", tools.selTournament,k=2,tournsize=2) 
    
    N=50 # number of individual in population 
    pop = toolbox.population(n=N) # popolazione, numero di elementi che mi ritrovo dall'inizio alla fine 
    count_ind=N
    # megapop=np.load('R:\\mygithub\\random_pruning\\matrix_\\genetico_mega_pop_80.0_03_19_20_09_49.npy',allow_pickle=True)
    # pop=megapop[-1][0]
    
    def sort_ind(population):
        pop_copy=toolbox.clone(population)
        new_pop=[]
        for i in range(len(population)):
            new_b=toolbox.select_b(pop_copy,1)
            # print(new_b)
            new_pop.append(new_b[0])
            pop_copy.remove(new_b[0])
            # print(new_pop)
        return new_pop
    
    start_time=time.time()
    # Evaluate the entire population
    fitnesses=[0]*len(pop) #inizialmente lunga 50 
    start_time=time.time()
    mega_pop=[]# contain all the population for every generation, keeping track of the individuals
    #inizio operazione di parallelizzazione
    pop_list=[list(i) for i in pop]
    
 
    var=str(time.time())[-3::] # indice univoco di questo work
    np.save(os.path.join(operation_path,'job_to_do_{}.npy'.format(var)),pop_list)
    
   
    for idx,i in enumerate(pop):
        
        while not os.path.isfile(os.path.join(operation_path,'job_{}_{}.h5'.format(var,idx))):
            time.sleep(2) # aspetto che il secondo kernel effettui il pruning
            
            if debug==1:
                print('attendo svolgimento lavoro {}'.format(idx))
            
        if debug==1:
            print('lavoro {} svolto'.format(idx))   
            
        print('individuo numero{} su {}'.format(idx,len(pop)))
        model = cifar100vgg(train=0) # build the model
        
        model.load(os.path.join(operation_path,'job_{}_{}.h5'.format(var,idx)))
        ind_name=model.return_name()
        # if debug==1:
        #     if str(list(i)) !=ind_name:
        #         print('ATTENZIONE INDIVIDUO DIVERSO')
        #         time.sleep(100)
        #     elif str(list(i)) ==ind_name:
        #          print('individuo corretto')
        #          time.sleep(5)
        
        # ind_name=model.return_name()
        # indice=pop.index(ind_name)
        # print('indice=',indice)
        
        fitnesses,MATRIX=toolbox.evaluate(i,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,MATRIX,0,model)
        del model # delete the model and clear the memory 
        K.clear_session()
        tf.reset_default_graph()
        i.fitness.values=(fitnesses,)
        os.remove(os.path.join(operation_path,'job_{}_{}.h5'.format(var,idx)))
        
    # fitnesses = list(map(toolbox.evaluate, pop,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced)) # lista di fitness, lunga quanto il numero di individui nella popolazione 
   
    mega_pop.append(pop)  

    def crossover_best(list1):
        new_child=[]
        lista=list(itertools.permutations(list1,2))
        print(len(lista))
        for i in lista:
            kk=toolbox.mate(i[0],i[1])
            del kk[0].fitness.values
            del kk[1].fitness.values
            new_child.append(kk[0])
            new_child.append(kk[1])
        return new_child
    
    n_gen=15
    for g in range(1,n_gen): #si cicla sul numero di generazioni
         # Select the next generation individuals
        
        best=toolbox.clone(toolbox.select_b(pop, 1)) #select the best (elitism)
        print('generazione numero:',g)
        offspring=toolbox.clone(pop) ## utilizzare una tecnica di selezione 

        offspring=sort_ind(offspring) # pop ordinata in ordine decrescente
        nn=int(N*0.2)
        for i in range(nn):
            
            offspring.remove(offspring[-1])
        
        new_population=[]# in new population ci sono tutti i nuovi individui figli o le mutazioni
        
        
        
        offspring = list(map(toolbox.clone, offspring))# superfluo? no, c'è un passaggio per riferimento e non per copia
        # for child1, child2 in zip(offspring[::2], offspring[1::2]): #off da 0 a 49 di 2 in 2, e da 1 a 50 di 2 in due, le due liste sono quindi (0,1),(2,3) ecc
        for i in range(int(len(offspring)/2)):
            if random.random() < 0.3: #30% di probabilità di cross over  
                
                child1,child2=toolbox.clone(toolbox.select_tour(pop))# i figli vengono clonati, quindi i genitori non vengono influenzati dalla mutazione
                
                
                # if child1 not in to_survive_list:
                #     to_survive_list.append(child1)
                # if child2 not in to_survive_list:
                #     to_survive_list.append(child2)
                figlio_1,figlio_2=toolbox.my_crossover(child1,child2)
                
                del figlio_1.fitness.values # rimuove effettivamente i valori della fitness dal offspring 
                del figlio_2.fitness.values
                
                new_population.append(figlio_1) 
                new_population.append(figlio_2)

                    # registrare piccolissima prob di flip sui figli in new_population, eventualmente anche sulla popolazione migliore, 
                #applicare eventualmetne una grossa prob di fli sugli individui peggiori 
                
        
                
        for mutant in new_population:  #effettuo una piccola mutazione sulla gen k+1 i.e crossover + mutazione
            if random.random() < 0.1: #10% prb di mutazione 
                toolbox.mutate_small(mutant)
                del mutant.fitness.values # elimina anche qui la fitness, è possibile eliminare più volte un campo vuoto, basta che sa a chi ti stai riferendo
        
        
        for mutant in offspring[-10::]:  # effettuo una grande mutazione (molti bit) sulla parte peggiore della popolazione
            if random.random() < 0.1: #10% prb di mutazione 
                child=toolbox.clone(mutant)
                toolbox.mutate_big(child)

                del child.fitness.values # elimina anche qui la fitness, è possibile eliminare più volte un campo vuoto, basta che sa a chi ti stai riferendo
                new_population.append(child)#
                
                
        for mutant in offspring[0:-10]:  #effettuo una piccola mutazione sulla parte migliore della popolazione della gen k
            if random.random() < 0.1: #10% prb di mutazione 
                child=toolbox.clone(mutant)
                toolbox.mutate_small(child)
                del child.fitness.values # elimina anche qui la fitness, è possibile eliminare più volte un campo vuoto, basta che sa a chi ti stai riferendo
                new_population.append(child)#
       
        

        offspring.append(best[0])
        for i in new_population:
            offspring.append(i)
            
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        pop_list=[list(i) for i in invalid_ind]
        

        var=str(time.time())[-3::] # indice univoco di questo work
        np.save(os.path.join(operation_path,'job_to_do_{}.npy'.format(var)),pop_list)
        
        for idx,i in enumerate(invalid_ind):
            count_ind=count_ind+1 #leggermente ridondandte in quanto si può sapere qeusto numero dalla lungezza di MATRIX
        
            while not os.path.isfile(os.path.join(operation_path,'job_{}_{}.h5'.format(var,idx))):
                time.sleep(2) # aspetto che il secondo kernel effettui il pruning           
            
            # print('individuo numero{} su {}'.format(idx,len(pop)))
            model = cifar100vgg(train=0) # build the model
            model.load(os.path.join(operation_path,'job_{}_{}.h5'.format(var,idx)))
            ind_name=model.return_name()
            # if debug==1:
            #     if i !=ind_name:
            #         print('ATTENZIONE INDIVIDUO DIVERSO')
            #         time.sleep(100)
            #     elif i==ind_name:
            #          print('individuo corretto')
                     # time.sleep(5)
            # ind_name=model.return_name()
            # indice=pop.index(ind_name)
            # print('indice=',indice)
            
            fitnesses,MATRIX=toolbox.evaluate(i,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,MATRIX,g,model)
            del model # delete the model and clear the memory 
            K.clear_session()
            tf.reset_default_graph()
            i.fitness.values=(fitnesses,)
            os.remove(os.path.join(operation_path,'job_{}_{}.h5'.format(var,idx)))
        
 
        
        offspring=sort_ind(offspring)
        if len(offspring)>int(N*0.95):
            pop=offspring[0:int(N*0.95)]# se ci sono almeno 50 individui ne prendo il 
        else:
            pop=offspring[::] # re inizializzo un tot% di individui per aumentare l'esplorazione
        for i in range(N-len(pop)): #generating a new 5%
            tmp=toolbox.individual()
            
            for idx,_ in enumerate(tmp):
                tmp[idx]=0
            new_rate=random.randint(4,8)/10 #new pruning rate between 40% and 80%
            n=len(tmp)
            pruning_idx=random.sample(range(n),int(new_rate*n))
            for i in pruning_idx:
                tmp[i]=1
                
            new_random.append(tmp) 
        
        
        var=str(time.time())[-3::] # indice univoco di questo work
        np.save(os.path.join(operation_path,'job_to_do_{}.npy'.format(var)),pop_list)
        
        for idx,i in enumerate(new_random):
        
            while not os.path.isfile(os.path.join(operation_path,'job_{}_{}.h5'.format(var,idx))):
                time.sleep(2) # aspetto che il secondo kernel effettui il pruning           
            
            # print('individuo numero{} su {}'.format(idx,len(pop)))
            model = cifar100vgg(train=0) # build the model
            model.load(os.path.join(operation_path,'job_{}_{}.h5'.format(var,idx)))
            ind_name=model.return_name()
            # if debug==1:
            #     if i !=ind_name:
            #         print('ATTENZIONE INDIVIDUO DIVERSO')
            #         time.sleep(100)
            #     elif i==ind_name:
            #          print('individuo corretto')
            #          time.sleep(5)
            fitnesses,MATRIX=toolbox.evaluate(i,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,MATRIX,g,model)
            i.fitness.values=(fitnes,)
            pop.append(i)
        mega_pop.append([pop])


end_time=time.time()
print('durata training={}'.format((end_time-start_time)/60)) 
now = datetime.now()

np.save()

current_time = now.strftime("%D_%H_%M")
current_time=current_time.replace('/','_')
#MODIFICARE DIRECTORY DI SALVATAGGIO 
tmp=time.time()
np.save('/home/labrizzi/Scaricati/random_pruning-master/mega_pop_server_0.4_{}.npy'.format(tmp),mega_pop)
np.save('/home/labrizzi/Scaricati/random_pruning-master/MATRIX_server_0.4_{}.npy'.format(tmp),MATRIX)
dummy_var=0
np.save(os.path.join(operation_path,'end_job.npy',dummy_var))



