# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:06:47 2020

@author: andrea
"""




from __future__ import print_function
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
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
import sklearn.metrics
import os 
from cifar100_reduced_dataset import *
import random
MATRIX=[]
load_flag=1 # set to 0 to prune, set to 1 to load the result 
from cifar100VGG import cifar100vgg # import the architecture and all the modules 
import random
from deap import tools     
import itertools
from numexpr.utils import set_vml_num_threads
import time
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1),
    device_count = {'GPU': 1}
)

set_vml_num_threads(16)
from datetime import datetime


path_to_files='R:\\mygithub\\random_pruning\\matrix_\\'
x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced=cifar100_reduced_dataset(path_to_files)

data={}
if __name__ == '__main__':
    P=0.8
    MATRIX=[]
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
    toolbox.register("indices", pruning_indexes, 4223, P)
    toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
 
    def evaluate(individual,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,MATRIX):
        model = cifar100vgg(train=0) # build the model
        model.load('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5') #load the pre trained model, change the path to the saved model
        _=model.selection_pruning(individual) # 

        score_val,score_test=model.train(x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,3)

        total_score=0.6*score_val+0.4*(sum(individual))
        # total_score=5
        # score_val=2
        # score_test=3
        MATRIX.append([individual,score_val,score_test,total_score])
        # gen_matrix.append([mega_list,score,sum(individual),total_score])
        del model # delete the model and clear the memory 
        K.clear_session()
        tf.reset_default_graph()
        return total_score,MATRIX
    
    toolbox.register("evaluate", evaluate)
    # a,MATRIX=toolbox.evaluate(pop[1],x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,MATRIX)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1) # OK
    toolbox.register("select_b", tools.selBest) #OK    # da un torneo di tournsize
    toolbox.register("select_w",tools.selWorst)
    toolbox.register("evaluate", evaluate)
    CXPB, MUTPB= 0.3, 0.3 # probabilità dicrossover 0,5, prob di mutazione 20%,40 gen
    N=30 # number of individual in population 
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
    for idx,i in enumerate(pop):
        print('individuo numero{} su {}'.format(idx,len(pop)))
        fitnesses[idx],MATRIX=toolbox.evaluate(i,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,MATRIX)
    # fitnesses = list(map(toolbox.evaluate, pop,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced)) # lista di fitness, lunga quanto il numero di individui nella popolazione 
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values =(fit,) # fit è una tupla di dimensione 1   
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
    
    n_gen=10
    for g in range(n_gen): #si cicla sul numero di generazioni
         # Select the next generation individuals
        
        best=toolbox.select_b(pop, 1) #select the best (elitism)
        print('generazione numero:',g)
        best_5=toolbox.select_b(pop,3)
        kk=crossover_best(best_5)
        offspring=pop[::]
        random.shuffle(offspring)
        offspring = list(map(toolbox.clone, offspring))# superfluo? no, c'è un passaggio per riferimento e non per copia
        for child1, child2 in zip(offspring[::2], offspring[1::2]): #off da 0 a 49 di 2 in 2, e da 1 a 50 di 2 in due, le due liste sono quindi (0,1),(2,3) ecc
            if random.random() < CXPB: 
                toolbox.mate(child1, child2) 
                del child1.fitness.values # rimuove effettivamente i valori della fitness dal offspring 
                del child2.fitness.values
        
        
        for mutant in offspring: # adesso in offspring ci sono sia gli individui vecchi che quelli accoppiati contraddistinti dall'assenza di fitness.values
            if random.random() < MUTPB:
                toolbox.mutate(mutant) # muta chiunque, anche quelli già mutati
                del mutant.fitness.values # elimina anche qui la fitness, è possibile eliminare più volte un campo vuoto, basta che sa a chi ti stai riferendo
        
        
        for i in kk : # append the forced crossover 
            offspring.append(i)
            
        offspring.append(best[0])
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        
        for idx,i in enumerate(invalid_ind):
            count_ind=count_ind+1 # count the number of times an individual has been modified (evaluated)
            fitnes,MATRIX=toolbox.evaluate(i,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,MATRIX)
            i.fitness.values=(fitnes,)
    
       
        
        
        offspring=sort_ind(offspring)
        
        pop=offspring[0:int(N*0.95)]# keeping the best 95% of N, 
        
        for i in range(N-int(N*0.95)): #generating a new 5%
            tmp=offspring[-1]
          
            for idx,_ in enumerate(tmp):
                tmp[idx]=0
            new_rate=random.randint(4,8)/10 #new pruning rate between 40% and 80%
            n=len(tmp)
            pruning_idx=random.sample(range(n),int(new_rate*n))
            for i in pruning_idx:
                tmp[i]=1
            fitnes,MATRIX=toolbox.evaluate(tmp,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,MATRIX)
            tmp.fitness.values=(fitnes,)
            pop.append(tmp)
        mega_pop.append([pop])
        # pop[:] = offspring   
        # data['{}'.format(g)]=[pop,gen_matrix]
            # The population is entirely replaced by the offspring
       

        # matrixcopia=MATRIX
        # np.save('R:\\MATRIX.npy',MATRIX)
    
        # minn=999
        # maxx=0
        # for i in MATRIX:
        #     if i[1]<minn:
        #         min=i[1]
        #     if i[1]>maxx:
        #         maxx=i[1]
        

end_time=time.time()
print('durata training={}'.format((end_time-start_time)/60)) 
now = datetime.now()



current_time = now.strftime("%D_%H_%M")
current_time=current_time.replace('/','_')
# np.save('R:\\mygithub\\random_pruning\\matrix_\\genetico_mega_pop_{}_{}.npy'.format(P*100,current_time),mega_pop)
# np.save('R:\\mygithub\\random_pruning\\matrix_\\genetico_MATRIX_{}_{}.npy'.format(P*100,current_time),MATRIX)


    
 
    
#         print('iterazione numero:',iii)
#         network_score=[]
        
            
        
#         _,score=model.train(model,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)

#         network_score.append(score)

            
#             model.save('G:\\semi_random.h5')
#             del model # delete the model and clear the memory 
#             K.clear_session()
#             tf.reset_default_graph()
#             model = cifar100vgg(train=0) # build the model
#             model.load('G:\\semi_random.h5')
#         my_pred=model._ct(x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,3)
#         y_pred=np.array([np.argmax(i) for i in my_pred])
#         del model # delete the model and clear the memory 
#         K.clear_session()
#         tf.reset_default_graph()
#         data['accuracy'].append(network_score)    
#         try:
#            cm=sklearn.metrics.confusion_matrix(y_test_reduced,y_pred)
#            data['cm'].append(cm)
#         except Exception:
#             pass
        
#         np.save('R:\\matrix_\\data_{}_alwaystrain_{}_in_order_num_{}___{}'.format(P,must_train,num,time.time()),data)
#     else:
#         data=np.load(path_to_files+'data_0.8_alwaystrain_1__1582862164.9756384.npy',allow_pickle=True)
#         data=data.item()
#     resulT=[]
#     for i in data['accuracy']:
#         resulT.append(i[-1])
#     resulT=np.array(resulT)
#         # # data=np.load('\\matrix_\\data_0.8_notrain.npy',allow_pickle=True)
#         # # data=data.item()
#         # # resulT=np.array(data['accuracy'])   
          
        # plt.hist(resulT, color = 'blue', edgecolor = 'black',bins = 30)
        # plt.show()
        # # Add labels
        # plt.title('Distribution accuracy on semi-random pruning, always train ')
        # plt.xlabel('Accuracy on test set')
        # plt.ylabel('Number of CNN generated')
        
        # print('numero campioni:',len(resulT))

































# # baseline_model = cifar100vgg(train=0) # build the model
# # baseline_model.load('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5') #load the pre trained model, change the path to the saved model
# # _,n_filters=baseline_model.semi_random_pruning([[]],0.1)


# # tmp=0
# # for i,j in zip(y_pred,y_test_reduced):
# #     if i==j:
# #         tmp+=1
# # my_accuracy=tmp/len(y_test_reduced)



# # y_pred=np.array([np.argmax(i) for i in my_pred])
# # acc=np.array([0,0,0])
# # for i in y_pred:
# #     for kk in range(3):
# #         if i==kk:
# #             acc[kk]+=1
# # acc=acc/len(x_test_reduced)       
# # print(acc)
# # print('mean acc:',acc.mean())
# # my_score=model.evaluate(x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
# # print(my_score)



# # TP=[0,0,0]
# # for i,j in zip(y_pred,y_test_reduced):
# #     if i==j:
# #         TP[int(i)]+=1


# # #rows=true label, columns= predicted values
resulT=[]
matrixcopia=MATRIX
for i in MATRIX:
    resulT.append(i[2])



bins = np.linspace(0.45, 0.8, 100)
plt.title('Distribution accuracy on semi-random pruning, always train vs train if val<0.7')

plt.hist(resulT, bins, alpha=0.7, label='always train',color = 'red')
# plt.hist(result2, bins, alpha=0.4, label='if acc <0.7',color = 'black')
plt.legend(loc='upper right')
plt.show()
       

# import time
# a=time.time()
# model = cifar100vgg(train=0) # build the model
# model.load('R:\mygithub\\'+'cifar100_baseline.h5') #load the pre trained model, change the path to the saved model
# # a=toolbox.evaluate(vec,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced)
# _,score=model.train(model,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,3)
# b=time.time()
# print(b-a)

# import numpy as np
# MATRIX=np.load('R:\\MATRIX.npy',allow_pickle=True)

# pop=list(MATRIX[0:50,0])

import numpy as np
import deap
mega=np.load('R:\\mygithub\\random_pruning\\matrix_\\genetico_mega_pop_80.0_03_23_20_10_15.npy',allow_pickle=True)
a=mega[-1]
max=0

