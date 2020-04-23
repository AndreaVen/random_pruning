#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:35:31 2020

@author: Andrea Venditti
Versione parallelizzata dell'algoritmo genetico, funzionamento teoricamente identico 
versione CPU: legge da disco la lsta di reti da creare e le inizializza effettuando il pruning, una volta che la rete è "potata" il modello
così ridotto è salvato su disco, la parte GPU carica il modello effetua il training e la valutazione. 
"""
from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


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
from cifar100VGG import cifar100vgg # import the architecture and all the modules 
import random
from deap import tools     
import itertools
# from numexpr.utils import set_vml_num_threads
import time
# config = tf.ConfigProto( device_count = {'GPU': 88 , 'CPU': 1} )  
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)
debug=1

# config = tf.ConfigProto(
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1),
#     device_count = {'GPU': 1}
# )
server=0

if server==1:
    operation_path='/home/labrizzi/Scaricati/random_pruning-master/'
else :
    operation_path='G:\\pruning_genetico'
        

    
### cpu part
while not os.path.isfile(os.path.join(operation_path,'end_job.npy')):
    
  

    job_to_do='X'      
    
    while not os.path.isfile(os.path.join(operation_path,job_to_do)): # aspetta fino a quando il file non è stato generato
        lista_files=os.listdir(operation_path)
        if debug==1:
            print('cerco lavoro...')
        for i in lista_files:
            if 'job_to_do' in i :
                 job_to_do=i
                 break
        time.sleep(2)
    if debug==1:
        print('lavoro trovato, leggo files...')
    time.sleep(2)
    pop_list=np.load(os.path.join(operation_path,job_to_do),allow_pickle=True)# carico la lista di lavoro
    job_to_do=job_to_do.replace('.npy','')
    var=job_to_do[-3::]
    if debug==1:
        print('var=',var)
    os.remove(os.path.join(operation_path,job_to_do+'.npy')) # ho letto la lista del lavoro da fare, quindi la elimino

    for idx,i in enumerate(pop_list):
         model = cifar100vgg(train=0)
         

         model.load(os.path.join(operation_path,'cifar100_baseline.h5')) #load the pre trained model, change the path to the saved model

             
         _=model.selection_pruning(i)
         model.give_name(str(i)) # i è un individuo che è stato castato a lista, tuttavia funziona list(individuo)==individuo
         model.save(os.path.join(operation_path,'job_{}_{}.h5'.format(var,idx)))

         del model # delete the model and clear the memory 
         K.clear_session()
         tf.reset_default_graph()
 
os.remove(os.path.join(operation_path,'end_job.npy'))