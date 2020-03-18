# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 11:00:49 2020

@author: andrea
"""
import os
import numpy as np
def cifar100_reduced_dataset(path):

    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.exists(path+'x_train_reduced.npy'): #check if the dataset has already been generated
        print('file not in folder: generating dataset')
        from sklearn.model_selection import train_test_split
        from keras.datasets import cifar100
        import keras

        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, test_size=0.2, random_state=42)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_val=x_val.astype('float32')
        # y_train = keras.utils.to_categorical(y_train, 100)
        # y_test = keras.utils.to_categorical(y_test, 100)
        # y_val=keras.utils.to_categorical(y_val, 100)
        #list of cifar 100 classes
        CIFAR100_LABELS_LIST = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
        ]

    
   
        #semplified list of class
        usefull_classes=['man','woman','wolf']
        tmp=[ CIFAR100_LABELS_LIST.index(i) for i in usefull_classes]
        x_test_reduced=[]
        y_test_reduced=[]
        x_train_reduced=[]
        y_train_reduced=[]
        x_val_reduced=[]
        y_val_reduced=[]
        new_index=0
        
        # now removing all the unnecessary patterns
        for i in tmp:
            for k in range(len(x_test)):
                if y_test[k]==i:
                    x_test_reduced.append(x_test[k])
                    y_test_reduced.append(np.array([new_index]))
                
            new_index+=1
        new_index=0
        for i in tmp:
            for k in range(len(x_train)):
                if y_train[k]==i:
                    x_train_reduced.append(x_train[k])
                    y_train_reduced.append(np.array([new_index]))
                    
        
            new_index+=1
            
        new_index=0
        for i in tmp:
            for k in range(len(x_val)):
                if y_val[k]==i:
                    x_val_reduced.append(x_val[k])
                    y_val_reduced.append(np.array([new_index]))
                    
        
            new_index+=1
        
        
        
        x_test_reduced=np.array(x_test_reduced)
        x_train_reduced=np.array(x_train_reduced)
        x_val_reduced=np.array(x_val_reduced)
        
        y_train_reduced=np.array(y_train_reduced)
        y_test_reduced=np.array(y_test_reduced)
        y_val_reduced=np.array(y_val_reduced)


        #check for 
        count_index_test=[]
        for i in y_train_reduced:
            if i not in count_index_test:
                count_index_test.append(i)        
        
        count_index_train=[]
        for i in y_train_reduced:
            if i not in count_index_train:
                count_index_train.append(i)        
        if len(count_index_test)!=3 or len(count_index_train)!=3:
            print('error')
        y_train_reduced = keras.utils.to_categorical(y_train_reduced, 3)
        y_test_reduced = keras.utils.to_categorical(y_test_reduced, 3)
        y_val_reduced = keras.utils.to_categorical(y_val_reduced, 3)
        np.save(path+'x_train_reduced.npy',x_train_reduced)  
        np.save(path+'x_test_reduced.npy',x_test_reduced)  
        np.save(path+'x_val_reduced.npy',x_val_reduced)  
        np.save(path+'y_train_reduced.npy',y_train_reduced)  
        np.save(path+'y_val_reduced.npy',y_val_reduced)  
        np.save(path+'y_test_reduced.npy',y_test_reduced)  
        
    else:
        print('loading files')
        x_test_reduced=np.load(path+'x_test_reduced.npy')
        x_train_reduced=np.load(path+'x_train_reduced.npy')
        x_val_reduced=np.load(path+'x_val_reduced.npy')
        y_test_reduced=np.load(path+'y_test_reduced.npy')
        y_train_reduced=np.load(path+'y_train_reduced.npy')
        y_val_reduced=np.load(path+'y_val_reduced.npy')
    return x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced
        