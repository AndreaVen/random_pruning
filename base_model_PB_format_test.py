# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:43:54 2020

@author: andrea
"""



"""
Created on Mon Jan 27 19:08:01 2020

@author: andrea
"""

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
from keras import backend as K
path_to_files='R:\\mygithub\\random_pruning\\matrix_\\'
x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced=cifar100_reduced_dataset(path_to_files)

load_flag=1 # set to 0 to prune, set to 1 to load the result 


class cifar100vgg:
    def __init__(self,train=False):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        if train:
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            # self.model,self.score = self.train(self.model,x_train,y_train,x_test,y_test)
            
            
        else:
            # self.model.load_weights('H:\\DOWNLOAD2\\cifar100vgg.h5') #inserire qui il percorso della rete pre-allenata
            batch_size = 128
            maxepoches = 100
            learning_rate = 0.05
            lr_decay = 1e-6
            lr_drop = 40
            self.model.pop()
            self.model.pop()
            self.model.add(Dense(3))
            self.model.add(Activation('softmax'))
            # The data, shuffled and split between train and test sets:
            # (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            # x_train = x_train.astype('float32')
            # x_test = x_test.astype('float32')
            # x_train, x_test = self.normalize(x_train, x_test)
            # y_train = keras.utils.to_categorical(y_train, self.num_classes)
            # y_test = keras.utils.to_categorical(y_test, self.num_classes)
            def lr_scheduler(epoch):
                return learning_rate * (0.5 ** (epoch // lr_drop))
            reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
            sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
            self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
            # self.score=self.model.evaluate(x_test,y_test)

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay
        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test,X_val):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        # print(mean)
        # print(std)
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        X_val = (X_val-mean)/(std+1e-7)
        return X_train, X_test,X_val

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.
        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 121.936
        std = 68.389
        return (x-mean)/(std+1e-7)

    def predict(self,x_train,y_train,x_test,y_test,num_classes):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        y_pred=self.model.predict(x_test)
        return y_pred
    
    def summary(self):
        return self.model.summary()
    def reduce(self,new_num_classes=3):
        self.model.pop()
        self.model.pop()
        self.model.add(Dense(new_num_classes))
        self.model.add(Activation('softmax'))
    def evaluate(self,x_train,y_train,x_test,y_test,x_val,y_val,num_classes):

        x_train, x_test,x_val = self.normalize(x_train, x_test,x_val)
        learning_rate = 0.001
        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
        my_score=self.model.evaluate(x_test,y_test)
        return my_score

        score=self.model.evaluate(x_test,y_test)
        return score
    def save(self,path):
         self.model.save(path)
    def load(self,path):
        self.model=load_model(path)
        
    def random_pruning(self,matrix,P):
        from kerassurgeon import Surgeon # k
        surgeon = Surgeon(self.model)
        layer_list=self.model.layers # list of layers in the model e.g [<keras.engine.input_layer.InputLayer at 0x2068f6c2048>,...
        to_prune_list=[]
        for i in range(len(layer_list)):
            if 'Conv2D' in str(layer_list[i]):
                to_prune_list.append(i) # check the indexes of the convolutional layers in the model
        n_filters=[]
        # print(to_prune_list)
        for i in to_prune_list:
            n_filters.append(self.model.get_layer(index=i).get_config()['filters']) # gets the number of filter in each convolutional layer
        # print('num,er of filters=',n_filters)
        # return n_filters
        mega_list=[]   
        for kk in range(len(to_prune_list)): # iterate on each convolutional layer 
            num_filter_layer=n_filters[kk]# number of filters in the current layer, which is to_prune_list[kk]
            random_index=[] # index of the filters to be pruned 
            for jj in range(int(np.ceil(P*num_filter_layer))): # P% of the filters 
                tmp=np.random.randint(0,num_filter_layer)
                while tmp in random_index:
                    tmp=np.random.randint(0,num_filter_layer) # if the filter has alread benn choosen, try again 
                random_index.append(tmp)
               
            mega_list.append(random_index) # contain a list of all the random index of that layer, the length of the mega list is thus the length 
            # of the convolutional filters 
            layerr=self.model.layers[to_prune_list[kk]]
            print(str(layerr))
            surgeon.add_job('delete_channels', layerr, channels=random_index) 
            
        self.model = surgeon.operate()
        matrix.append(mega_list)
        return matrix
    
    def outputs(self):
        return self.model.outputs
    
    def get_filters(self):
        layer_list=self.model.layers # list of layers in the model e.g [<keras.engine.input_layer.InputLayer at 0x2068f6c2048>,...
        to_prune_list=[]
        
        for i in range(len(layer_list)):
            if 'Conv2D' in str(layer_list[i]):
                to_prune_list.append(i) # check the indexes of the convolutional layers in the model
        n_filters=[]
        
        # print(to_prune_list)
        for i in to_prune_list:
            n_filters.append(self.model.get_layer(index=i).get_config()['filters']) # gets the number of filter in each convolutional layer
        # print('num,er of filters=',n_filters)
        # return n_filters
        mega_list=[]   
        to_prune_list.reverse() # reverse the list of filter to start at the END
        n_filters.reverse()
        return to_prune_list,n_filters
        
        
    def semi_random_pruning(self,P,kk):
        from kerassurgeon import Surgeon
        layer_list=self.model.layers # list of layers in the model e.g [<keras.engine.input_layer.InputLayer at 0x2068f6c2048>,...
        to_prune_list=[]
        
        for i in range(len(layer_list)):
            if 'Conv2D' in str(layer_list[i]):
                to_prune_list.append(i) # check the indexes of the convolutional layers in the model
        n_filters=[]
        
        print(to_prune_list)
        for i in to_prune_list:
            n_filters.append(self.model.get_layer(index=i).get_config()['filters']) # gets the number of filter in each convolutional layer
        # print('num,er of filters=',n_filters)
        # return n_filters
        mega_list=[]   
        # for kk in range(len(to_prune_list)): # iterate on each convolutional layer 
        num_filter_layer=n_filters[kk]# number of filters in the current layer, which is to_prune_list[kk]
        random_index=[] # index of the filters to be pruned 
        for jj in range(int(np.ceil(P*num_filter_layer))): # P% of the filters 
            tmp=np.random.randint(0,num_filter_layer)
            while tmp in random_index:
                tmp=np.random.randint(0,num_filter_layer) # if the filter has alread benn choosen, try again 
            random_index.append(tmp)
               
        # mega_list.append(random_index) # contain a list of all the random index of that layer, the length of the mega list is thus the length 
        # of the convolutional filters 
        layerr=self.model.layers[to_prune_list[kk]]
        print(str(layerr))
        surgeon = Surgeon(self.model)
        surgeon.add_job('delete_channels', layerr, channels=random_index) 
        self.model = surgeon.operate()

      
      
        
    def train(self,model,x_train,y_train,x_test,y_test,x_val,y_val,num_classes):
        # self.model.load_weights('H:\\DOWNLOAD2\\cifar100vgg.h5') #base model pre trained on 100 classes
        #training parameters
        batch_size = 256
        maxepoches = 200
        learning_rate = 0.001
        lr_decay = 1e-6
        lr_drop = 40
        x_train, x_test,x_val = self.normalize(x_train, x_test,x_val)
        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
        # , decay=lr_decay
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
        # NAME='cifar_100_reduced{}'.format(int(time.time()))
        # tensorboard=TensorBoard(log_dir='R:/logs/{}'.format(NAME))
        # training process in a for loop with learning rate drop every 25 epoches.
        check_name='G:\\cifar100_best_TMP.h5'
        checkpoint = ModelCheckpoint(check_name,monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        historytemp = self.model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_val, y_val),callbacks=[checkpoint],verbose=1)
        # reduce_lr,
        self.model.load_weights(check_name)
        self.score=self.model.evaluate(x_test,y_test)
        
        # model.save_weights('cifar100vgg.h5')
        print('score=',self.score)
        return model,self.score
   
        
    
    
    
    
if __name__ == '__main__':
 
    model = cifar100vgg(train=0) # build the model
    model.load('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5') #load the pre trained model, change the path to the saved model
    # score=model.evaluate(x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,x_val_reduced,y_val_reduced,3)
    # _,score=model.train(model,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,3)
    
    
    

    def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
        """
        Freezes the state of a session into a pruned computation graph.
        
        Creates a new computation graph where variable nodes are replaced by
        constants taking their current value in the session. The new graph will be
        pruned so subgraphs that are not necessary to compute the requested
        outputs are removed.
        @param session The TensorFlow session to be frozen.
        @param keep_var_names A list of variable names that should not be frozen,
                              or None to freeze all the variables in the graph.
        @param output_names Names of the relevant graph outputs.
        @param clear_devices Remove the device directives from the graph for better portability.
        @return The frozen graph definition.
        """
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph
    

    #export model in .pb format
    
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs()])


    tf.train.write_graph(frozen_graph, 'R:\\', "my_model.pb", as_text=False)