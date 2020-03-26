
"""
Created on Mon Jan 27 19:08:01 2020

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
# import matplotlib.pyplot as plt
import time
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping #, TensorBoard
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
import os 
import random
from cifar100_reduced_dataset import *

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1),
    device_count = {'GPU': 1}
)

from numexpr.utils import set_vml_num_threads

set_vml_num_threads(16)
load_flag=0 # set to 0 to prune, set to 1 to load the result 
laptop=0
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
            batch_size = 128
            maxepoches = 150
            learning_rate = 0.01
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
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_val = x_val.astype('float32')
        x_train, x_test,x_val = self.normalize(x_train, x_test,x_val)
        # y_train = keras.utils.to_categorical(y_train, num_classes)
        # y_test = keras.utils.to_categorical(y_test, num_classes)
        my_score=self.model.evaluate(x_test,y_test)
        return my_score

        score=self.model.evaluate(x_test,y_test)
        return score
    def save(self,path):
         self.model.save(path)
    def load(self,path):
        self.model=load_model(path)
    def selection_pruning(self,Vec):
        vec=[]
        for idx,i in enumerate(Vec):
            if i==1:
                vec.append(idx)
        
        #this function takes a vector of indexes, if there is a 1 in the j-th position the j-th filter of the network will be pruned.
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
        
        
        sumlist=[0]*len(n_filters) 
        for idx in range(len(n_filters)):
            sumlist[idx]=sum(n_filters[0:idx+1])   
            
        # print(sumlist)
        # print('vec=',vec)
        mega_list=[[],[],[],[],[],[],[],[],[],[],[],[],[]]
        for num in vec:
            for idx,i in enumerate(sumlist):
                if num-i<0:
                    strato=idx
                    if idx==0:
                        numero=num
                    else:
                        numero=num-sumlist[idx-1]
                    # print('strato=',idx,'numero=',numero)
                    mega_list[idx].append(numero)
         
                    break
                   
            # print('dimensione di mega list=',len(mega_list))
        for kk in range(len(to_prune_list)): # iterate on each convolutional layer 
            # of the convolutional filters 
            layerr=self.model.layers[to_prune_list[kk]]
            # print('len di mega_list[kk]=',len(mega_list[kk]))
            surgeon.add_job('delete_channels', layerr, channels=mega_list[kk]) 
            
        self.model = surgeon.operate()
        return mega_list
       
        
    def random_pruning(self,matrix,P,default=[]):
        
        from kerassurgeon import Surgeon
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
        if default==[]:
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
        else:
            print('using passed filters to prune')
            for idx,i in enumerate(default):
                layerr=self.model.layers[to_prune_list[idx]]
                surgeon.add_job('delete_channels', layerr, channels=i) 
        self.model = surgeon.operate()
        matrix.append(mega_list)
        return matrix
    def train(self,x_train,y_train,x_test,y_test,x_val,y_val,num_classes):
        # self.model.load_weights('H:\\DOWNLOAD2\\cifar100vgg.h5') #base model pre trained on 100 classes
        #training parameters
        batch_size = 256
        maxepoches = 150
        learning_rate = 0.01
        lr_decay = 1e-6
        lr_drop = 40
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_val = x_val.astype('float32')
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
        early = EarlyStopping(monitor='val_acc', min_delta=0.005, patience=40, verbose=1, mode='auto')
        # training process in a for loop with learning rate drop every 25 epoches.
        check_name='G:\\cifar100_best_TMP.h5'
        checkpoint = ModelCheckpoint(check_name,monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        historytemp = self.model.fit_generator(datagen.flow(x_train, y_train,
                            batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches, validation_data=(x_val, y_val),
                            callbacks=[early,checkpoint],verbose=1)
        
        # reduce_lr,
        self.model.load_weights(check_name)
        score_val=self.model.evaluate(x_val,y_val)
        score_test=self.model.evaluate(x_test,y_test)
        # model.save_weights('cifar100vgg.h5')
        
        print('score_val={},score_terst={}'.format(score_val,score_test))
        return score_val[1],score_test[1]

   
        
    
    
    
    



# y_train_reduced = keras.utils.to_categorical(y_train_reduced, 100)
# y_test_reduced = keras.utils.to_categorical(y_test_reduced, 100)
# if the load flag is set to 0 the script will simply load the matrices, else it will to N pruning 
#generation with a P% pruning rate in every convolutional layer 
PP=[0.8]


# P=0.4 # pruning rate 
n=4223
path_to_files='R:\\mygithub\\random_pruning\\matrix_\\'
if laptop==1:
    path_to_files='C:\\mygithub\\random_pruning\\matrix_\\'

result=[]
x_train_reduced2,y_train_reduced2,x_test_reduced2,y_test_reduced2,x_val_reduced2,y_val_reduced2=cifar100_reduced_dataset(path_to_files)
for P in PP:
    if not load_flag:
        for i in range(200):
            # model.train(model,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
            try:
        
                result=np.load(path_to_files+'random_{}_.npy'.format(P*100))
            except:
                print('no files to load')
           
            print('iteration number:',i)
            model = cifar100vgg(train=0) # build the model
            
            if laptop==1:
                model.load('C:\\mygithub\\random_pruning\\cifar100_baseline.h5')
            else:
                model.load('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5') #load the pre trained model, change the path to the saved model
        
            # score=model.evaluate(x_train_reduced2,y_train_reduced2,x_test_reduced2,y_test_reduced2,x_val_reduced2,y_val_reduced2,3)
            tmp=[0 for i in range(n)]
            pruning_idx=random.sample(range(n),int(P*n))
            for i in pruning_idx:
                tmp[i]=1
                
            _=model.selection_pruning(tmp) # 
            val_acc,test_acc=model.train(x_train_reduced2,y_train_reduced2,x_test_reduced2,y_test_reduced2,x_val_reduced2,y_val_reduced2,3)
            result.append([tmp,val_acc,test_acc])
            # matrix[-1].append(score)
            del model # delete the model and clear the memory 
            K.clear_session()
            tf.reset_default_graph()
            # np.save(path_to_files+'random_{}_.npy'.format(P*100),result)
            
    #     path='R:\\matrix_{}'.format(int(P*100)) #path to the saved matrix folder 
    #     if not os.path.exists(path):
    #         os.mkdir(path)    
    #     # np.save(path+'\\'+'matrix_{}_{}'.format(int(P*100),P,time.time()),matrix) # save the matrix containing all the filters and score to file 
        
    #     filters=[]
    #     result=[]
    #     for i in os.listdir(path):
    #         final_matrix=np.load(os.path.join(path,i),allow_pickle=True)
    #         for k in range(len(final_matrix)): # ciclo su tutte le prove
    #             filters.append(final_matrix[k][0:-2])
    #             result.append(final_matrix[k][-1])
    #     idx=np.array(result)
    #     idx=idx[:,-1]
    #     idx=(-idx).argsort()
    #     filterS=np.array(filters)[idx]
    #     resulT=np.array(result)[idx]
    #     new_path=path.replace('{}'.format(int(P*100)),'')
    #     if not os.path.exists(new_path):
    #         os.mkdir(new_path)
    #     np.save(new_path+'\\'+'result_{}'.format(P),resulT)
    #     np.save(new_path+'\\'+'filters_{}'.format(P),filterS)
        
    # # matplotlib histogram
    # else:
    #     resulT=np.load('\\matrix_\\result_{}.npy'.format(P))
    # resulT=[]
    # for i in random_matrix_05:
        
    #     resulT.append(i[1])
    # plt.hist(resulT, color = 'blue', edgecolor = 'black',
    #           bins = 20)
    # plt.show()
    # # Add labels
    # plt.title('Distribution accuracy on random pruning')
    # plt.xlabel('Accuracy on test set')
    # plt.ylabel('Number of CNN generated')
    
    # print('numero campioni:',len(resulT))
# np.save('R:\\random_05baseline85.npy',random_matrix_05)
# np.save('R:\\random_08baseline85.npy',random_matrix_08)
# baseline_model = cifar100vgg(train=0) # build the model
# baseline_model.load('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5') #load the pre trained model, change the path to the saved model

# my_pred=baseline_model.predict(x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
# y_pred=np.array([np.argmax(i) for i in my_pred])

# tmp=0
# for i,j in zip(y_pred,y_test_reduced):
#     if i==j:
#         tmp+=1
# my_accuracy=tmp/len(y_test_reduced)



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



# TP=[0,0,0]
# for i,j in zip(y_pred,y_test_reduced):
#     if i==j:
#         TP[int(i)]+=1

# FP=[0,0,0]
# FN=[0,0,0]
# for i in range(3):
#     for kk in y_pred:
#         if (int(y_pred)==i) and (int(y_test_reduced)==j) and (i!=j):
#             #j=0,i=1 pred 1 quando deve essere 0
            
    
    

# #rows=true label, columns= predicted values
# sklearn.metrics.confusion_matrix(y_test_reduced,y_pred)
