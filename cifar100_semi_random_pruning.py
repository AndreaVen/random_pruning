
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
import matplotlib.pyplot as plt
import time
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping #, TensorBoard
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
import sklearn.metrics
import os 

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
            batch_size = 128
            maxepoches = 100
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


    def normalize(self,X_train,X_test):
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
        return X_train, X_test

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
    def evaluate(self,x_train,y_train,x_test,y_test,num_classes):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
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
       
            # self.model.save('G:\\semi_random.h5')
            # del model # delete the model and clear the memory 
            # K.clear_session()
            # tf.reset_default_graph()
            
            
        # self.model = surgeon.operate()
      
      
        
    def train(self,model,x_train,y_train,x_test,y_test,num_classes):
        # self.model.load_weights('H:\\DOWNLOAD2\\cifar100vgg.h5')
        #training parameters
        batch_size = 256
        maxepoches = 100
        learning_rate = 0.001
        lr_decay = 1e-6
        lr_drop = 40


        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)
        y_train = keras.utils.to_categorical(y_train,num_classes) 
        y_test = keras.utils.to_categorical(y_test,num_classes)

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
        early = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=12, verbose=1, mode='auto')
        # training process in a for loop with learning rate drop every 25 epoches.
        check_name='G:\\cifar100_best_TMP.h5'
        checkpoint = ModelCheckpoint(check_name,monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        historytemp = self.model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[early,checkpoint],verbose=0)
        # reduce_lr,
        self.model.load_weights(check_name)
        self.score=self.model.evaluate(x_test,y_test)
        
        # model.save_weights('cifar100vgg.h5')
        print('score=',self.score)
        return model,self.score
   
        
    
    
    
    
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 100)
    y_test = keras.utils.to_categorical(y_test, 100)
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
#load dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
#semplified list of class
usefull_classes=['man','woman','wolf']
tmp=[ CIFAR100_LABELS_LIST.index(i) for i in usefull_classes]
x_test_reduced=[]
y_test_reduced=[]
x_train_reduced=[]
y_train_reduced=[]
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


x_test_reduced=np.array(x_test_reduced)
x_train_reduced=np.array(x_train_reduced)
y_train_reduced=np.array(y_train_reduced)
y_test_reduced=np.array(y_test_reduced)


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

# y_train_reduced = keras.utils.to_categorical(y_train_reduced, 100)
# y_test_reduced = keras.utils.to_categorical(y_test_reduced, 100)
# if the load flag is set to 0 the script will simply load the matrices, else it will to N pruning 
#generation with a P% pruning rate in every convolutional layer 
PP=[0.8]
# P=0.4 # pruning rate 
P=.8
for P in PP:
    
    if not load_flag:
        matrix=[]
      
        # model.train(model,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
        for i in range(200):
           
            print('iteration number:',i)
            model = cifar100vgg(train=0) # build the model
            model.load('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5') #load the pre trained model, change the path to the saved model
            matrix=model.random_pruning(matrix,P)
            _,score=model.train(model,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
            matrix[-1].append(score)
            del model # delete the model and clear the memory 
            K.clear_session()
            tf.reset_default_graph()
            
            
            
        path='R:\\matrix_{}'.format(int(P*100)) #path to the saved matrix folder 
        if not os.path.exists(path):
            os.mkdir(path)    
        # np.save(path+'\\'+'matrix_{}_{}'.format(int(P*100),P,time.time()),matrix) # save the matrix containing all the filters and score to file 
        
        filters=[]
        result=[]
        for i in os.listdir(path):
            final_matrix=np.load(os.path.join(path,i),allow_pickle=True)
            for k in range(len(final_matrix)): # ciclo su tutte le prove
                filters.append(final_matrix[k][0:-2])
                result.append(final_matrix[k][-1])
        idx=np.array(result)
        idx=idx[:,-1]
        idx=(-idx).argsort()
        filterS=np.array(filters)[idx]
        resulT=np.array(result)[idx]
        new_path=path.replace('{}'.format(int(P*100)),'')
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        np.save(new_path+'\\'+'result_{}'.format(P),resulT)
        np.save(new_path+'\\'+'filters_{}'.format(P),filterS)
        
    # matplotlib histogram
    else:
        # data={'accuracy':[],'cm':[]}
        # P=0.8
        # for iii in range(50):
        #     model = cifar100vgg(train=0) # build the model
        #     model.load('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5') #load the pre trained model, change the path to the saved model
        #     print('iterazione numero:',iii)
        #     network_score=[]
        #     for kk in range(12,-1,-1):
        #         print(f'generazione numero:{iii}, strato numero:{kk}/12')
        #         model.semi_random_pruning(P,kk)
        #         _,score=model.train(model,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
        #         network_score.append(score)
        #         model.save('G:\\semi_random.h5')
        #         del model # delete the model and clear the memory 
        #         K.clear_session()
        #         tf.reset_default_graph()
        #         model = cifar100vgg(train=0) # build the model
        #         model.load('G:\\semi_random.h5')
        #     my_pred=model.predict(x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
        #     y_pred=np.array([np.argmax(i) for i in my_pred])
        #     del model # delete the model and clear the memory 
        #     K.clear_session()
        #     tf.reset_default_graph()
        #     data['accuracy'].append(network_score)    
        #     try:
        #        cm=sklearn.metrics.confusion_matrix(y_test_reduced,y_pred)
        #        data['cm'].append(cm)
        #     except Exception:
        #         pass
            
        # np.save('R:\\matrix_\\data_0.8{}.npy'.format(time.time()),data)
        
        data={'accuracy':[],'cm':[],'recall':[],'precision':[]}

        P=0.8
        must_train=1
        
        for iii in range(50):
            model = cifar100vgg(train=0) # build the model
            model.load('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5') #load the pre trained model, change the path to the saved model
            print('iterazione numero:',iii)
            network_score=[]
            for kk in range(0,12):
                print(f'generazione numero:{iii}, strato numero:{kk}/12')
                model.semi_random_pruning(P,kk)
                if not must_train:  
                    score=model.evaluate(x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
                    if score[1]<0.7:
                        
                        _,score=model.train(model,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
                
                        network_score.append(score)
                    else:   
                        network_score.append([0,0])
                elif must_train==1:
                    _,score=model.train(model,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
                    network_score.append(score)
                    
                model.save('G:\\semi_random.h5')
                del model # delete the model and clear the memory 
                K.clear_session()
                tf.reset_default_graph()
                model = cifar100vgg(train=0) # build the model
                model.load('G:\\semi_random.h5')
            my_pred=model.predict(x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
            y_pred=np.array([np.argmax(i) for i in my_pred])
            del model # delete the model and clear the memory 
            K.clear_session()
            tf.reset_default_graph()
            data['accuracy'].append(network_score)    
            try:
               cm=sklearn.metrics.confusion_matrix(y_test_reduced,y_pred)
               data['cm'].append(cm)
            except Exception:
                pass
        
        
        
        for a in data['cm']:
            recall=[0,0,0]
            precision=[0,0,0]
            for k in range(len(recall)):
                recall[k]=a[k,k]/sum(a[:,k]) # recall=true positive/(true positives + false negative)
                precision[k]=a[k,k]/sum(a[k,:]) #precision=true positive/(true positive + false positive)
              # tmp=(sum(np.diag(a))-a[k,k])/((np.diag(a)-a[k,k])+sum(a[k,:]))
              # J=np.mean(recall+ tmp-1)
              # Informdeness # J=sensitivity+specificity-1 == TP/(TP+FN)+ TN/(TN+FP)-1 ==recal+TN/(TN+FP)
            Precision=np.mean(precision)
            Recall=np.mean(recall)
            data['precision'].append(Precision)
            data['recall'].append(recall)
        
        np.save('R:\\matrix_\\data_{}_alwaystrain_{}_in_order_{}'.format(P,must_train,time.time()),data)
        
        
        
        
        
        data1=np.load('R:\\matrix_\\data_0.8_notrain.npy',allow_pickle=True)
        data1=data1.item()
        data2=np.load('R:\\matrix_\\data_0.8_alwaystrain_0__1582899140.798901.npy',allow_pickle=True)
        data2=data2.item()
        data=[]
        for i in data1['accuracy']:
            data.append(i)
            
        for i in data2['accuracy']:
            data.append(i)
            
        


        
       
    resulT=np.array(data)  
    # data=np.load('\\matrix_\\data_0.8_notrain.npy',allow_pickle=True)
    # data=data.item()
    # resulT=np.array(data['accuracy'])   
      
    plt.hist(resulT[:,-1,-1], color = 'blue', edgecolor = 'black',
              bins = 30)
    plt.show()
    # Add labels
    plt.title('Distribution accuracy on semi-random pruning, always train ')
    plt.xlabel('Accuracy on test set')
    plt.ylabel('Number of CNN generated')
    
    print('numero campioni:',len(resulT))

# baseline_model = cifar100vgg(train=0) # build the model
# baseline_model.load('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5') #load the pre trained model, change the path to the saved model
# _,n_filters=baseline_model.semi_random_pruning([[]],0.1)


# tmp=0
# for i,j in zip(y_pred,y_test_reduced):
#     if i==j:
#         tmp+=1
# my_accuracy=tmp/len(y_test_reduced)



# y_pred=np.array([np.argmax(i) for i in my_pred])
# acc=np.array([0,0,0])
# for i in y_pred:
#     for kk in range(3):
#         if i==kk:
#             acc[kk]+=1
# acc=acc/len(x_test_reduced)       
# print(acc)
# print('mean acc:',acc.mean())
# my_score=model.evaluate(x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
# print(my_score)



# TP=[0,0,0]
# for i,j in zip(y_pred,y_test_reduced):
#     if i==j:
#         TP[int(i)]+=1


# #rows=true label, columns= predicted values




















bins = np.linspace(0.45, 0.85, 100)
plt.title('Distribution accuracy on semi-random pruning, always train vs train if val<0.7')

plt.hist(result, bins, alpha=0.7, label='always train',color = 'red')
plt.hist(result2, bins, alpha=0.4, label='if acc <0.7',color = 'black')
plt.legend(loc='upper right')
plt.show()