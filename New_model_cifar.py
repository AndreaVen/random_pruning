# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:07:50 2020

@author: andrea
"""


#import pandas as pd

from __future__ import print_function
from sklearn.model_selection import train_test_split

import os
import numpy as np
import pickle
import h5py
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from cifar100_reduced_dataset import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping #, TensorBoard


classes = 100
current_path = os.path.join(os.getcwd(), 'current_model')
#print(current_path)
#model_name = 'cifar100.h5'
(x_train, y_train) , (x_test, y_test) = cifar100.load_data()
x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, test_size=0.2, random_state=42)

print('x_train_dims : ' , x_train.shape)
print('x_test_dims : ', x_test.shape)
print('y_train_dims : ', y_train.shape)
print('y_test_dims : ', y_test.shape)

print( 'number of training examples available : ', x_train.shape[0])
print('number of testing examples available : ', x_test.shape[0])

y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)
y_val = keras.utils.to_categorical(y_val, classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

x_train /= 255.0
x_test /= 255.0
x_val /= 255.0
model = Sequential()
# layer one
model.add(Conv2D(128,(3,3), padding = 'same', input_shape = x_train.shape[1:]))
model.add(Activation('elu'))
model.add(Conv2D(128, (3,3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#layer two
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('elu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#layer three
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('elu'))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(classes))
model.add(Activation("softmax"))


opt = keras.optimizers.rmsprop(lr=0.0001, decay = 1e-6)
early = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=20, verbose=1, mode='auto')
check_name='G:\\cifar100_nuovo_modello_best_TMP_REDUCED.h5'
checkpoint = ModelCheckpoint(check_name,monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
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




model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics = ['accuracy'])






epochs = 120
data_augmentation = False
predictions = 20
batch_size = 256
validation = []
model.load_weights('R:\\nuovo_modello.h5')

for i in  range(epochs):
    if not data_augmentation:
        print("************We are not using Data Augmentation************")
        model.fit(x_train, y_train, batch_size= batch_size, epochs = epochs, validation_data = (x_val, y_val),callbacks=[checkpoint], shuffle = True)
    else:
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val),callbacks=[checkpoint],verbose=1)
        
        
        
       
        

model.pop()
model.pop()
model.add(Dense(3))
model.add(Activation('softmax'))

path_to_files='R:\\mygithub\\random_pruning\\matrix_\\'
x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced=cifar100_reduced_dataset(path_to_files)
x_train_reduced /= 255.0
x_test_reduced /= 255.0
x_val_reduced /= 255.0


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
datagen.fit(x_train_reduced)

model.fit_generator(datagen.flow(x_train_reduced, y_train_reduced,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(x_val_reduced, y_val_reduced),callbacks=[checkpoint],verbose=1)

score=model.evaluate(x_test,y_test)

model.save('R:\\nuovo_modello.h5')



      