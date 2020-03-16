
"""
Created on Mon Jan 27 19:08:01 2020

@author: andrea venditti
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
path_to_files='R:\\mygithub\\random_pruning\\matrix_\\'
x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced=cifar100_reduced_dataset(path_to_files)
load_flag=1 # set to 0 to prune, set to 1 to load the result 
from cifar100VGG import cifar100vgg # import the architecture and all the modules 
        
if __name__ == '__main__':
 
# if the load flag is set to 0 the script will simply load the matrices, else it will to N pruning 
#generation with a P% pruning rate in every convolutional layer 
    
    
    def random_pruning(P,num):  # total random pruning 
        matrix=[]
        # model.train(model,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
        for i in range(num):
            print('iteration number:',i)
            model = cifar100vgg(train=0) # build the model
            model.load('R:\mygithub\\'+'cifar100_baseline.h5') #load the pre trained model, change the path to the saved model
            matrix=model.random_pruning(matrix,P)
            score=model.evaluate(x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,3)
            _,score=model.train(model,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,3)
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
    # semi random plot
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
        
        # nuovo semi random plot, ora considera anche seè il caso di allenare oppure no 
    
    data={'accuracy':[],'cm':[],'recall':[],'precision':[]}
    P=0.8
    must_train=1
    
    # must train==0-> check if precision has decreas and then train if necessary, 
                 # must_train==1->always train 
                 # must_train==2->just load the result from file and plot them
    num=50 # number of CNN to generate
    if must_train !=2:
        for iii in range(num):
            model = cifar100vgg(train=0) # build the model
            model.load('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5') #load the pre trained model, change the path to the saved model
            print('iterazione numero:',iii)
            network_score=[]
            for kk in range(0,12):
                print(f'generazione numero:{iii}, strato numero:{kk}/12')
                model.semi_random_pruning(P,kk)
                if  must_train==0:  
                    score=model.evaluate(x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,3)
                    if score[1]<0.7:
                        
                        _,score=model.train(model,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,3)
                
                        network_score.append(score)
                    else:   
                        network_score.append([0,0])
                elif must_train==1:
                    _,score=model.train(model,x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,3)
                    network_score.append(score)
                    
                model.save('G:\\semi_random.h5')
                del model # delete the model and clear the memory 
                K.clear_session()
                tf.reset_default_graph()
                model = cifar100vgg(train=0) # build the model
                model.load('G:\\semi_random.h5')
            my_pred=model.predict(x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced,3)
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
        
        np.save('R:\\matrix_\\data_{}_alwaystrain_{}_in_order_num_{}___{}'.format(P,must_train,num,time.time()),data)
    else:
        data=np.load(path_to_files+'data_0.8_alwaystrain_1__1582862164.9756384.npy',allow_pickle=True)
        data=data.item()
    
        # recall nad precision implementation, to be modified
        # for a in data['cm']:
        #     recall=[0,0,0]
        #     precision=[0,0,0]
        #     for k in range(len(recall)):
        #         recall[k]=a[k,k]/sum(a[:,k]) # recall=true positive/(true positives + false negative)
        #         precision[k]=a[k,k]/sum(a[k,:]) #precision=true positive/(true positive + false positive)
        #       # tmp=(sum(np.diag(a))-a[k,k])/((np.diag(a)-a[k,k])+sum(a[k,:]))
        #       # J=np.mean(recall+ tmp-1)
        #       # Informdeness # J=sensitivity+specificity-1 == TP/(TP+FN)+ TN/(TN+FP)-1 ==recal+TN/(TN+FP)
        #     Precision=np.mean(precision)
        #     Recall=np.mean(recall)
        #     data['precision'].append(Precision)
        #     data['recall'].append(recall)
        
    resulT=[]
    for i in data['accuracy']:
        resulT.append(i[-1])
    resulT=np.array(resulT)
        # # data=np.load('\\matrix_\\data_0.8_notrain.npy',allow_pickle=True)
        # # data=data.item()
        # # resulT=np.array(data['accuracy'])   
          
        plt.hist(resulT[:,-1], color = 'blue', edgecolor = 'black',bins = 30)
        plt.show()
        # Add labels
        plt.title('Distribution accuracy on semi-random pruning, always train ')
        plt.xlabel('Accuracy on test set')
        plt.ylabel('Number of CNN generated')
        
        # print('numero campioni:',len(resulT))

































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





# bins = np.linspace(0.45, 0.85, 100)
# plt.title('Distribution accuracy on semi-random pruning, always train vs train if val<0.7')

# plt.hist(result, bins, alpha=0.7, label='always train',color = 'red')
# plt.hist(result2, bins, alpha=0.4, label='if acc <0.7',color = 'black')
# plt.legend(loc='upper right')
# plt.show()