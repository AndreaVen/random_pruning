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
from keras import backend as K
from cifar100_reduced_dataset import *
path_to_files='R:\\mygithub\\random_pruning\\matrix_\\'

load_flag=1 # set to 0 to prune, set to 1 to load the result 



if __name__ == '__main__':

   x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced=cifar100_reduced_dataset(path_to_files)

# The export path contains the name and the version of the model
   tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
   model = tf.keras.models.load_model('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5')
   
   # mean = np.mean(x_train_reduced,axis=(0,1,2,3))
   # std = np.std(x_train_reduced, axis=(0, 1, 2, 3))
   # x_test_reduced = (x_test_reduced-mean)/(std+1e-7)
   # idx=1
   # true_label=y_test_reduced[idx]
   # predicted=model.predict(x_train_reduced)
   # idx=5
   # data=x_test_reduced[idx]
   # predicted=model.predict(x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced, x_val_reduced, y_val_reduced, 3)
   # directly_predicted=predicted[idx]
   # print(directly_predicted)
   
   
    
    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
   export_path = 'R:\\mio_modello'
   keras.backend.get_session().run(tf.global_variables_initializer())
   sess=K.get_session()
   tf.saved_model.simple_save(sess,export_path,inputs={'input_image': model.input},outputs={t.name: t for t in model.outputs})


    






 
 inputs={'input_image': model.input}
 
    #nuovo tentativo
MODEL_EXPORT_DIR = 'R:\\mio_modello3'
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
MODEL_VERSION = 2
MODEL_EXPORT_PATH = os.path.join(MODEL_EXPORT_DIR, str(MODEL_VERSION))
print("Model dir: ", MODEL_EXPORT_PATH)
model = tf.keras.models.load_model('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5')
mean = np.mean(x_train_reduced,axis=(0,1,2,3))
std = np.std(x_train_reduced, axis=(0, 1, 2, 3))
x_test_reduced = (x_test_reduced-mean)/(std+1e-7)
predicted=model.predict(x_train_reduced)
print(predicted[0])

print(model.inputs)
input_names = ['image']
name_to_input = {name: t_input for name, t_input in zip(input_names, model.inputs)}
print(name_to_input)



tf.saved_model.simple_save(
    K.get_session(),
    MODEL_EXPORT_PATH,
    inputs={'input_image': model.input},
    outputs={t.name: t for t in model.outputs})








# to predict
INCEPTIONV3_TARGET_SIZE = (299, 299)


def predict(image_path):
    x = img_to_array(load_img(image_path, target_size=INCEPTIONV3_TARGET_SIZE))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return inception_model.predict(x)   


class TFServingClient:
    """
    This is a base class that implements a Tensorflow Serving client
    """
    TF_SERVING_URL_FORMAT = '{protocol}://{hostname}:{port}/v1/models/{endpoint}:predict'

    def __init__(self, hostname, port, endpoint, protocol="http"):
        self.protocol = protocol
        self.hostname = hostname
        self.port = port
        self.endpoint = endpoint

    def _query_service(self, req_json):
        """

        :param req_json: dict (as define in https://cloud.google.com/ml-engine/docs/v1/predict-request)
        :return: dict
        """
        server_url = self.TF_SERVING_URL_FORMAT.format(protocol=self.protocol,
                                                       hostname=self.hostname,
                                                       port=self.port,
                                                       endpoint=self.endpoint)
        response = requests.post(server_url, json=req_json)
        response.raise_for_status()
        return np.array(response.json()['predictions'])


# Define a specific client for our inception_v3 model
class InceptionV3Client(TFServingClient):
    # INPUT_NAME is the config value we used when saving the model (the only value in the `input_names` list)
    INPUT_NAME = "image"
    TARGET_SIZE = INCEPTIONV3_TARGET_SIZE

    def load_image(self, image_path):
        """Load an image from path"""
        img = img_to_array(load_img(image_path, target_size=self.TARGET_SIZE))
        return preprocess_input(img)

    def predict(self, image_paths):
        imgs = [self.load_image(image_path) for image_path in image_paths]

        # Create a request json dict
        req_json = {
            "instances": [{self.INPUT_NAME: img.tolist()} for img in imgs]
        }
        print(req_json)
        return self._query_service(req_json)
    
    hostname = "172.17.0.3"
port = "8501"
endpoint="inception_v3"
client = InceptionV3Client(hostname=hostname, port=port, endpoint=endpoint)

cat_local_preds = predict(cat_filename)
cat_remote_preds = client.predict([cat_filename])

dog_local_preds = predict(dog_filename)
dog_remote_preds = client.predict([dog_filename])
    
    
    
    
    

    # def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    #     """
    #     Freezes the state of a session into a pruned computation graph.
        
    #     Creates a new computation graph where variable nodes are replaced by
    #     constants taking their current value in the session. The new graph will be
    #     pruned so subgraphs that are not necessary to compute the requested
    #     outputs are removed.
    #     @param session The TensorFlow session to be frozen.
    #     @param keep_var_names A list of variable names that should not be frozen,
    #                           or None to freeze all the variables in the graph.
    #     @param output_names Names of the relevant graph outputs.
    #     @param clear_devices Remove the device directives from the graph for better portability.
    #     @return The frozen graph definition.
    #     """
    #     graph = session.graph
    #     with graph.as_default():
    #         freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
    #         output_names = output_names or []
    #         output_names += [v.op.name for v in tf.global_variables()]
    #         input_graph_def = graph.as_graph_def()
    #         if clear_devices:
    #             for node in input_graph_def.node:
    #                 node.device = ""
    #         frozen_graph = tf.graph_util.convert_variables_to_constants(
    #             session, input_graph_def, output_names, freeze_var_names)
    #         return frozen_graph
    

    # #export model in .pb format
    
    # frozen_graph = freeze_session(K.get_session(),
    #                               output_names=[out.op.name for out in model.outputs()])


    # tf.train.write_graph(frozen_graph, 'R:\\', "my_model.pb", as_text=False)