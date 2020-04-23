# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A client that performs inferences on a ResNet model using the REST API.

The client downloads a test image of a cat, queries the server over the REST API
with the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a ResNet SavedModel
from:

https://github.com/tensorflow/models/tree/master/official/resnet#pre-trained-model

The SavedModel must be one that can take JPEG images as inputs.

Typical usage example:

    resnet_client.py
"""

from __future__ import print_function
import numpy as np  

import base64
import requests
import numpy as np
from cifar100_reduced_dataset import *
import json
from cifar100VGG import cifar100vgg # import the architecture and all the modules 
# model = tf.keras.models.load_model('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5')

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/mio_modello:predict'
path_to_files='R:\\mygithub\\random_pruning\\matrix_\\'
x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced=cifar100_reduced_dataset(path_to_files)

mean = np.mean(x_train_reduced,axis=(0,1,2,3))
std = np.std(x_train_reduced, axis=(0, 1, 2, 3))
x_test_reduced = (x_test_reduced-mean)/(std+1e-7)

SERVER_URL = 'http://localhost:8501/v1/models/mio_modello:predict'
# http://host:port/v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]:predict
# json_request = '{{ "instances" : {} }}'.format(np.array2string(data, separator=',', formatter={'float':lambda x: "%.1f" % x}))
idx=np.random.randint(0,300)
data=x_test_reduced[idx].reshape(1,32,32,3)
json_request = '{{ "inputs" : {} }}'.format(data.tolist())

resp = requests.post(SERVER_URL, data=json_request)
print('response.status_code: {}'.format(resp.status_code))     
print('response.content: {}'.format(resp.content))

print('true label=',y_test_reduced[idx])







