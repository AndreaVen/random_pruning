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

import base64
import requests
import numpy as np
from cifar100_reduced_dataset import *
import json
from cifar100VGG import cifar100vgg # import the architecture and all the modules 
model = tf.keras.models.load_model('R:\\coco_dataset\\dati_salvati\\cifar100_baseline.h5')

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/mio_modello:predict'
path_to_files='R:\\mygithub\\random_pruning\\matrix_\\'
x_train_reduced,y_train_reduced,x_test_reduced,y_test_reduced,x_val_reduced,y_val_reduced=cifar100_reduced_dataset(path_to_files)

mean = np.mean(x_train_reduced,axis=(0,1,2,3))
std = np.std(x_train_reduced, axis=(0, 1, 2, 3))
x_test_reduced = (x_test_reduced-mean)/(std+1e-7)

true_label=y_test_reduced[0]
predicted=model.predict(x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced, x_val_reduced, y_val_reduced, 3)
idx=5
data=x_test_reduced[idx].reshape(1,32,32,3)
predicted=model.predict(x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced, x_val_reduced, y_val_reduced, 3)

directly_predicted=predicted[idx]
print(directly_predicted)
print()
# x=np.random.randn(32,32,3)
# Dump jpeg image bytes as 28x28x1 tensor
# np.set_printoptions(threshold=np.inf)      
# le predizioni son osbagliate
import numpy as np  
# data=np.random.randn(1,32,32,3)
data=x_test_reduced[idx].reshape(1,32,32,3)*10
model = VGG16()


data=np.random.randn(1,224,224,3)

from keras.preprocessing.image import load_img
# load an image from file
data = load_img('mug.jpg', target_size=(224, 224))
from keras.preprocessing.image import img_to_array
# convert the image pixels to a numpy array
data = img_to_array(data).reshape(1,224,224,3)

data=np.random.randn(1,224,224,3)
data2=data
idx=idx+1
SERVER_URL = 'http://localhost:8501/v1/models/mio_modello:predict'
# http://host:port/v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]:predict
# json_request = '{{ "instances" : {} }}'.format(np.array2string(data, separator=',', formatter={'float':lambda x: "%.1f" % x}))

json_request = '{{ "inputs" : {} }}'.format(data.tolist())

resp = requests.post(SERVER_URL, data=json_request)
print('response.status_code: {}'.format(resp.status_code))     
# print('response.content: {}'.format(resp.content))
prediction = json.loads(resp.text)['outputs']
prediction=np.argmax(prediction)
print('serving prediction:',prediction)
# data2=(data2-mean)/(std+1e-7)
y=model.predict(data2)
from keras.applications.vgg16 import decode_predictions
y=decode_predictions(y)
print('model prediction:',y[0][0],'equal index:',np.argmax(model.predict(data2)))
# print('true label:',y_test_reduced[idx])
    


#rookie mistake
# data = {"instances" : [{"ims_ph": ims.tolist()}, {"samples_ph": samples.tolist()} ]}

# idx=6

# data=x_test_reduced[idx]
# data = {"instances" : [data.tolist()]}
# result = requests.post(url=SERVER_URL, data=json.dumps(data))
# print(result.text)



# data = {"inputs" : { "ims_ph": ims, "samples_ph": samples} }


# predicted=model.predict(x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced, x_val_reduced, y_val_reduced, 3)









# da errore { "error": "Missing \'inputs\' or \'instances\' key" }
data=list(data)
endpoint="http://localhost:8501/v1/models/mio_modello:predict" #indirizzo ip
input_data={"model_name":"default","model_version":1,"features": [data]}
result=requests.post(endpoint,json=input_data)
print(result.text)







# The request body for predict API must be JSON object formatted as follows:

# {
 
  

#   // Input Tensors in row ("instances") or columnar ("inputs") format.
#   // A request can have either of them but NOT both.
#   "instances": <value>|<(nested)list>|<list-of-objects>
#   "inputs": <value>|<(nested)list>|<object>
# }

                                      
# def main():
#   # Download the image
#   dl_request = requests.get(IMAGE_URL, stream=True)
#   dl_request.raise_for_status()

#   # Compose a JSON Predict request (send JPEG image in base64).
#   jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')
#   predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes

#   # Send few requests to warm-up the model.
#   for _ in range(3):
#     response = requests.post(SERVER_URL, data=predict_request)
#     response.raise_for_status()

#   # Send few actual requests and report average latency.
#   total_time = 0
#   num_requests = 10
#   for _ in range(num_requests):
#     response = requests.post(SERVER_URL, data=predict_request)
#     response.raise_for_status()
#     total_time += response.elapsed.total_seconds()
#     prediction = response.json()['predictions'][0]

#   print('Prediction class: {}, avg latency: {} ms'.format(
#       prediction['classes'], (total_time*1000)/num_requests))


# if __name__ == '__main__':
#   main()
# da google colab

import numpy as np
import json
idx=3
test_images=x_test_reduced[idx].reshape(1,32,32,3)
test_images=np.random.randn(3,32,32,1)

data = json.dumps({"signature_name": "serving_default", "instances": test_images.tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))



import requests
headers = {"content-type": "application/json"}


json_response = requests.post('http://localhost:8501/v1/models/mio_modello:predict', data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
print(predictions)

#ancora altro
endpoint="http://localhost:8501/v1/models/mio_modello:predict"
headers = {"content-type":"application-json"}
image=np.random.randn(1,32,32,3)
instances = image.tolist()
data = json.dumps({"signature_name":"serving_default","instances": instances})
response = requests.post(endpoint, data=data, headers=headers)
prediction = json.loads(response.text)['predictions']
print('response.status_code: {}'.format(resp.status_code))     
print('response.content: {}'.format(resp.content))
prediction[0]


#show(0, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
 # class_names[np.argmax(predictions[0])], np.argmax(predictions[0]), class_names[test_labels[0]], test_labels[0]))









#ancoraa
def make_vector(image):
    vector = []
    for item in image.tolist():
        vector.extend(item)
    return vector

image=np.random.randn(1,32,32,3)
endpoint="http://localhost:8501/v1/models/mio_modello:predict"
def make_prediction_request(image, endpoint):
    vector = make_vector(image)
    json = {
        "inputs": [vector]
    }
    response = requests.post(endpoint, json=json)

    print('HTTP Response %s' % response.status_code)
    print(response.text)
make_prediction_request(image, endpoint)