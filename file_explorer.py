# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:02:00 2020

@author: andrea
"""


import numpy as np
acc=[]
data=np.load('R:\\matrix_40 - Copia\\filters_0.8.npy',allow_pickle=True)

result=np.load('R:\\matrix_40 - Copia\\result_0.8.npy',allow_pickle=True)
risultato=[]
for i in result:
    risultato.append(i[-1])
max_idx=np.argmax(risultato)





try:
    data=data.item()
    data=data['accuracy']
except:
    print('')


arr=np.array(data)
if arr.shape[1]==14:
    for i in data:
        acc.append(i[-1])

if arr.shape[1]==2:
    for i in data:
        acc.append(i[-1])


# for i in data:
#     acc.append(i[-1])
    
import matplotlib.pyplot as plt 
plt.hist(acc, color = 'blue', edgecolor = 'black',
          bins = 20)
plt.show()
# Add labels
plt.title('Distribution accuracy on random pruning')
plt.xlabel('Accuracy on test set')
plt.ylabel('Number of CNN generated')