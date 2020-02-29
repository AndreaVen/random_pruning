# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:48:43 2020

@author: andrea
"""


import os
import numpy as np

path='R:\\matrix 90'



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


import matplotlib.pyplot as plt


# matplotlib histogram
plt.hist(resulT[:,-1], color = 'blue', edgecolor = 'black',
         bins = 20)

plt.show()

# Add labels
plt.title('Distribution accuracy on 40% random pruning')
plt.xlabel('Accuracy on test set')
plt.ylabel('Number of CNN generated')



