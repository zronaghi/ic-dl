
# coding: utf-8

import keras
import sklearn
from keras.models import load_model, Model
from keras.layers import *
from sklearn import metrics
import sys
import time
import os
import numpy as np
import h5py
from load_data import get_data
import keras
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with h5py.File('inxreserv0.h5', 'r') as hf:
    inx = hf['inx0hdf5'][:]
with h5py.File('inyreserv0.h5', 'r') as hf:
    iny = hf['iny0hdf5'][:]

with h5py.File('weight0.h5', 'r') as hf:
    ls2_array = hf['weighthdf5'][:]

print (inx.shape)
print (iny.shape)

inx=np.expand_dims(inx,axis=1)

inx = np.transpose(inx,axes=[0,3,4,2,1])


inputs = Input(shape=inx.shape[1:])

model = load_model('resnetgpu3dall.h5')

y_pred = model.predict(inx)

print (iny.shape)
print (y_pred[y_pred[:,1] == iny].shape)


from sklearn.metrics import roc_curve, auc, roc_auc_score

print (roc_auc_score(iny, y_pred[:,1]), sample_weight = ls2_array)

false_positive_rate, true_positive_rate, thresholds = roc_curve(iny, y_pred[:,1], sample_weight = ls2_array)

np.savetxt('truepredweight0.csv', np.transpose([iny[i],y_pred[:,1][i],ls2_array[i]]))

roc_auc = auc(false_positive_rate, true_positive_rate, reorder=True)
print(auc(false_positive_rate, true_positive_rate))

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid(True)
plt.savefig('resnetgpu3d.png')
