import collections
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.ndimage.measurements import label, standard_deviation
#from scipy.stats.mstats_basic import kstest, normaltest
from sklearn.cluster import KMeans
import sys
from scipy.signal import savgol_filter
import math
from subprocess import call
import os.path
from utils import Gene, TSS, Point
from scipy import stats
from sklearn import svm
import sympy
import math
from math import e
import random
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Input,Conv2D,Activation,Dense,Lambda,Flatten,Embedding,PReLU,BatchNormalization,Bidirectional,LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
import copy
from sklearn.utils import shuffle

def set_seef(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
set_seef(random.randint(1,10000))
dic = {
    'gene':'../data/gene/GRCh37.gene.bed',
    'non_gene':'../data/gene/non_gene_1234567.bed',
    'non_gene_4':'../data/gene/non_gene_4.bed',
    'non_gene_2':'../data/gene/non_gene_2.bed',
    # 'fasta':'/home/jiay/Desktop/hg19/hg19.fa',
    'bam1234567':'../data/051_1234567.bam',
    'TSS_low':'../data/gene/low_expressed.bed',
    'TSS_HK':'../data/gene/HK.bed',
    'TSS_silent':'../data/gene/silent_gene_TSS.bed',
    'ATAC_Bcell':'../data/gene/ATAC_Bcell.bed',
    'ATAC_Brain':'../data/gene/ATAC_Brain.bed',
    'ATAC_hema':'../data/gene/ATAC_hema.bed',
    'model_save':'../model'
    }


rate=0.7
cnn_test_0 = np.load('../content2/recentData/27filter_ccr_cnn_x.npy')
lstm_test_0= np.load('../content2/recentData/27filter_ccr_lstm_x.npy')

cnn_test_1 = np.load('../content2/recentData/27filter_ocr_cnn_x.npy')
lstm_test_1= np.load('../content2/recentData/27filter_ocr_lstm_x.npy')

cnn_test_2=cnn_test_1*rate+cnn_test_0[:len(cnn_test_1)]*(1-rate)
lstm_test_2=lstm_test_1*rate+lstm_test_0[:len(cnn_test_1)]*(1-rate)
# cnn_test_2 = np.load('../content2/data/27Partocr_cnn_x.npy')
# lstm_test_2= np.load('../content2/data/27Partocr_lstm_x.npy')

print(len(cnn_test_0),len(cnn_test_1),len(cnn_test_2))
# print(cnn_test_2)

lstm_test = np.concatenate((lstm_test_0, lstm_test_1, lstm_test_2))
cnn_test = np.concatenate((cnn_test_0, cnn_test_1,cnn_test_2))
print(len(cnn_test))

mymodel_a5=load_model('../trained_models/OCRFinder/model_a_OCRFinder_3.h5',compile=False)
mymodel_b5=load_model('../trained_models/OCRFinder/model_b_OCRFinder_3.h5',compile=False)

myre_a5 = mymodel_a5([cnn_test,lstm_test]).numpy()
myre_b5 = mymodel_b5([cnn_test,lstm_test]).numpy()
# print(re_a)
myre5 = (myre_a5+ myre_b5) / 2
# myre5.sort(axis=0)
mycount5 = 0
for rr5 in myre5:
    if rr5 >= 0.5:
        mycount5 += 1

plt.plot(range(0,len(myre5)),myre5,'o')
plt.show()


np.savetxt('./recentData/filter_rate0.7_pc.txt',myre5,fmt='%f')
# with open('./pc_0.8rate.txt', mode='a+') as f:
#     for rr5 in myre5:
#         list = str(rr5)
#         f.write(str(list) + '\n')


