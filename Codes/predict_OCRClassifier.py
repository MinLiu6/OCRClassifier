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
cnn_test_1 = np.load('../content2/testData/test_DNase_hema_cnn_x.npy')
lstm_test_1= np.load('../content2/testData/test_DNase_hema_lstm_x.npy')
print(len(cnn_test_1))
#
cnn_test_0 = np.load('../content2/testData/nonocr1_cnn_x.npy')
lstm_test_0= np.load('../content2/testData/nonocr1_lstm_x.npy')
cnn_test_0 = shuffle(cnn_test_0)
lstm_test_0 = shuffle(lstm_test_0)

rate=0.7
# rate=0.6
cnn_test_2=cnn_test_1*rate+cnn_test_0[:len(cnn_test_1)]*(1-rate)
lstm_test_2=lstm_test_1*rate+lstm_test_0[:len(lstm_test_1)]*(1-rate)

lstm_test = np.concatenate((lstm_test_0[:1*len(lstm_test_1)],lstm_test_1, lstm_test_2))
cnn_test = np.concatenate((cnn_test_0[:1*len(cnn_test_1)],cnn_test_1, cnn_test_2))

print(lstm_test.shape,cnn_test.shape)



def roc_save(filename, fpr, tpr, auc):#filename为写入CSV文件的路径，data为要写入数据列表.
    with open(filename, mode='a+') as f:
        list = str("fpr") +'\t'+ str("tpr") + '\t'+str("auc")
        f.write(str(list) + '\n')
        for i in range(len(fpr)):
            list = str(fpr[i])+'\t'+str(tpr[i])+'\t'+str(auc)
            f.write(str(list) + '\n')
    f.close()

def pr_save(filename, precision, recall, aupr):#filename为写入CSV文件的路径，data为要写入数据列表.
    with open(filename, mode='a+') as f:
        list = str("precision") +'\t'+ str("recall") +'\t'+ str("aupr")
        f.write(str(list) + '\n')
        for i in range(len(precision)):
            list = str(precision[i])+'\t'+str(recall[i])+'\t'+str(aupr)
            f.write(str(list) + '\n')
    f.close()

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score, f1_score, \
    precision_score, recall_score
from sklearn.metrics import precision_recall_curve
y_test = np.array([0] * len(cnn_test_0[:1*len(lstm_test_1)])+[1]*len(cnn_test_1) + [2] * len(cnn_test_2))
y_test_one_hot=to_categorical(y_test)

mymodel_a=load_model('../content2/trainModels_3classes/ucl_x_rate/0.7model_a_convlstm_OCRFinder_STslice3_0.h5',compile=False)
mymodel_b=load_model('../content2/trainModels_3classes/ucl_x_rate/0.7model_b_convlstm_OCRFinder_STslice3_0.h5',compile=False)
myre_a = mymodel_a([cnn_test,lstm_test]).numpy()
myre_b = mymodel_b([cnn_test,lstm_test]).numpy()
myre = (myre_a + myre_b) / 2
y_pred=[]
fpr = dict()
tpr = dict()
roc_auc = dict()
for rr in myre:
    y_pred.append(np.argmax(rr))
print(len(y_test),len(y_pred))

cm1 = confusion_matrix(y_test,y_pred)
print(cm1)
print(classification_report(y_test,y_pred,target_names=["闭合区域","开放区域","部分开放区域"],digits=2))


print("----------------")
#准确率 正确分类的样本数 比上 总样本数

print('精度：',precision_score(y_test,y_pred,average='macro'))
print('recall：',recall_score(y_test,y_pred,average='macro'))
print('f1-score：',f1_score(y_test,y_pred,average='macro'))
print('准确率：',accuracy_score(y_test,y_pred))



model_a=load_model('../content2/trainModels_3classes/new/0.7model_a_convlstm_OCRFinder_STslice3_ucl.h5',compile=False)
model_b=load_model('../content2/trainModels_3classes/new/0.7model_b_convlstm_OCRFinder_STslice3_ucl.h5',compile=False)
# model_a=load_model('../trained_models/OCRFinder/model_a_OCRFinder_3.h5',compile=False)
# model_b=load_model('../trained_models/OCRFinder/model_b_OCRFinder_3.h5',compile=False)
re_a = model_a([cnn_test,lstm_test]).numpy()
re_b = model_b([cnn_test,lstm_test]).numpy()
re = (re_a + re_b) / 2
y_pred2=[]
for r in re:
    y_pred2.append(np.argmax(r))

print("--------没有调整ucl--------")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)
print(classification_report(y_test,y_pred2,target_names=["闭合区域","开放区域","部分开放区域"],digits=2))

print("----------------")
#准确率 正确分类的样本数 比上 总样本数

print('精度：',precision_score(y_test,y_pred2,average='macro'))
print('recall：',recall_score(y_test,y_pred2,average='macro'))
print('f1-score：',f1_score(y_test,y_pred2,average='macro'))
print('准确率：',accuracy_score(y_test,y_pred2))

