
import math
import peakutils
import pysam as ps
from sklearn import preprocessing, __all__, linear_model

from OCRDetectBycfDNA import AdjustWPS, savgol_filter_func, mergePeaks, getValley, mergeValley, \
    scipy_signal_find_peaks, getTriangleArea
from utils import Gene, TSS, Point

# from scipy import stats
# from sklearn import svm
# import sympy
# import math
# from math import e
import random
# from tensorflow.keras import Model
# from tensorflow.keras.models import load_model
import numpy as np
# import os
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras import backend as K
# from sklearn.model_selection import KFold
# from tensorflow.keras.layers import Input,Conv2D,Activation,Dense,Lambda,Flatten,Embedding,PReLU,BatchNormalization,Bidirectional,LSTM
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import layers
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.model_selection import KFold
# from tensorflow.keras import backend as K
# import copy
from sklearn.utils import shuffle
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# def set_seef(seed):
#     np.random.seed(seed)
#     random.seed(seed)
#     tf.random.set_seed(seed)
# set_seef(random.randint(1,10000))




dic = {
    'gene':'../data/gene/GRCh37.gene.bed',
    # 'gene1':'../content2/align_gene/27ocr_filter.txt',
    'gene1':'../content2/align_gene/DNase_hema.bed',
    # 'bam1234567':'/share/home/3120305302/lm/wps_bamdata/SRR2130051/051_1234567.bam',
    'bam1234567':'/mnt/hgfs/Science/wps_bamdata/SRR2130051/051_1234567.bam',
    }

cnnpath='./testData/DNase_hema_cnn_x.npy'
lstmpath='./testData/DNase_hema_lstm_x.npy'
smoothwpspath='./testData/DNase_hema_smoothwps_x.npy'
rawwpspath='./testData/DNase_hema_rawwps_x.npy'
depthpath='./testData/DNase_hema_depth_x.npy'
indexpath='./testData/DNase_hema_info.txt'


TSS_NonGene = []
with open(dic['gene1'],'r') as f:
    for line in f:
        ll = line.strip().split('\t')
        if ll[0][-1] in ['1']:
        # if ll[0][-1] in ['2', '3', '4', '5', '6', '7']:
        # if ll[0][-1] in ['1','2','3']:
            TSS_NonGene.append(TSS(ll[0], int(ll[1])+int(1000)))

bamfile = ps.AlignmentFile(dic['bam1234567'],'rb')


# TSS_NonGene = shuffle(TSS_NonGene)

TSSes_x = TSS_NonGene
perm = random.sample(range(len(TSSes_x)),len(TSSes_x))
TSSes_x = np.array(TSSes_x)
TSSes_x = TSSes_x[perm[:len(perm)]]
up = 1000
down = 1000

'''cnn_input'''
raw_data = []
for j, tss in enumerate(TSSes_x):
    chrom = tss.chrom
    start = tss.pos - up
    end = tss.pos + down
#    distribution_matrix = np.zeros((int(up+down), 200), dtype=int)
    distribution_matrix = np.zeros((200,200),dtype=int)
    for r in bamfile.fetch(chrom[-1], start-500, end + 500):
        if (not r.is_reverse) and (not r.is_unmapped) and (not r.mate_is_unmapped) and r.mate_is_reverse and 50 < abs(r.isize) < 250:
            if r.reference_start + abs(r.isize) < start:
                continue
            if r.reference_start >= end:
                continue
            if r.reference_start < start:
                continue
            if r.reference_start + abs(r.isize) > end:
                continue
            ss = max(0, r.reference_start - start)
            relative_isize = abs(r.isize)-50
            distribution_matrix[ss//10,relative_isize] += 1
    raw_data.append(distribution_matrix)
raw_data = np.array(raw_data)
cnn_x = []
for mat in raw_data:
    cnn_x.append(mat)
cnn_x = np.array(cnn_x)
# cnn_x = np.load('../datas/train_hk27_cnn_x.npy')
# lstm_x = np.load('../datas/train_hk27_lstm_x.npy')
# labels = np.load('../datas/train_hk27_y.npy')
# cnn_x, lstm_x, labels = shuffle(cnn_x, lstm_x, labels)
'''lstm input'''
feature_matrix = []
for j, tss in enumerate(TSSes_x):
    chrom = tss.chrom
    start = tss.pos - up
    end = tss.pos + down
    up_end = np.zeros(up+down, dtype= int)
    down_end = np.zeros(up+down, dtype= int)
    long = np.zeros(up+down, dtype= int)
    short = np.zeros(up+down, dtype= int)
    cov = np.zeros(up+down, dtype= int)
    wps = np.zeros(up+down, dtype=float)
    win = 120
    for r in bamfile.fetch(chrom[-1], start-500, end + 500):
        if (not r.is_reverse) and (not r.is_unmapped) and (not r.mate_is_unmapped) and r.mate_is_reverse:
            if r.reference_start + abs(r.isize) < start:
                continue
            if r.reference_start >= end:
                continue
            ss = r.reference_start - start
            ee = r.reference_start - start + abs(r.isize)
            if ss >= 0:
                up_end[ss] += 1
            else:
                ss = 0
            if ee < end - start:
                down_end[ee] += 1
            else:
                ee = end - start
            for i in range(ss, ee):
                cov[i] += 1
            if 200 >= abs(r.isize) > 130:
                for i in range(ss, ee):
                    long[i] += 1
            if abs(r.isize) <= 130:
                for i in range(ss, ee):
                    short[i] += 1
            # wps_total
            region1 = int(max(0, ss + win/2))
            region2 = int(min(ee - win/2, end-start))
            i = region1
            while i < region2:
                wps[i] += 1
                i = i+1
            # wps_part
            region1 = int(max(0, ss - win/2))
            region2 = int(min(end-start, ss + win/2))
            i = region1
            while i < region2:
                wps[i] -= 1
                i = i + 1
            # wps_part
            region1 = int(max(ee - win/2, 0))
            region2 = int(min(ee + win/2, end-start))
            i = region1
            while i < region2:
                wps[i] -= 1
                i = i+1
    k = 0
    win = 40
    feature_win = np.zeros((int((up+down)/win), 4), dtype= int)
    while k < (up+down)/win:
        ss = k * win
        ee = k * win + win
        ff = []
        ff.append(int(round(np.mean(cov[ss:ee]))))
        ff.append(int(round(np.mean(long[ss:ee]-short[ss:ee]))))
        ff.append(int(round(np.sum(abs(up_end[ss:ee]-down_end[ss:ee])))))
        ff.append(int(round(np.mean(wps[ss:ee]))))
        feature_win[k] = np.array(ff)
        k = k + 1
    feature_matrix.append(feature_win)
feature_matrix = np.array(feature_matrix)
lstm_x = []
for mat in feature_matrix:
    lstm_x.append(mat)
lstm_x = np.array(lstm_x)
# np.save('./27train_nonocr_cnn_x.npy',cnn_x)
# np.save('./27train_nonocr_lstm_x.npy',lstm_x)
np.save(cnnpath,cnn_x)
np.save(lstmpath,lstm_x)

'''Hotelling T2 input'''
def callOneBed(contig,bed1,bed2,win):
    bed1 = bed1 - win
    bed2 = bed2 + win
    length = bed2 - bed1 + 1
    array = np.zeros(length, dtype=int)
    depth = np.zeros(length, dtype=int)
    depth2 = np.zeros(length, dtype=int)
    for r in bamfile.fetch(contig, bed1, bed2):  # 提取出比对到目标区域内的全部reads
        # if (not r.is_reverse) and (not r.is_unmapped) and (not r.mate_is_unmapped) and r.mate_is_reverse :
        if (not r.is_reverse) and (not r.is_unmapped) and (not r.mate_is_unmapped) and r.mate_is_reverse:
            # is_reverse判断是否比对到了ref的负义链上，is_unmapped判断是否没比对上，mate_is_unmapped配对reads是否没比对上，mate_is_reverse配对reads是否比对到负义链
            # print(r.reference_name,r.isize,r.reference_start,r.reference_start+r.isize)
            if r.isize >= 35 and r.isize <= 80:  # i.isize代表PE read直接的插入片段长度，有时也称Fragment长度
                start = r.reference_start - bed1  # reference_start和reference能比对上部分的起始坐标（reference坐标）
                end = r.reference_start + r.isize - bed1
                # depth + 1
                dstart = start
                dend = end
                if dstart < 0:
                    dstart = 0
                if dend > length:
                    dend = length
                d = dstart
                while d < dend:
                    depth2[d] += 1
                    d += 1

            if r.isize < win or r.isize > 180:
                continue
            start = r.reference_start - bed1
            end = r.reference_start + r.isize - bed1
            # depth + 1
            dstart = start
            dend = end
            if dstart < 0:
                dstart = 0
            if dend >= length:
                dend = length
            d = dstart
            while d < dend:
                depth[d] += 1
                d += 1

            # [$start+W/2,$end-W/2] WPS+1
            region1 = start + int(win / 2)
            region2 = end - int(win / 2)
            if region1 < 0:
                region1 = 0
            if region2 > length:
                region2 = length
            i = region1
            while i < region2:
                array[i] += 1
                i += 1
            # [$start-w/2,$start-1+w/2] WPS-1
            region1 = start - int(win / 2)
            region2 = start + int(win / 2) + 1
            if region1 < 0:
                region1 = 0
            if region2 > length:
                region2 = length
            i = region1
            while i < region2:
                array[i] -= 1
                i += 1
            # [end-w/2+1,$end+w/2] WPS-1
            region1 = end - int(win / 2) + 1
            region2 = end + int(win / 2)
            if region1 < 0:
                region1 = 0
            if region2 > length:
                region2 = length
            i = region1
            while i < region2:
                array[i] -= 1
                i += 1
    # adjustWPS = AdjustWPS(array)
    lenth1 = len(array) - win - 1
    bed1 += win
    bed2 -= win
    array = np.array(array[win: lenth1], dtype=np.float)
    depth = depth[win: lenth1]
    depth2 = depth2[win: lenth1]

    return array,depth,depth2

def calculateFeature(peakObjectList, smoothData, rawDataList):
    peakListSize = len(peakObjectList)
    peakWidth = np.zeros(peakListSize)
    peakHeight = np.zeros(peakListSize)
    peakAngel = np.zeros(peakListSize)
    peakDis = []
    peakArea = np.zeros(peakListSize)
    troughWidth = np.zeros(peakListSize)
    for i in range(len(peakObjectList) - 1):
        # 当前峰的峰高=峰所在位置处对应的wps值-峰初始位置和end位置对应wps的最大值
        peakHeight[i] = smoothData[peakObjectList[i].peakIndex] - max(smoothData[peakObjectList[i].startPos],
                                                                      smoothData[peakObjectList[i].endPos])

        # 峰宽=峰的终止位置（end）- 峰的起始位置（start)
        peakWidth[i] = peakObjectList[i].width

        maxDisPeakCount = 0
        if i < peakListSize - 1 and peakObjectList[i + 1].peakIndex - peakObjectList[
            i].peakIndex < 300 and maxDisPeakCount < 5:
            peakDis.append(peakObjectList[i + 1].peakIndex - peakObjectList[i].peakIndex)
        elif i < peakListSize - 1 and peakObjectList[i + 1].peakIndex - peakObjectList[i].peakIndex > 360:
            maxDisPeakCount += 1
        elif i < peakListSize - 1 and maxDisPeakCount >= 5:
            peakDis.append(10000)
        # 峰的面积
        peakArea[i] = getTriangleArea([float(peakObjectList[i].startPos), rawDataList[peakObjectList[i].startPos]],
                                      [float(peakObjectList[i].peakIndex), rawDataList[peakObjectList[i].peakIndex]],
                                      [float(peakObjectList[i].endPos), rawDataList[peakObjectList[i].endPos]]) / 100

        # 峰的角度
        if 1 + peakObjectList[i].leftK * peakObjectList[i].rightK != 0:
            peakAngel[i] = math.atan(abs((peakObjectList[i].leftK - peakObjectList[i].rightK) / (
                        1 + peakObjectList[i].leftK * peakObjectList[i].rightK))) * 180 / 3.1415
        else:
            peakAngel[i] = 90

        troughWidth[i] = peakObjectList[i + 1].startPos - peakObjectList[i].endPos

    peakDis = np.array(peakDis)

    varpeakHeight = '%.2f' % np.var(peakHeight)
    varpeakWidth = '%.2f' % np.var(peakWidth)
    varpeakDis = '%.2f' % np.var(peakDis)
    varpeakArea = '%.2f' % np.var(peakArea)
    varpeakAngel = '%.2f' % np.var(peakAngel)

    mpeakHeight = '%.2f' % np.mean(peakHeight)
    mpeakWidth = '%.2f' % np.mean(peakWidth)
    mpeakDis = '%.2f' % np.mean(peakDis)
    mpeakArea = '%.2f' % np.mean(peakArea)
    mpeakAngel = '%.2f' % np.mean(peakAngel)

    return varpeakHeight, varpeakWidth, varpeakDis, varpeakArea, varpeakAngel, mpeakHeight, mpeakWidth, mpeakDis, mpeakArea, mpeakAngel

def saveFeature(contig, start,end,k,b,k1,b1,k2,b2,varpeakHeight, varpeakWidth, varpeakDis, varpeakArea, varpeakAngel, mpeakHeight, mpeakWidth,
                mpeakDis, mpeakArea, mpeakAngel):
    with open(featurepath, mode='a+') as f:
        list = str(contig) + "\t" +str(start)+'\t'+str(end)
        list0=str(k*100000) + '\t' + str(b*10) + '\t' + str(k1*100000) + '\t' + str(b1*10) + '\t' + str(k2*100000)+ '\t' + str(b2*10)
        list1=str(varpeakDis) + '\t' + str(varpeakHeight) + '\t' + str(varpeakWidth) + '\t' \
              + str(varpeakArea) + '\t' + str(varpeakAngel)
        list2 = str(mpeakDis) + '\t' + str(mpeakHeight) + '\t' + str(mpeakWidth) + '\t' +\
                str(mpeakArea) + '\t' + str(mpeakAngel)
        f.write(str(list) + '\t'+str(list0) + '\t'+str(list1) + '\t' + str(list2) + '\n')

def get_depthFeature(depth):
    # 新建一个线性回归模型，并把数据放进去对模型进行训练[0,2000bp]
    X_train=range(0,2000)
    Y_train=depth
    X=np.array(X_train).reshape((len(X_train), 1))
    Y=np.array(Y_train).reshape((len(Y_train), 1))
    lineModel=linear_model.LinearRegression()
    lineModel.fit(X,Y)
    Y_predict=lineModel.predict(X)
    k=lineModel.coef_[0][0] #斜率
    b=lineModel.intercept_[0]   #截距

    #[0,1200bp]
    X1_train = range(0, 1200)
    Y1_train = depth[0:1200]
    X1 = np.array(X1_train).reshape((len(X1_train), 1))
    Y1 = np.array(Y1_train).reshape((len(Y1_train), 1))
    lineModel1 = linear_model.LinearRegression()
    lineModel1.fit(X1, Y1)
    Y1_predict = lineModel1.predict(X1)
    k1 = lineModel1.coef_[0][0]  # 斜率
    b1 = lineModel1.intercept_[0]  # 截距

    #[800,2000bp]
    X2_train = range(800, 2000)
    Y2_train = depth[800:2000]
    X2 = np.array(X2_train).reshape((len(X2_train), 1))
    Y2 = np.array(Y2_train).reshape((len(Y2_train), 1))
    lineModel2 = linear_model.LinearRegression()
    lineModel2.fit(X2, Y2)
    Y2_predict = lineModel2.predict(X2)
    k2 = lineModel2.coef_[0][0]  # 斜率
    b2 = lineModel2.intercept_[0]  # 截距

    minPos=0
    minValue=depth[0]
    for i in range(0,2000):
        if(depth[i]<minValue):
            minValue=depth[i]
            minPos=i
    broad_interval_depth=np.average(depth[minPos-300:minPos+300])
    narrow_interval_depth = np.average(depth[minPos - 100:minPos + 100])

    return Y_predict,Y1_predict,Y2_predict,k, b, k1, b1, k2, b2, broad_interval_depth, narrow_interval_depth

def dataWinSum(data,winsize):
    n=len(data)
    cnt = int(n / winsize)
    dataWinSumList=np.zeros(cnt)
    # print(cnt)
    for i in range(0,cnt):
        start=winsize*i
        end=winsize*(i+1)
        dataWinSumList[i]=sum(data[start:end])
        # print(start,end)
    return dataWinSumList

def Save(contig, start, end):
    with open(indexpath, mode='a+') as f:
        list = str(contig) + "\t" +str(start)+'\t'+str(end)
        f.write(str(list) + '\n')

smooth_matrix=[]
raw_matrix=[]
depth_matrix=[]
for j, tss in enumerate(TSSes_x):
    win=120
    contig = tss.chrom
    start = tss.pos - up
    end = tss.pos + down
    wpsList_Nor, lFdepth_Nor, sFdepth_Nor=callOneBed(contig,start,end,win)
    rawWPS = np.array(wpsList_Nor)
    adjustWpsList_Nor = AdjustWPS(wpsList_Nor)
    squareWave = []
    try:
        base = peakutils.baseline(adjustWpsList_Nor, 8)
    except ZeroDivisionError:  # 'ZeroDivisionError'除数等于0的报错方式^M
        base = np.zeros(len(adjustWpsList_Nor))
    adjustWpsList_Nor = np.subtract(adjustWpsList_Nor, base)
    smoothWpsList_Nor = savgol_filter_func(adjustWpsList_Nor, 51, 1)  # SG Filter
    norm_lFdepth_Nor = preprocessing.minmax_scale(lFdepth_Nor)


    smooth_matrix.append(smoothWpsList_Nor)
    raw_matrix.append(rawWPS)
    depth_matrix.append(norm_lFdepth_Nor)
    Save(contig, start, end)

smooth_matrix = np.array(smooth_matrix)
raw_matrix = np.array(raw_matrix)
depth_matrix = np.array(depth_matrix)

smooth_x = []
for mat in smooth_matrix:
    smooth_x.append(mat)
smooth_x = np.array(smooth_x)

raw_x = []
for mat in raw_matrix:
    raw_x.append(mat)
raw_x = np.array(raw_x)

depth_x = []
for mat in depth_matrix:
    depth_x.append(mat)
depth_x = np.array(depth_x)

np.save(smoothwpspath,smooth_x)
np.save(rawwpspath,raw_x)
np.save(depthpath,depth_x)


    # peakHeight = []
    # peaksList = []
    # for data in [smoothWpsList_Nor]:
    #     peaks = scipy_signal_find_peaks(data, height=0.28, distance=25, prominence=0.25, width=[25, 170])
    #     peakObjectList = getValley(data, rawWPS, peaks[1], 5)  # peaks[1]为峰index
    #     mergeValley(peakObjectList)
    #     peakObjectList = mergePeaks(peakObjectList)
    #     peaksList.append([data, peaks[1], peakObjectList])
    #
    # winsize = 200
    # ocrDepthWinSum = dataWinSum(ocrDepthSmoothData, winsize)
    # sd_ocrDepthWinSum = np.std(ocrDepthWinSum)
    # u_ocrDepthWinSum = np.mean(ocrDepthWinSum)
    #
    # WpsWinSum = dataWinSum(smoothWpsList_Nor, winsize)
    # sd_WpsWinSum = np.std(WpsWinSum)
    # u_WpsWinSum = np.mean(WpsWinSum)
    #
    # ocrvarpeakHeight, ocrvarpeakWidth, ocrvarpeakDis, ocrvarpeakArea, ocrvarpeakAngel, \
    # ocrmpeakHeight, ocrmpeakWidth, ocrmpeakDis, ocrmpeakArea, ocrmpeakAngel = calculateFeature(
    #     peakObjectList, smoothWpsList_Nor, rawWPS)
    #
    # ocry_predict, ocry1_predict, ocry2_predict, ocrk, ocrb, ocrk1, ocrb1, ocrk2, ocrb2, ocrbroad, ocrnarrow = get_depthFeature(
    #     ocrDepthSmoothData)
    #
    # saveFeature(contig, start, end,ocrk, ocrb, ocrk1, ocrb1, ocrk2, ocrb2,
    #             ocrvarpeakHeight, ocrvarpeakWidth, ocrvarpeakDis, ocrvarpeakArea, ocrvarpeakAngel,
    #             ocrmpeakHeight, ocrmpeakWidth,ocrmpeakDis, ocrmpeakArea, ocrmpeakAngel)
    # winSave(contig, start, end,sd_WpsWinSum,sd_ocrDepthWinSum,u_WpsWinSum,u_ocrDepthWinSum)
