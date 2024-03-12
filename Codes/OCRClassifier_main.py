import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from tqdm import tqdm

def set_seef(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
set_seef(random.randint(1,10000))

ucl=0.15
rate=0.85
cl=ucl*rate
cnn_test_0 = np.load('../content2/recentData/27filter_ccr_cnn_x.npy')
lstm_test_0= np.load('../content2/recentData/27filter_ccr_lstm_x.npy')

cnn_test_1 = np.load('../content2/recentData/27filter_ocr_cnn_x.npy')
lstm_test_1= np.load('../content2/recentData/27filter_ocr_lstm_x.npy')

cnn_test_2=cnn_test_1*rate+cnn_test_0[:len(cnn_test_1)]*(1-rate)
lstm_test_2=lstm_test_1*rate+lstm_test_0[:len(cnn_test_1)]*(1-rate)

print(len(cnn_test_0),len(cnn_test_1),len(cnn_test_2))

lstm_x = np.concatenate((lstm_test_0, lstm_test_1, lstm_test_2))
cnn_x = np.concatenate((cnn_test_0, cnn_test_1,cnn_test_2))
#labels = np.array([0] * len(cnn_test_0)+[1]*len(cnn_test_1) + [1]*len(cnn_test_2))
st=pd.read_csv('../content2/recentData/st0.85_27train.csv',header=None,sep=',')
labels=[]
st=st[0]

cnt0=0
cnt1=0
cnt2=0
for i in range(len(st)):
    if(st[i]<ucl):
        labels.append(0)
        cnt0 += 1
    elif(st[i]>0.7):
        labels.append(1)
        cnt1 += 1
    else:
        labels.append(2)
        cnt2 += 1

# cnt0=0
# cnt1=0
# cnt2=0
# for i in range(len(st)):
#     if i<len(cnn_test_0):
#         labels.append(0)
#         cnt0+=1
#     if i>=len(cnn_test_0):
#         if (st[i] > 0.7):
#             labels.append(1)
#             cnt1+=1
#         else:
#             labels.append(2)
#             cnt2+=1
print(cnt0,cnt1,cnt2)

labels=np.array(labels)
cnn_x, lstm_x, labels = shuffle(cnn_x, lstm_x, labels)
labels_one_hot=to_categorical(labels)
print(len(cnn_x),len(lstm_x),len(labels))


'''create OCRFinder-model function'''
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Convolution1D, MaxPooling1D, Flatten, Bidirectional,Dropout, AveragePooling1D
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
def create_model():
    cnn_input = Input(shape=(cnn_x.shape[1], cnn_x.shape[2]), name='cnn_input')
    lstm_input = Input(shape=(lstm_x.shape[1], lstm_x.shape[2]), name='lstm_input')

    conv1 = Convolution1D(filters=64, kernel_size=2, strides=2, activation='relu', padding='same',name='conv1')(cnn_input)
    conv1 = AveragePooling1D(pool_size=2, strides=2)(conv1)
    cnn_output = Model(inputs=cnn_input, outputs=conv1)
    main_input = concatenate([cnn_output.output, lstm_input])
    lstm_out = Bidirectional(LSTM(50, return_sequences=True),name='0')(main_input)
    conv = Convolution1D(filters=100, kernel_size=3, activation='relu',strides=1,padding='same',name='1')(lstm_out)
    pool = MaxPooling1D(pool_size=2, strides=2,name='2')(conv)
    drop = Dropout(0.2)(pool)
    flatten = Flatten()(drop)
    dense = Dense(300, activation='relu', kernel_regularizer=None, bias_regularizer=None,name='4')(flatten)
    drop = Dropout(0.2)(dense)
    # main_output = Dense(1, activation='sigmoid', kernel_regularizer=None, bias_regularizer=None,name='6')(drop)
    main_output = Dense(3, activation='softmax', kernel_regularizer=None, bias_regularizer=None, name='6')(drop)#3个概率
    model = Model(inputs=[cnn_output.input, lstm_input], outputs=main_output)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-4), loss='categorical_crossentropy',
                    metrics=['accuracy'])


    return model

for vv in range(1):
    print(vv)
    nb_epochs = 150
    batchsize = 128

    n_split=5
    vvv=0
    id = []

    ypred_a = np.empty(shape=[0,3])
    ypred_b = np.empty(shape=[0,3])

    from tqdm import tqdm
    model_a = create_model()
    model_b = create_model()
    # model_a.fit([cnn_x, lstm_x], labels_one_hot, epochs=100, batch_size=batchsize, shuffle=True, verbose=0)
    # model_b.fit([cnn_x, lstm_x], labels_one_hot, epochs=100, batch_size=batchsize, shuffle=True, verbose=0)
    # model_a.save('../content2/trainModels_3classes/ucl_x_rate/0.7model_a_convlstm_OCRFinder_STslice3_noCL_'+str(vv)+'.h5')
    # model_b.save('../content2/trainModels_3classes/ucl_x_rate/0.7model_b_convlstm_OCRFinder_STslice3_noCL_'+str(vv)+'.h5')

    #
    for train_index, test_index in KFold(n_split).split(cnn_x):
        vvv+=1
        cnn_x_train, cnn_x_test = cnn_x[train_index], cnn_x[test_index]
        lstm_x_train, lstm_x_test = lstm_x[train_index], lstm_x[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        y_train_one_hot = to_categorical(y_train)
        y_test_one_hot = to_categorical(y_test)

        model_a.fit([cnn_x_train, lstm_x_train], y_train_one_hot, epochs=5, batch_size=batchsize, shuffle=True, verbose=0)
        model_b.fit([cnn_x_train, lstm_x_train], y_train_one_hot, epochs=5, batch_size=batchsize, shuffle=True, verbose=0)

        pre_pro_a = model_a([cnn_x_test,lstm_x_test],training=False).numpy()  #预训练
        pre_pro_b = model_b([cnn_x_test,lstm_x_test],training=False).numpy()
        # y_pred_a = np.argmax(pre_pro_a, axis=-1)
        # y_pred_b = np.argmax(pre_pro_b, axis=-1)
        id += list(test_index)
        ypred_a = np.concatenate((ypred_a, pre_pro_a))
        ypred_b = np.concatenate((ypred_b, pre_pro_b))

    cnn_x_train_data = cnn_x[id]
    lstm_x_train_data = lstm_x[id]
    y_train = labels[id]


    from cleanlab.filter import find_label_issues

    label_errors_a = find_label_issues(y_train,ypred_a,n_jobs=1)
    label_errors_b = find_label_issues(y_train,ypred_b,n_jobs=1)

    cleaned_id_a=[]
    cleaned_id_b=[]
    for i in range(len(cnn_x_train_data)):
        if label_errors_a[i]==False:
            cleaned_id_a.append(i)

    for i in range(len(cnn_x_train_data)):
        if label_errors_b[i]==False:
            cleaned_id_b.append(i)

    clean_labels_one_hot_a=to_categorical(y_train[cleaned_id_a])
    clean_labels_one_hot_b=to_categorical(y_train[cleaned_id_b])

    model_a.fit([cnn_x[cleaned_id_b], lstm_x[cleaned_id_b]],clean_labels_one_hot_b, epochs=100, batch_size=128,verbose=0)
    model_b.fit([cnn_x[cleaned_id_a], lstm_x[cleaned_id_a]],clean_labels_one_hot_a, epochs=100, batch_size=128,verbose=0)


    model_a.save('../content2/trainModels_3classes/new/0.85model_a_convlstm_OCRFinder_STslice3_ucl'+'.h5')
    model_b.save('../content2/trainModels_3classes/new/0.85model_b_convlstm_OCRFinder_STslice3_ucl'+'.h5')

