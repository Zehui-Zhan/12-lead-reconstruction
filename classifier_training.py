import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy.io import savemat
import random

import os
import argparse
import csv
import glob
import time
from os.path import basename
import tensorflow_addons as tfa
from  config import myconfig
from classifier_structure import classifier,lr_warmup,classifier2
from loaddata_4_classifier import  read_tfrecords
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)
# 参数设置
bs=32
epoch = 100
patience = 50
Train = 1
Test = 1
load_oldmodel=0
ecglen=myconfig.ecglen
dim=myconfig.dim
classnum=myconfig.classnum

modelpath='./models/classifier2_'+str(dim)
def F1newget(y,m):
    classnum = y.shape[1]
    CM = np.zeros((classnum, classnum))
    F1 = np.zeros((classnum, 1))
    sen=np.zeros((classnum, 1))
    ppr=np.zeros((classnum, 1))
    for i in range(y.shape[0]):
        result = np.argmax(m[i])
        T = np.argmax(y[i])
        CM[T][result] += 1
    print(CM)
    a=0
    for i in range(classnum):
        F1[i] = 2 * CM[i][i] / (sum(CM[i, :]) + sum(CM[:, i]))
        a+=CM[i][i]
        sen[i]= CM[i][i] / (sum(CM[i, :]))
        ppr[i]= CM[i][i] / (sum(CM[:,i]))
        print('F1-score of', i, 'is:', F1[i])
    print("Overall Accuracy is",a/sum(sum(CM)))
    print("macro-F1 is",np.mean(F1))

def modeltrain(model, path, trainds,testds):
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.categorical_crossentropy,
                  # loss={'y0': tf.keras.losses.categorical_crossentropy,
                  # #       'yl': tf.keras.losses.categorical_crossentropy,
                  # #       'yv': tf.keras.losses.categorical_crossentropy,
                  # #       'y12': tf.keras.losses.categorical_crossentropy
                  # #       # 'max': 0,
                  #       },
                  metrics=['accuracy'])

    if Train:
        ckpt_saver = ModelCheckpoint(path,monitor="max_accuracy",mode='max',save_best_only=True,save_freq='epoch',verbose=1)
        lrsetting = tf.keras.callbacks.LearningRateScheduler(lr_warmup)
        early_stop = EarlyStopping(monitor="max_accuracy", patience=patience)
        model.fit(trainds,epochs=epoch,batch_size=bs,callbacks=[ckpt_saver, early_stop,lrsetting])

    if Test:
        model = tf.keras.models.load_model(path)
        print("The classification result of testing is:")
        model.evaluate(testds)
        nodata=1
        for step,inputs in enumerate(testds):
            x1,y1=inputs
            if nodata:
                x_test=x1
                y_test=y1
                nodata=0
            else:
                x_test=tf.concat([x_test,x1],axis=0)
                y_test=tf.concat([y_test,y1],axis=0)
        m12 = model.predict(x_test)[-1]
        F1newget(y_test,m12)

        m = model.predict(x_test)[0]
        F1newget(y_test,m)

if __name__=='__main__':
    # trian4cwriter和test4cwriter为12导联数据集
    # trian4cwriter_lead1和test4cwriter_lead1为1导联数据集

    trainds = read_tfrecords('train4cwriter_lead'+str(dim)).batch(bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).shuffle(100*bs)
    testds = read_tfrecords('test4cwriter_lead'+str(dim)).batch(4*bs)
    Gtestds=read_tfrecords('test4cwriter_Gen'+str(dim)+'lead').batch(4*bs)
    if load_oldmodel:
        classifier = tf.keras.models.load_model(modelpath)
    else:
        classifier=classifier2(length=ecglen,dim=dim)
    classifier.summary()

    modeltrain(classifier,modelpath,trainds,testds)
    Train = 0
    modeltrain(classifier,modelpath,trainds,Gtestds)
