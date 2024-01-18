from ecgpoint_detector import signalDelineation
import  os
import  numpy as np
from config import  myconfig
import  tensorflow as tf
from metric_chen import *
from utils_datat import *
import matplotlib.pyplot as plt
import neurokit2 as nk
os.environ['CUDA_VISIBLE_DEVICES'] = '3' #指定程序在显卡0、1、2中运行
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"

from loaddata_all import read_tfrecords,signal_filter,decode_tfrecords_label,read_tfrecords_label,toonehot,ecgsplit_R,read_tfrecords10240,ecgsplit_R2

def ecg_re(ecg, ecg_split,ecglen=1024):
    ecg_r = []
    waves = signalDelineation(ecg[:, 0], 500)
    R = waves[1][:, 2]
    length = R.shape[0]
    a = 0
    for i in range(length):
        if R[i] > ecglen/2 and (ecg.shape[0] - R[i]) > ecglen/2 and i != len:
            dis = R[i + 1] - R[i]
            if a == 0:
                ecg_r.append(ecg_split[a, 0:(513 + dis), :])
            else:
                ecg_r.append(ecg_split[a, 513:513 + dis, :])
            a = a + 1
    ecg_c = tf.concat((ecg_r[0], ecg_r[1]), axis=0)
    for i in range(2,len(ecg_r)):
        ecg_c = tf.concat((ecg_c, ecg_r[i]), axis=0)
    return ecg_c
def ecg_re2(R, ecg_split,a=10240):
    ecg_r = []
    length = R.shape[0]
    for i in range(length):
        if i != (length-1):
            if ((R[i+1]-R[i])%2==1):
                ecg_r.append(ecg_split[i, 512:512 + int((R[i+1] - R[i])/2), :])
                ecg_r.append(ecg_split[i + 1, 512 - int((R[i+1] - R[i])/2)-1:512,:])
            else:
                ecg_r.append(ecg_split[i, 512:512 +int((R[i + 1] - R[i])/2), :])
                ecg_r.append(ecg_split[i + 1, 512 - int((R[i + 1] - R[i])/2):512, :])
    ecg_c = tf.concat((ecg_split[0,512-R[0]:512], ecg_r[0]), axis=0)
    for i in range(1,len(ecg_r)):
        ecg_c = tf.concat((ecg_c, ecg_r[i]), axis=0)
    ecg_c = tf.concat((ecg_c,ecg_split[length-1,512:512+a-R[length-1],:]),axis=0)
    return ecg_c

if __name__ == '__main__':
    ecglen=1024
    trainds = read_tfrecords10240('CPSC_train_10').batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test = read_tfrecords10240('CPSC_test_10240').batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)##总共有220条数据
    ecg_10 = []
    for step, inputs in enumerate(test):
        if step %2 == 0:
            ecgx, ecgy = inputs
            ecgx = ecgx[0, :, :]
            ecgy = ecgy[0, :, :]
            ecgx = tf.transpose(ecgx)
            ecgy = tf.transpose(ecgy)
            ecg = tf.concat((ecgx, ecgy), axis=1)
            ecg_10.append(ecg)
    generator = tf.keras.models.load_model('GeneratorMeg_resnet_CPSC_train40.9', compile=False)
    #上述为一端原信号
    ecg_split_ = []
    ecg_split_x = []
    ecg_split_y = []
    ecg_split_m = []
    R = []
    for i in range(len(ecg_10)):
        #输入为(n,lead）的数据类型
        ecg_split,R_1,R_0 = ecgsplit_R2(ecg_10[i], 1024)
        ecg_split_.append(ecg_split)
        ecg_split_x.append(tf.expand_dims(ecg_split[:,:,0],axis=-1))
        ecg_split_y.append(ecg_split[:,:,1:12])
        R.append(R_1)
        ecg_split_m.append(generator(tf.expand_dims(ecg_split[:,:,0],axis=-1)))
    # #进行切割
    # #进行合成
    ecg_g10240 = []
    ecg_y10240 = []
    for i in range(len(ecg_split_m)):
        ecg_c=ecg_re2(R[i],ecg_split_m[i])
        ecg_y = ecg_re2(R[i],ecg_split_y[i])
        ecg_y10240.append(ecg_y)
        ecg_g10240.append(ecg_c)




