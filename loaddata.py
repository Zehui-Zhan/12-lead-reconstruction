import glob
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
import h5py #注意：还有很多库如：pywt、scipy等未安装
import  numpy as np
import pywt
from scipy import signal
import scipy.io as sio
from config import myconfig
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from ecgpoint_detector import signalDelineation
os.environ['CUDA_VISIBLE_DEVICES'] = '3' #指定程序在显卡0、1、2中运行
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
print(physical_devices)

for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

path1=glob.glob(r'TrainingSet/TrainingSet1/*.mat') #获取TrainingSet1下所有mat文件
path2=glob.glob(r'TrainingSet/TrainingSet2/*.mat')
path3=glob.glob(r'TrainingSet/TrainingSet3/*.mat')

path=path1[0]
allpath=path1
allpath.remove('TrainingSet/TrainingSet1/label.mat')
for path in path2:
    allpath.append(path)
for path in path3:
    allpath.append(path) #将所有训练数据地址全部放在allpath中

allpath = [str(path) for path in allpath]
allpath= sorted(allpath, key = lambda i:int(i[-8:-4]))


trainpath,testpath=train_test_split(allpath,test_size=0.2, random_state=0)
trainpath= sorted(trainpath, key = lambda i:int(i[-8:-4]))
testpath = sorted(testpath,key= lambda i:int(i[-8:-4]))

def loadecg(path,ecglen = 1024,step = 100,lead = 1):
    file=sio.loadmat(path)
    ecg=file['ECG'][0][0][2]
    c = max(abs(ecg[0,:]))
    ecg=ecg/c #二维矩阵（n，12）
    ecg=np.transpose(ecg) #对该二维矩阵进行转置，（n，12）
    ecg2 = ecgsplit_R2(ecg,ecglen=1024)
    return ecg2

def datatorecord(tfrecordwriter,paths):
    writer = tf.io.TFRecordWriter(tfrecordwriter)  # 1. 定义 writer对象，创建tfrecord文件，输入的为文件名
    for path in paths:
        ecgs = loadecg(path[0])  # 将每个路径下的ECG信号提取出来
        ecg = np.asarray(ecgs).astype(np.float32).tostring()
        # ecg=tf.cast(tf.convert_to_tensor(ecg),dtype=tf.float32)
        """ 2. 定义features """
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    # 'ecg': tf.train.Feature(
                    #     float_list=tf.train.FloatList(value=[ecg])),
                    'ecg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ecg])),
                }))
        """ 3. 序列化,写入"""
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

def decode_tfrecords(example):
    # 定义Feature结构，告诉解码器每个Feature的类型是什么
    feature_description = {
        'ecg': tf.io.FixedLenFeature([], tf.string),
    }
    # 按照feature_description解码
    feature_dict = tf.io.parse_single_example(example, feature_description)
    # 由bytes码转化为tf.float32
    ecg = (tf.io.decode_raw(feature_dict['ecg'], out_type=tf.float32))
    ecg=tf.reshape(ecg,[1024,12])
    return ecg[:,0:1],ecg[:,1:12]
    # return ecg[:,0:1], ecg[:,6:]
    # tf.concat([ecg[:,:lead-1],],axis=2)
def decode_tfrecords10240(example):
    # 定义Feature结构，告诉解码器每个Feature的类型是什么
    feature_description = {
        'ecg': tf.io.FixedLenFeature([], tf.string),
    }
    # 按照feature_description解码
    feature_dict = tf.io.parse_single_example(example, feature_description)
    # 由bytes码转化为tf.float32
    ecg = (tf.io.decode_raw(feature_dict['ecg'], out_type=tf.float32))
    ecg=tf.reshape(ecg,[12,10240])
    #三导联数据集合读取
    # ecg=np.asarray(ecg)
    # ecg=np.reshape(ecg,[1024,12])
    # ecgx=ecg[:,[0,1,6]]
    # ecgy=ecg[:,7:]
    # ecgx=tf.reshape(ecgx,[1024,3])
    # ecgy=tf.reshape(ecgy,[1024,5])
    # return ecgx, ecgy
    return ecg[lead-1:lead,:],ecg[lead:,:]
    # return ecg[:,0:1], ecg[:,6:]
    # tf.concat([ecg[:,:lead-1],],axis=2)

def decode_tfrecords_data4(example):
    # 定义Feature结构，告诉解码器每个Feature的类型是什么
    # 定义Feature结构，告诉解码器每个Feature的类型是什么
    feature_description = {
        'ecg': tf.io.FixedLenFeature([], tf.string),
    }
    # 按照feature_description解码
    feature_dict = tf.io.parse_single_example(example, feature_description)
    # 由bytes码转化为tf.float32
    ecg = (tf.io.decode_raw(feature_dict['ecg'], out_type=tf.float32))
    ecg=tf.reshape(ecg,[1024,12])
    return ecg[:,lead-1:lead],ecg[:,lead:]

def read_tfrecords(tfrecord_file):
    #读取文件,数据预处理的一部分
    dataset = tf.data.TFRecordDataset(tfrecord_file)  # 读取 TFRecord 文件
    dataset = dataset.map(decode_tfrecords,num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 解析数据
    return dataset

def read_tfrecords10240(tfrecord_file):
    #读取文件,数据预处理的一部分
    dataset = tf.data.TFRecordDataset(tfrecord_file)  # 读取 TFRecord 文件
    dataset = dataset.map(decode_tfrecords10240,num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 解析数据
    return dataset

def ecg_re2(R, ecg_split,a):
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

def ecgsplit_R2(ecg, ecglen=1024):
    ecg2 = []
    ecg = tf.cast(ecg,dtype=tf.float32)
    processed_ecg = nk.ecg_process(ecg[:, 0], 500)
    R = processed_ecg[1]['ECG_R_Peaks']
    length = R.shape[0]
    if R[0] > ecglen/2:
        R0_index=np.argmax(ecg[0:512,0])
        a = tf.zeros(512 - R0_index)
        a = tf.expand_dims(a, axis=1)
        a = tf.tile(a, [1, ecg.shape[1]])
        b = ecg[0:R0_index + 512]
        b = tf.cast(b,dtype=tf.float32)
        a = tf.concat((a,b), axis=0)
        ecg2.append(a)
        ecg2.append(ecg[R[0] - 512:R[0] + 512])
        R0_index = np.asarray([R0_index])
        R_new = np.concatenate((R0_index,R))
    else:
        a = tf.zeros(512 - R[0])
        a = tf.expand_dims(a, axis=1)
        a = tf.tile(a, [1, ecg.shape[1]])
        b = ecg[0:R[0] + 512]
        b = tf.cast(b,dtype=tf.float32)
        a = tf.concat((a,b), axis=0)
        ecg2.append(a)
        R_new = R
    for i in range(1, length):
        if R[i] >= ecglen / 2 and (ecg.shape[0] - R[i]) >= ecglen / 2:
            if i == (length - 1):
                ecg2.append(ecg[R[i] - 512:R[i] + 512])
                R_end=R[i] + np.argmax(ecg[R[i]:len(ecg[:,0]),0])
                a = tf.zeros(len(ecg[:,0]) - R_end)
                a = tf.expand_dims(a, axis=1)
                a = tf.tile(a, [1, ecg.shape[1]])
                a = tf.concat((ecg[R_end-512:len(ecg[:,0])],a), axis=0)
                ecg2.append(a)
                R_end = np.asarray([R_end])
                R_new = np.concatenate((R_new,R_end))
            else:
                ecg2.append(ecg[R[i] - 512:R[i] + 512])
        elif R[i] < ecglen/2:
            a = tf.zeros(512 - R[i])
            a = tf.expand_dims(a, axis=1)
            a = tf.tile(a, [1, ecg.shape[1]])
            a = tf.concat((a, ecg[0:R[i] + 512]), axis=0)
            ecg2.append(a)
        elif R[i] > ecglen / 2 and (ecg.shape[0] - R[i]) < ecglen / 2:
            b = tf.zeros(R[i] + 512 - ecg.shape[0])
            b = tf.expand_dims(b, axis=1)
            b = tf.tile(b, [1, ecg.shape[1]])
            b = tf.concat((ecg[R[i] - 512:ecg.shape[0]], b), axis=0)
            ecg2.append(b)
    return np.asarray(ecg2),R_new,R

def signal_filter(data, frequency=myconfig.fs, highpass=myconfig.hf, lowpass=myconfig.lf):
    # Frequence 为信号的采样频率
    # highpass 低通滤波器可以通过的最高频率
    # lowpass  高通滤波器可以通过的最低频率
    [b, a] = signal.butter(3, [lowpass / frequency * 2, highpass / frequency * 2], 'bandpass')
    Signal_pro = signal.filtfilt(b, a, data)
    return Signal_pro

def signal_filter_wavelet(signal):
    # 设置小波变换参数
    wavelet = 'db4'
    level = 4
    # 对信号进行小波变换
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 通过阈值滤波去除小波系数中的噪声
    threshold = 0.1
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))

    # 对去噪后的信号进行小波重构
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal

def butter_lowpass_filter(signal, cutoff_freq, sample_rate, order):
    nyquist_freq = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)




