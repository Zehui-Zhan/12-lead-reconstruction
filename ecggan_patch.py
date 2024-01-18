import os
import numpy as np
from config import  myconfig
import tensorflow as tf
# 最好是在gpu上运行，如果有
import time
import pydot
import graphviz

os.environ['CUDA_VISIBLE_DEVICES'] = '0' #指定程序在显卡0、1、2中运行
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
assert len(gpus) > 0, "Not enough GPU hardware devices available"

for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)

GP_weight=10.0

from  tensorflow import keras
from ECG_GAN import Discriminator_s,Generator_Meg
from GAN_loss import discriminator_loss,generator_loss
from loaddata_all import read_tfrecords,decode_tfrecords_label,read_tfrecords_label,read_tfrecords10240,ecgsplit_R2
from ecgpoint_detector import signalDelineation
from utils_datat import *
from metric_chen import *
import matplotlib.pyplot as plt
import scipy.io as sio
import metric_chen
import time
assert tf.__version__.startswith('2.')

def train_1to12(trainds,generator,discriminator):
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0,beta_2=0.9)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0,beta_2=0.9)
    for epoch in range(100):
        # start = time.time()
        for step, inputs in enumerate(trainds):
            ecgx, ecgy = inputs
            with tf.GradientTape() as gen_tape, tf.GradientTape(watch_accessed_variables=False) as disc_tape:
                disc_tape.watch(discriminator.trainable_variables)
                gen_output = tf.cast(generator(ecgx, training=True), dtype=tf.float32)
                ecgy = tf.concat([ecgx, ecgy], axis=2)
                gen_output = tf.concat([ecgx, gen_output], axis=2)
                disc_real_output = discriminator(ecgy, training=True)
                disc_generated_output = discriminator(gen_output, training=True)
                gen_loss = generator_loss(disc_generated_output, gen_output, ecgy,a=1,b=1,c=1)
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
                generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
                g_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
                discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                d_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
            if step % 200 == 0:
                print(epoch, step, gen_loss.numpy(), disc_loss.numpy())
                loss_d.append(disc_loss)
                loss_g.append(gen_loss)
    print('train_1to12')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.subplot(211)
    plt.plot(loss_d, color='r')
    plt.title('D_loss')
    plt.grid(True)
    plt.subplot(212)
    plt.plot(loss_g, color='r')
    plt.title('G_loss')
    plt.grid(True)
    plt.show()
    show_12lead2(0,ecgx,ecgy)
    show_12lead2(0,ecgx,gen_output)

if __name__ == '__main__':
    loss_d = []
    loss_g = []
    ecgx_test=[]
    ecgy_test=[]
    batch_size=64
    # trainds = read_tfrecords('train').batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # test = read_tfrecords('test').batch(64).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # generator = Generator_Meg(in_shape=(myconfig.ecglen,1),outdim=11) #建造一个初始生成模型
    # discriminator = Discriminator_s(in_shape=(myconfig.ecglen, 12))
    # trainds = trainds.shuffle(1000, seed=1)
    # train_1to12(trainds,generator,discriminator)


