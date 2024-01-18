import keras.losses
from numpy import zeros
from numpy import ones
import numpy as np
import tensorflow as tf
from numpy import expand_dims
from numpy.random import randn, randint
from Norm_layer import *
from matplotlib import pyplot
import h5py
import os
from scipy import signal
from ecgpoint_detector import signalDelineation

def discriminator_loss(disc_real_output, disc_generated_output):
    # disc_loss=tf.reduce_mean(disc_generated_output-disc_real_output)
    real_loss = tf.reduce_mean(tf.pow(tf.ones_like(disc_real_output)-disc_real_output,2))#理想值与真实值相减平方，取batchsize的平均。
    generated_loss = tf.reduce_mean(tf.pow(tf.zeros_like(disc_generated_output)-disc_generated_output,2))
    disc_loss = real_loss + generated_loss
    return disc_loss
def Classification_loss(logits,signal_label):
    cross_entroy = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(signal_label, logits, from_logits = False))
    return cross_entroy
def generator_loss(disc_generated_output, gen_output, target,a=1,b=1,c=0):
    LAMBDA = 200
    gan_loss = tf.reduce_mean(tf.pow(tf.ones_like(disc_generated_output)-disc_generated_output,2))
    # l1_loss = tf.reduce_mean(tf.abs(tf.cast(target, dtype=tf.float32) - tf.cast(gen_output, dtype=tf.float32)))
    loss_lead=[]
    gen_output = tf.reshape(gen_output,[len(gen_output[:,0,0]),-1])
    target = tf.reshape(target,[len(target[:,0,0]),-1])
    for i in range(len(gen_output[:,0])):
        loss_lead.append(soft_dtw(gen_output[i,:],target[i,:],0.1))
    l1_loss = tf.reduce_mean(loss_lead)
    l1_loss = tf.cast(l1_loss,dtype=tf.float32)
    # l1_loss = (LAMBDA * l1_loss * l1_loss)
    total_gen_loss = gan_loss + l1_loss
    return total_gen_loss

