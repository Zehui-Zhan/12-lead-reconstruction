from tensorflow.keras.models import Model
from Norm_layer import *
def Discriminator_s(in_shape=(1024, 12)):
    myinput = tf.keras.layers.Input(shape=in_shape)
    #32个不同的卷积核、一维卷积核尺度为3*11，往上升高、每次步长为1、padding表示尺寸长度不变(param=3*11*32+32)
    x11 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='elu', padding='same')(myinput)
    x12 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation='elu', padding='same')(myinput)
    x13 = tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, activation='elu', padding='same')(myinput)
    x14 = tf.keras.layers.Conv1D(filters=32, kernel_size=9, strides=1, activation='elu', padding='same')(myinput)
    x1 = x11 + x12 + x13 + x14

    x4 = tf.keras.layers.GlobalMaxPooling1D()(x1)
    x4 = tf.keras.layers.Dense(64, activation='elu')(x4)
    y = tf.keras.layers.Dense(1, activation='sigmoid')(x4)
    # activation='sigmoid'
    mymodel = tf.keras.Model(inputs=myinput, outputs=y)
    return mymodel
def Generator_Meg(in_shape=(1024, 1), outdim=12, concate=1,kernel=4,filnums=64):
    # define model
    myinput = tf.keras.layers.Input(shape=in_shape)
    x0 = tf.keras.layers.Conv1D(filters=filnums, kernel_size=17, activation=tf.keras.layers.LeakyReLU(0.2), padding='same')(myinput)

    x1 = gblock2(x0, filnums*2,kernel)
    x0_ = tf.keras.layers.Conv1D(filters=filnums*2, kernel_size=1,strides=2,activation=tf.keras.layers.LeakyReLU(0.2), padding='same')(x0)
    x1=x0_+x1
    # x2=256,128
    x2 = gblock2(x1, filnums*4,kernel)
    x1_ = tf.keras.layers.Conv1D(filters=filnums*4, kernel_size=1,strides=2,activation=tf.keras.layers.LeakyReLU(0.2), padding='same')(x1)
    x2=x1_+x2
    # x3=128,256
    x3 = gblock2(x2, filnums*8,kernel)
    x2_ = tf.keras.layers.Conv1D(filters=filnums*8, kernel_size=1,strides=2,activation=tf.keras.layers.LeakyReLU(0.2), padding='same')(x2)
    x3=x2_+x3
    # x4=64,512
    x4 = gblock2(x3, filnums*8,kernel)
    x3_ = tf.keras.layers.Conv1D(filters=filnums*8, kernel_size=1,strides=2,activation=tf.keras.layers.LeakyReLU(0.2), padding='same')(x3)
    x4=x3_+x4
    # x5=32,1024
    x5 = gblock2(x4, filnums*8,kernel)
    x4_ = tf.keras.layers.Conv1D(filters=filnums*8, kernel_size=1,strides=2,activation=tf.keras.layers.LeakyReLU(0.2), padding='same')(x4)
    x5=x4_+x5
    # x6=16,1024
    x6 = gblock2(x5,filnums*8,kernel)
    x5_ = tf.keras.layers.Conv1D(filters=filnums*8, kernel_size=1,strides=2,activation=tf.keras.layers.LeakyReLU(0.2), padding='same')(x5)
    x6=x5_+x6
    # x7=8,1024
    x7 = gblock2(x6,filnums*16,kernel)
    x6_ = tf.keras.layers.Conv1D(filters=filnums*16, kernel_size=1,strides=2,activation=tf.keras.layers.LeakyReLU(0.2), padding='same')(x6)
    x7=x6_+x7
##开始进行解码
    x8 = upblock2(x7, filnums*8,kernel)
    x7_ = tf.keras.layers.Conv1DTranspose(filters=filnums*8, kernel_size=1, strides=2, activation='elu', padding='same')(x7)
    x8 =x7_ + x8
    if concate:
        x9 = tf.keras.layers.concatenate([x8, x6], axis=2)
    else:
        x9 = x8 + x6
    # 32,1024
    # x9 = attentionblock(x9)
    x9 = upblock2(x9, filnums*8,kernel)
    x8 = tf.keras.layers.Conv1DTranspose(filters=filnums*8, kernel_size=1, strides=2, activation='elu', padding='same')(x8)
    x9 = x8 + x9
    if concate:
        x10 = tf.keras.layers.concatenate([x9, (x5)], axis=2)
    else:
        x10 = x9 + x5
    # 64,512
    # x10 = attentionblock(x10)
    x10 = upblock2(x10, filnums*8,kernel)
    x9 = tf.keras.layers.Conv1DTranspose(filters=filnums*8, kernel_size=1, strides=2, activation='elu', padding='same')(x9)
    x10 = x9 + x10

    if concate:
        x11 = tf.keras.layers.concatenate([x10, x4], axis=2)
    else:
        x11 = x10 + x4
    x11 = upblock3(x11, filnums*8)
    x10 = tf.keras.layers.Conv1DTranspose(filters=filnums*8, kernel_size=1, strides=2, activation='elu', padding='same')(x10)
    x11 = x10 + x11
    if concate:
        x12 = tf.keras.layers.concatenate([x11, x3], axis=2)
    else:
        x12 = x11 + x3

    x12 = upblock3(x12, filnums*4)
    x11 = tf.keras.layers.Conv1DTranspose(filters=filnums*4, kernel_size=1, strides=2, activation='elu', padding='same')(x11)
    x12 = x11 + x12

    if concate:
        x13 = tf.keras.layers.concatenate([x12, x2], axis=2)
    else:
        x13 = x12 + x2

    x13 = upblock3(x13, filnums*2)
    x12 = tf.keras.layers.Conv1DTranspose(filters=filnums*2, kernel_size=1, strides=2, activation='elu', padding='same')(x12)
    x13 = x12 + x13
    if concate:
        x14 = tf.keras.layers.concatenate([x13, x1], axis=2)
    else:
        x14 = x13 + x1
    # x14 = attentionblock(x14)
    x14 = upblock3(x14, filnums)
    x13 = tf.keras.layers.Conv1DTranspose(filters=filnums, kernel_size=1, strides=2, activation='elu', padding='same')(x13)
    x14 = x13 + x14
    # x14 = attentionblock(x14)
    y = tf.keras.layers.Conv1DTranspose(outdim, 5, activation=tf.keras.layers.LeakyReLU(0.2), padding='same')(x14)
    model = Model(myinput, y)
    return model
def gblock2(x, filnum,kernel):
    x = tf.keras.layers.Conv1D(filters=filnum, kernel_size=kernel, strides=1,padding='same')(x)
    x = InstanceNormalization()(x)
    x1 = tf.keras.layers.LeakyReLU(0.2)(x)
    x1 = tf.keras.layers.MaxPooling1D(pool_size=2,padding='same',strides=2)(x1)
    return x1
def upblock2(x, filnum,kernel):
    x1 = tf.keras.layers.Conv1DTranspose(filters=filnum, kernel_size=kernel, strides=2, activation='relu', padding='same')(x)
    x2 = InstanceNormalization()(x1)
    x3 = tf.keras.layers.Dropout(0.2)(x2)
    x4 = tf.keras.layers.ReLU(0.2)(x3)
    return x4
def upblock3(x, filnum):
    x1 = tf.keras.layers.Conv1DTranspose(filters=filnum, kernel_size=4, strides=2,padding='same')(x)
    x2 = tf.keras.layers.BatchNormalization()(x1)
    x3 = tf.keras.layers.LeakyReLU(0.2)(x2)
    return x3

