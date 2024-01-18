import  tensorflow as tf
import tensorflow_addons as tfa
def attentionblock(x):
    # SE ResNet
    y = x
    z = tf.keras.layers.GlobalMaxPooling1D()(x)
    z = tf.keras.layers.Dense(int(x.shape[-1] / 4), activation="elu")(z)
    z = tf.keras.layers.Dense(x.shape[-1], activation="sigmoid")(z)
    x = tf.keras.layers.Multiply()([x, z])
    return tf.keras.layers.Add()([x, y])

def multidropout(y0,dropr):
    y=tf.keras.layers.Dropout(dropr[0])(y0)/len(dropr)
    for i in range(1,len(dropr)):
        y+= tf.keras.layers.Dropout(dropr[i])(y0) / len(dropr)
    return y

def resblock(x, kerlen=3, kernum=32, poolsize=2):

    x0 = tf.keras.layers.Conv1D(kernum,kerlen, activation='linear', padding='same', strides=1)(x)
    x0 = tf.keras.layers.ELU()(x0)
    x0 = tf.keras.layers.MaxPooling1D(strides=poolsize,padding='same',pool_size=poolsize)(x0)
    x1 = tf.keras.layers.Conv1D(kernum, kerlen, activation='linear', padding='same', strides=1)(x0)
    x1 = tf.keras.layers.ELU()(x1)
    x = x1+x0
    x = tf.keras.layers.MaxPooling1D(strides=poolsize,padding='same',pool_size=poolsize)(x)
    x = tfa.layers.GroupNormalization(kernum//8)(x)
    x = attentionblock(x)
    return x
def model_9class(length=15000, dim=12,classnum=9,kerlen=5,poolsize=2,filnum=[32,32,64,128]):
    myinput = tf.keras.layers.Input(shape=(length,dim))
    x0 = resblock(myinput, kerlen=kerlen, kernum=filnum[0], poolsize=poolsize)
    x1 = resblock(x0, kerlen=kerlen, kernum=filnum[1], poolsize=poolsize)
    x2 = resblock(x1, kerlen=kerlen, kernum=filnum[2], poolsize=poolsize)
    x3 = resblock(x2, kerlen=kerlen, kernum=filnum[3], poolsize=poolsize)
    x40 = resblock(x0,kerlen=kerlen, kernum=filnum[-1], poolsize=poolsize**4)
    x41 = resblock(x1,kerlen=kerlen, kernum=filnum[-1], poolsize=poolsize**3)
    x42 = resblock(x2,kerlen=kerlen, kernum=filnum[-1], poolsize=poolsize**2)
    x43 = resblock(x3,kerlen=kerlen, kernum=filnum[-1], poolsize=poolsize)
    x4 = x40 + x41 + x42 + x43
    x4 = tfa.layers.GroupNormalization()(x4)
    y0 = tf.keras.layers.GlobalAveragePooling1D()(x4)
    y0 = multidropout(y0,[0.0,0.2,0.4,0.8])
    y = tf.keras.layers.Dense(classnum, activation='linear')(y0)
    y = tf.keras.layers.Activation('softmax')(y)
    mymodel = tf.keras.Model(inputs=myinput, outputs=y)
    return mymodel