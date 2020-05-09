# COMP39X Final Year Project
# DefenseGAN project
# Maizhen Ning
# define several CNN model for test the defense-GAN algorithm

import tensorflow as tf 
from tensorflow.keras import layers


def get_model(model_type, batchNorm=True):
    models = {
        'A': model_a, 'B': model_b, 'C': model_c, 'D': model_d, 
        'E': model_e, 'F': model_f}

    model = models[model_type](batchNorm)
    return model

def model_a(batchNorm):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same',input_shape=[28, 28, 1]))
    model.add(layers.ReLU())

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='valid'))
    model.add(layers.ReLU())
    model.add(layers.Flatten())
    
    model.add(layers.Dropout(0.25)) #further added dropout layer
    model.add(layers.Dense(128))
    
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.5))

    if batchNorm:
        model.add(layers.BatchNormalization())

    model.add(layers.Dense(10, activation='softmax'))
    return model

def model_b(batchNorm):
    model = tf.keras.Sequential()
    model.add(layers.Dropout(0.2, input_shape=[28, 28, 1]))

    model.add(layers.Conv2D(64, (8, 8), strides=(2, 2), padding='same'))
    model.add(layers.ReLU())

    model.add(layers.Conv2D(128, (6, 6), strides=(2, 2), padding='valid'))
    model.add(layers.ReLU())

    model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='valid'))
    model.add(layers.ReLU())

    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())

    if batchNorm:
        model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(10, activation='softmax'))
    return model

def model_c(batchNorm):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',input_shape=[28, 28, 1]))
    model.add(layers.ReLU())

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='valid'))
    model.add(layers.ReLU())

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128))
    model.add(layers.ReLU())

    model.add(layers.Dropout(0.5))
    if batchNorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))
    return model

def model_d(batchNorm):
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28, 1]))
    model.add(layers.Dense(200))
    model.add(layers.ReLU())   

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(200))
    model.add(layers.ReLU())  

    if batchNorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))
    return model

def model_e(batchNorm):
    #removed dropout layer from model_d
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28, 1]))
    model.add(layers.Dense(200))
    model.add(layers.ReLU())   

    model.add(layers.Dense(200))
    model.add(layers.ReLU())  

    if batchNorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))
    return model

def model_f(batchNorm):
    #removed dropout layer from model_b
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (8, 8), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.ReLU())

    model.add(layers.Conv2D(128, (6, 6), strides=(2, 2), padding='valid'))
    model.add(layers.ReLU())

    model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='valid'))
    model.add(layers.ReLU())

    model.add(layers.Flatten())

    if batchNorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))
    return model