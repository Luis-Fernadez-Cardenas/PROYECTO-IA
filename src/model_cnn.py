

import tensorflow as tf
from tensorflow.keras import layers, models

def bloque_conv(x, filtros, k=7, s=1, p='same', dropout=0.1):
    x = layers.Conv1D(filtros, k, strides=s, padding=p, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x

def crear_modelo(input_shape=(5000, 12), n_classes=3):
    """
    Modelo CNN 1D para ECG (multi-etiqueta): salida sigmoide + binary_crossentropy
    """
    inp = layers.Input(shape=input_shape)

    x = bloque_conv(inp, 32,  k=11, dropout=0.1)
    x = bloque_conv(x,   64,  k=9,  dropout=0.1)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = bloque_conv(x,   64,  k=7,  dropout=0.1)
    x = bloque_conv(x,  128,  k=5,  dropout=0.1)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = bloque_conv(x,  128,  k=5,  dropout=0.15)
    x = bloque_conv(x,  256,  k=3,  dropout=0.15)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.25)(x)

    # Multi-etiqueta (CD, STTC, NORM pueden representarse como vector multi-hot)
    out = layers.Dense(n_classes, activation='sigmoid')(x)

    model = models.Model(inp, out, name='ecg_cnn_multilabel')
    return model
