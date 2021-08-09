
from tensorflow.python.keras.layers.merge import Average
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
from keras.models import Sequential
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from scipy import signal
import pandas as pd
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, AveragePooling1D, SeparableConv1D, Dropout, Flatten, Concatenate, Reshape, \
    Activation, BatchNormalization, SeparableConv1D, Add, Activation, GlobalAveragePooling1D
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam, SGD
import numpy as np
import os
import sys
import joblib
from scipy.signal import butter, sosfiltfilt
import math
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
import tensorflow_addons as tfa
from sklearn.preprocessing import normalize
from create_training_data import create_training_data_time
import visualkeras
from collections import defaultdict


def scale_between_0_1(arr):
    return normalize(arr, norm="max")


def assign_weigths(sums):
    if np.count_nonzero(sums) != 2:
        sums[np.where(sums == 0)[0]] = 1
    total = np.sum(sums)
    sums_div = 1 / (sums / total)
    weights_dict = {}
    for i in range(len(sums)):
        weights_dict.__setitem__(i, sums_div[i] / 2)
    return weights_dict


def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 20
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


def inception_module_1(layer_in):
    conv4 = Conv1D(32, 4, padding='same', activation='relu',
                   kernel_initializer='GlorotNormal')(layer_in)
    conv16 = Conv1D(32, 16, padding='same', activation='relu',
                    kernel_initializer='GlorotNormal')(layer_in)
    layer_out = concatenate([conv4, conv16], axis=-1)
    x3 = BatchNormalization()(layer_out)
    return x3


def res_net_block1(input_data, filters, conv_size):
    x = Conv1D(filters, conv_size, activation='relu',
               padding='same')(input_data)
    x = BatchNormalization()(x)
    x = Conv1D(filters, conv_size, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_data])
    x = Activation('relu')(x)
    return x


def res_net_block_trans(input_data, filters, conv_size):
    input_trans = Conv1D(filters, 1, activation='relu',
                         padding='same')(input_data)
    x0 = Conv1D(filters, conv_size, activation='relu',
                padding='same')(input_data)
    x1 = BatchNormalization()(x0)
    x2 = Conv1D(filters, conv_size, activation=None, padding='same')(x1)
    x3 = BatchNormalization()(x2)
    x4 = Add()([x3, input_trans])
    x = Activation('relu')(x4)
    return x


def Lead_II_way(lead_II):
    layer_out = Conv1D(32, 8, activation='relu',
                       kernel_initializer='GlorotNormal', padding='same')(lead_II)
    Batch1 = BatchNormalization()(layer_out)
    layer_out_0 = res_net_block_trans(Batch1, 32, 8)
    layer_out_1 = res_net_block_trans(layer_out_0, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(layer_out_1)
    #Incept_1 = inception_module_1(Pool1)
    res2 = res_net_block_trans(Pool1, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(res2)
    # flat = Flatten()(Pool1)
    return Pool1


def Lead_V5_way(lead_V5):
    layer_out = Conv1D(32, 8, activation='relu',
                       kernel_initializer='GlorotNormal', padding='same')(lead_V5)
    Batch1 = BatchNormalization()(layer_out)
    layer_out_0 = res_net_block_trans(Batch1, 32, 8)
    layer_out_1 = res_net_block_trans(layer_out_0, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(layer_out_1)
    #Incept_1 = inception_module_1(Pool1)
    res2 = res_net_block_trans(Pool1, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(res2)
    # Pool1 = signal.decimate(res2, 2)
    # flat = Flatten()(Pool1)
    return Pool1


def define_model(in_shape=(1251, 1,), out_shape=2):
    input_II = Input(shape=in_shape)
    input_V5 = Input(shape=in_shape)

    out_II = Lead_II_way(input_II)
    out_V5 = Lead_V5_way(input_V5)
    layer_out = concatenate([out_II, out_V5], axis=-1)
    sep1 = SeparableConv1D(128, 4, activation='relu',
                           kernel_initializer='GlorotNormal', padding='same')(layer_out)
    flat = Flatten()(sep1)
    Dense_1 = Dense(128, activation='relu')(flat)
    Dropout1 = Dropout(0.4)(Dense_1)
    out = Dense(out_shape, activation='softmax')(Dropout1)
    model = Model(inputs=[input_II, input_V5], outputs=out)
    # compile model
    opt = Adam(learning_rate=0.0003)
    # opt = SGD(lr=0.01, momentum=0.9, nesterov=False)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[
        'Recall', "accuracy", tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='macro')])
    return model


model = define_model()
color_map = defaultdict(dict)
color_map[Conv1D]['fill'] = 'red'
color_map[Dropout]['fill'] = 'pink'
color_map[MaxPooling1D]['fill'] = 'brown'
color_map[MaxPooling1D]['fill'] = 'brown'
color_map[AveragePooling1D]['fill'] = 'brown'
color_map[Dense]['fill'] = 'green'
color_map[Flatten]['fill'] = 'cyan'
color_map[Activation]['fill'] = 'blue'
color_map[SeparableConv1D]['fill'] = 'lightgoldenrodyellow'
color_map[BatchNormalization]['fill'] = 'yellowgreen'
color_map[Add]['fill'] = 'purple'
color_map[tf.keras.layers.Input]['fill'] = 'cyan'


visualkeras.graph_view(model, connector_width=2,
                       show_neurons=True, to_file="network.png", color_map=color_map)

print(model.summary())
