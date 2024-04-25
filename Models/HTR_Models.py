
import tensorflow as tf
import numpy as np

# HTR Dependency
import cv2
import string
import h5py

keras = tf.keras
print(tf.__version__)


class FullGatedConv2D(keras.layers.Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, filters, **kwargs):
        super(FullGatedConv2D, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        """Apply gated convolution"""
        output = super(FullGatedConv2D, self).call(inputs)
        linear = keras.layers.Activation("linear")(
            output[:, :, :, :self.nb_filters])
        sigmoid = keras.layers.Activation("sigmoid")(
            output[:, :, :, self.nb_filters:])

        return keras.layers.Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output"""
        output_shape = super(
            FullGatedConv2D, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters,)

    def get_config(self):
        """Return the config of the layer"""
        config = super(FullGatedConv2D, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config


def FlorHTR(input_shape, output_shape):
    input_data = keras.layers.Input(name="input", shape=input_shape)
    cnn = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=40, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=56, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)

    cnn = keras.layers.MaxPooling2D(pool_size=(
        1, 2), strides=(1, 2), padding="valid")(cnn)

    shape = cnn.get_shape()
    bgru = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = keras.layers.TimeDistributed(keras.layers.Dense(units=128))(bgru)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = keras.layers.TimeDistributed(
        keras.layers.Dense(units=output_shape, activation="softmax"))(bgru)
    return (input_data, output_data)


def FlorHTR2(input_shape, output_shape):
    input_data = keras.layers.Input(name="input", shape=input_shape)
    cnn = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=56, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)

    cnn = keras.layers.MaxPooling2D(pool_size=(
        1, 2), strides=(1, 2), padding="valid")(cnn)

    shape = cnn.get_shape()
    bgru = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = keras.layers.TimeDistributed(keras.layers.Dense(units=128))(bgru)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = keras.layers.TimeDistributed(
        keras.layers.Dense(units=output_shape, activation="softmax"))(bgru)
    return (input_data, output_data)


def SmallFlorHTR(input_shape, output_shape):
    input_data = keras.layers.Input(name="input", shape=input_shape)

    cnn = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(
        3, 3), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)

    cnn = keras.layers.MaxPooling2D(pool_size=(
        1, 2), strides=(1, 2), padding="valid")(cnn)

    shape = cnn.get_shape()
    bgru = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = keras.layers.TimeDistributed(keras.layers.Dense(units=128))(bgru)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=64, return_sequences=True, dropout=0.5))(bgru)
    output_data = keras.layers.TimeDistributed(
        keras.layers.Dense(units=output_shape, activation="softmax"))(bgru)
    return (input_data, output_data)


def PuigCerver(input_shape, output_shape):
    """
    Convolucional Recurrent Neural Network by Puigcerver et al.

    Reference:
        Puigcerver, J.: Are multidimensional recurrent layers really
        necessary for handwritten text recognition? In: Document
        Analysis and Recognition (ICDAR), 2017 14th
        IAPR International Conference on, vol. 1, pp. 67–72. IEEE (2017)
    """

    input_data = keras.layers.Input(name="input", shape=input_shape)

    cnn = keras.layers.Conv2D(filters=16, kernel_size=(3, 3),
                              strides=(1, 1), padding="same")(input_data)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = keras.layers.LeakyReLU(alpha=0.01)(cnn)
    cnn = keras.layers.MaxPooling2D(pool_size=(
        2, 2), strides=(2, 2), padding="valid")(cnn)

    cnn = keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                              strides=(1, 1), padding="same")(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = keras.layers.LeakyReLU(alpha=0.01)(cnn)
    cnn = keras.layers.MaxPooling2D(pool_size=(
        2, 2), strides=(2, 2), padding="valid")(cnn)

    cnn = keras.layers.Dropout(rate=0.2)(cnn)
    cnn = keras.layers.Conv2D(filters=48, kernel_size=(3, 3),
                              strides=(1, 1), padding="same")(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = keras.layers.LeakyReLU(alpha=0.01)(cnn)
    cnn = keras.layers.MaxPooling2D(pool_size=(
        2, 2), strides=(2, 2), padding="valid")(cnn)

    cnn = keras.layers.Dropout(rate=0.2)(cnn)
    cnn = keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                              strides=(1, 1), padding="same")(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = keras.layers.LeakyReLU(alpha=0.01)(cnn)

    cnn = keras.layers.Dropout(rate=0.2)(cnn)
    cnn = keras.layers.Conv2D(filters=80, kernel_size=(3, 3),
                              strides=(1, 1), padding="same")(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = keras.layers.LeakyReLU(alpha=0.01)(cnn)

    shape = cnn.get_shape()
    blstm = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = keras.layers.Bidirectional(
        keras.layers.LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = keras.layers.Bidirectional(
        keras.layers.LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = keras.layers.Bidirectional(
        keras.layers.LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = keras.layers.Bidirectional(
        keras.layers.LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = keras.layers.Bidirectional(
        keras.layers.LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)

    blstm = keras.layers.Dropout(rate=0.5)(blstm)
    output_data = keras.layers.Dense(
        units=output_shape, activation="softmax")(blstm)

    return (input_data, output_data)


def Terng_HTR(input_shape, output_shape):
    input_data = keras.layers.Input(name="input", shape=input_shape)

    cnn = keras.layers.Conv2D(filters=56, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=40, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)

    cnn = keras.layers.MaxPooling2D(pool_size=(
        1, 2), strides=(1, 2), padding="valid")(cnn)

    shape = cnn.get_shape()
    bgru = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = keras.layers.TimeDistributed(keras.layers.Dense(units=128))(bgru)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = keras.layers.TimeDistributed(
        keras.layers.Dense(units=output_shape, activation="softmax"))(bgru)
    return (input_data, output_data)


def Terng_HTR2(input_shape, output_shape):
    # BAD

    input_data = keras.layers.Input(name="input", shape=input_shape)

    conv1 = keras.layers.Conv2D(
        32, (3, 3), activation='relu', padding='same')(input_data)
    conv1 = keras.layers.Dropout(0.2)(conv1)

    conv1 = keras.layers.Conv2D(
        32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Dropout(0.2)(conv2)
    conv2 = keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D((2, 2))(conv2)

    conv3 = keras.layers.Conv2D(
        128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Dropout(0.2)(conv3)
    conv3 = keras.layers.Conv2D(
        128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = keras.layers.concatenate(
        [keras.layers.UpSampling2D((2, 2))(conv3), conv2], axis=-1)
    conv4 = keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = keras.layers.Dropout(0.2)(conv4)
    conv4 = keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = keras.layers.concatenate(
        [keras.layers.UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = keras.layers.Conv2D(
        32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = keras.layers.Dropout(0.2)(conv5)
    conv5 = keras.layers.Conv2D(
        32, (3, 3), activation='relu', padding='same')(conv5)

    shape = conv5.get_shape()
    bgru = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(conv5)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = keras.layers.TimeDistributed(keras.layers.Dense(units=128))(bgru)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = keras.layers.TimeDistributed(
        keras.layers.Dense(units=output_shape, activation="softmax"))(bgru)
    return (input_data, output_data)


def Terng_HTR3(input_shape, output_shape):
    input_data = keras.layers.Input(name="input", shape=input_shape)

    cnn = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=8, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=40, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=56, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)

    cnn = keras.layers.MaxPooling2D(pool_size=(
        1, 2), strides=(1, 2), padding="valid")(cnn)

    shape = cnn.get_shape()
    bgru = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = keras.layers.TimeDistributed(keras.layers.Dense(units=128))(bgru)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = keras.layers.TimeDistributed(
        keras.layers.Dense(units=output_shape, activation="softmax"))(bgru)
    return (input_data, output_data)


def FlorResHTR(input_shape, output_shape):
    input_data = keras.layers.Input(name="input", shape=input_shape)
    cnn = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn1 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn1)
    res1 = keras.layers.add([cnn1, cnn])

    cnn = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(res1)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn2 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn2)
    res2 = keras.layers.add([cnn2, cnn])

    cnn = keras.layers.Conv2D(filters=40, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(res2)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn3 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn3)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)
    res3 = keras.layers.add([cnn3, cnn])

    cnn = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(res3)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn4 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn4)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)
    res4 = keras.layers.add([cnn4, cnn])

    cnn = keras.layers.Conv2D(filters=56, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn5 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn5)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)
    res5 = keras.layers.add([cnn5, cnn])

    cnn = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)

    cnn = keras.layers.MaxPooling2D(pool_size=(
        1, 2), strides=(1, 2), padding="valid")(cnn)

    shape = cnn.get_shape()
    bgru = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = keras.layers.TimeDistributed(keras.layers.Dense(units=128))(bgru)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = keras.layers.TimeDistributed(
        keras.layers.Dense(units=output_shape, activation="softmax"))(bgru)
    return (input_data, output_data)

# extend encoder


def ExtendFlorHTR(input_shape, output_shape):
    input_data = keras.layers.Input(name="input", shape=input_shape)
    cnn = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=40, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=56, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)

    cnn = FullGatedConv2D(filters=64, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.MaxPooling2D(pool_size=(
        1, 2), strides=(1, 2), padding="valid")(cnn)

    cnn = keras.layers.Conv2D(filters=72, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)

    shape = cnn.get_shape()
    bgru = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = keras.layers.TimeDistributed(keras.layers.Dense(units=128))(bgru)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = keras.layers.TimeDistributed(
        keras.layers.Dense(units=output_shape, activation="softmax"))(bgru)
    return (input_data, output_data)


def FlorResAcHTR(input_shape, output_shape):

    # https://arxiv.org/pdf/1512.03385.pdf
    input_data = keras.layers.Input(name="input", shape=input_shape)
    cnn = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn1 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn1)
    res1 = keras.layers.add([cnn1, cnn])

    rac = keras.layers.PReLU(shared_axes=[1, 2])(res1)

    cnn = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(rac)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn2 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn2)
    res2 = keras.layers.add([cnn2, cnn])

    rac = keras.layers.PReLU(shared_axes=[1, 2])(res2)

    cnn = keras.layers.Conv2D(filters=40, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(rac)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn3 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn3)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)
    res3 = keras.layers.add([cnn3, cnn])

    rac = keras.layers.PReLU(shared_axes=[1, 2])(res3)

    cnn = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(rac)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn4 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn4)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)
    res4 = keras.layers.add([cnn4, cnn])

    rac = keras.layers.PReLU(shared_axes=[1, 2])(res4)

    cnn = keras.layers.Conv2D(filters=56, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(rac)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn5 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn5)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)
    res5 = keras.layers.add([cnn5, cnn])

    rac = keras.layers.PReLU(shared_axes=[1, 2])(res5)

    cnn = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(rac)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)

    cnn = keras.layers.MaxPooling2D(pool_size=(
        1, 2), strides=(1, 2), padding="valid")(cnn)

    shape = cnn.get_shape()
    bgru = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = keras.layers.TimeDistributed(keras.layers.Dense(units=128))(bgru)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = keras.layers.TimeDistributed(
        keras.layers.Dense(units=output_shape, activation="softmax"))(bgru)
    return (input_data, output_data)
