import pickle

from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from constants import Constants
from utils import *
from data_generator import create_test_data

F=tf.cast(np.exp(Constants.Y).reshape(1,Constants.N,Constants.N,1),tf.dtypes.float32)
#dFy_diff=tf_diff(F,axis=2,rank=4)

dFy_conv = Dy(F, Constants.FILTER_YEE)

class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        x, y=inputs
        print(type(x))

        return matmul(x, self.w, self.b)

def matmul(A,B,C):
    return tf.matmul(A, B) + C

inputs1 = keras.Input(shape=(784,), name="digits1")
inputs2 = keras.Input(shape=(784,), name="digits2")
x = Linear(32,784)([inputs1, inputs2])
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=[inputs1, inputs2], outputs=[outputs])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation
x_val = x_train[-100:]
y_val = y_train[-100:]
x_train = x_train[:-100]
y_train = y_train[:-100]

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print("Fit model on training data")
history = model.fit(
    [x_train, x_train],
    [y_train],
    batch_size=64,
    epochs=2
)