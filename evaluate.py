import tensorflow as tf
from utils import MyDenseLayer
from tensorflow import keras
import numpy as np

layer = MyDenseLayer(10)
E_input = keras.Input(shape=(20,1), name="E")
E_output=layer(E_input)
model = keras.Model(
    inputs=[E_input],
    outputs=[E_output]
)
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=1000),
    loss= keras.losses.MeanAbsoluteError()
)
