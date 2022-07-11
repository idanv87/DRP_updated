import pickle

import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from constants import Constants
from utils import MAIN_LAYER, custom_loss


with open('/Users/idanversano/documents/pycharm/files/ex.pkl', 'rb') as file:
    ex = tf.cast(pickle.load(file), tf.dtypes.float64)
with open('/Users/idanversano/documents/pycharm/files/ey.pkl', 'rb') as file:
    ey = tf.cast(pickle.load(file), tf.dtypes.float64)
with open('/Users/idanversano/documents/pycharm/files/hx_x.pkl', 'rb') as file:
    hx_x = tf.cast(pickle.load(file), tf.dtypes.float64)
with open('/Users/idanversano/documents/pycharm/files/hy_x.pkl', 'rb') as file:
    hy_x = tf.cast(pickle.load(file), tf.dtypes.float64)
with open('/Users/idanversano/documents/pycharm/files/hx_y.pkl', 'rb') as file:
    hx_y = tf.cast(pickle.load(file), tf.dtypes.float64)
with open('/Users/idanversano/documents/pycharm/files/hy_y.pkl', 'rb') as file:
    hy_y = tf.cast(pickle.load(file), tf.dtypes.float64)
with open('/Users/idanversano/documents/pycharm/files/inte.pkl', 'rb') as file:
    inte_test = tf.cast(pickle.load(file), tf.dtypes.float64)
with open('/Users/idanversano/documents/pycharm/files/inth.pkl', 'rb') as file:
    inth_test = tf.cast(pickle.load(file), tf.dtypes.float64)

E_input = keras.Input(shape=(Constants.N, Constants.N, 1), name="e")
Hx_input = keras.Input(shape=(Constants.N - 2, Constants.N - 1, 1), name="hx")
Hy_input = keras.Input(shape=(Constants.N - 1, Constants.N - 2, 1), name="hy")
layer1 = MAIN_LAYER()
layer2 = MAIN_LAYER()
E_output = layer1([E_input, Hx_input, Hy_input])[0]
Hx_output = layer2([E_input, Hx_input, Hy_input])[1]
Hy_output = layer2([E_input, Hx_input, Hy_input])[2]
inte_output=layer1([E_input, Hx_input, Hy_input])[3]




model = keras.Model(
    inputs=[E_input, Hx_input, Hy_input],
    outputs=[E_output, Hx_output, Hy_output, inte_output]
)
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=1e-2),
    loss=[custom_loss, custom_loss, custom_loss, tf.keras.losses.MeanAbsoluteError()]
)
if __name__ == "__main__":
    history = model.fit(
        [ex, hx_x, hy_x], [ey, hx_y, hy_y, inte_test],
        epochs=2,
        batch_size=64,
        shuffle=True, validation_split=0.2)
    print(model.trainable_weights)
    model.save('/Users/idanversano/documents/pycharm/files/mymodel_multiple')
    pickle.dump(history.history, open("/Users/idanversano/documents/pycharm/files/multiple_history.pkl", "wb"))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))





















# optimizer = keras.optimizers.SGD(learning_rate=1e-2)
# epochs = 2
# #e, hx, hy, inte = model(([ex, hx_x, hy_x]), training=True)
# #print(tf.math.reduce_max(abs(ey- e)))
# for epoch in range(epochs):
#
#     with tf.GradientTape() as tape:
#         e, hx, hy, inte= model(([ex, hx_x, hy_x]))
#         print(tf.math.reduce_mean(abs(inte-inte_test)))
#
#         #inth=model.call(([ex, hx_x, hy_x]), training=True)[3]
#         #print(model.trainable_weights)
#         loss_value = custom_loss(ey,e)+ custom_loss(hx_y,hx)+ custom_loss(hy_y,hy)
#         grads = tape.gradient(loss_value, model.trainable_weights)
#         #print(loss_value)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))

#raise Exception("The number shouldn't be an odd integer")
# history=model.fit(
#     [ex[1:100,:,:,:],hx_x[1:100,:,:,:], hy_x[1:100,:,:,:]],[ey[1:100,:,:,:]]],
#     epochs=40,
#     batch_size=5,
#     shuffle=True
# )
# model.evaluate([ex,hx_x, hy_x], [ey], batch_size=2)

# model = keras.Model(
#     inputs=[E_input, Hx_input, Hy_input],
#     outputs=[E_output,Hx_output, Hy_output]
# )
#
# model.compile(
#     optimizer=keras.optimizers.SGD(1e-3),
#     loss= [keras.losses.MeanSquaredError(),keras.losses.MeanSquaredError(), keras.losses.MeanSquaredError()]
# )
# history=model.fit(
#     [ex[1:10,:,:,:],hx_x[1:10,:,:,:], hy_x[1:10,:,:,:]],[ey[1:10,:,:,:],hx_y[1:10,:,:,:],hy_y[1:10,:,:,:]],
#     epochs=10,
#     batch_size=1
# )
# print(model.trainable_weights)
#

