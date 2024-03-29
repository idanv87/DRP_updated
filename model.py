import pickle
import os
import time

from keras import callbacks
import numpy as np
import tensorflow as tf
from tensorflow import keras

from constants import Constants
from utils import DRP_LAYER, custom_loss, custom_loss3
from drp import calculate_DRP

path = Constants.PATH




# matplotlib.use("TkAgg")
l = {"N": Constants.N, "CFL": Constants.CFL}
model_details = {"name": '1001_N='+str(l['N']), "net_num": 1, "energy_loss": False, "div_loss": False, "div_preserve": True,
                 "initial_": -0.125, "params": l, "options": 'lt', "number_oututs": 6}
name = model_details["name"]

saving_path = path + 'Experiment_' + name + '_details/'
isExist = os.path.exists(saving_path)
if not isExist:
    os.makedirs(saving_path)

pickle.dump(model_details, open(saving_path + 'experiment_' + name + '_details' + '.pkl', "wb"))


if Constants.DTYPE == tf.dtypes.float64:
    tf.keras.backend.set_floatx('float64')
else:
    tf.keras.backend.set_floatx('float32')



with open(path + 'train/input.pkl', 'rb') as file:
    net_input = pickle.load(file)
with open(path + 'train/output.pkl', 'rb') as file:
    net_output= pickle.load(file)










#div_y = tf.zeros([X['e_x'].shape[0], Constants.N - 3, Constants.N - 3, 1], dtype=Constants.DTYPE)


start_time = time.time()
for k in range(Constants.CROSS_VAL):
    E1_input = keras.Input(shape=(Constants.N, Constants.N, 1))
    Hx1_input = keras.Input(shape=(Constants.N - 2, Constants.N - 1, 1))
    Hy1_input = keras.Input(shape=(Constants.N - 1, Constants.N - 2, 1))
    E2_input = keras.Input(shape=(Constants.N, Constants.N, 1))
    Hx2_input = keras.Input(shape=(Constants.N - 2, Constants.N - 1, 1))
    Hy2_input = keras.Input(shape=(Constants.N - 1, Constants.N - 2, 1))
    E3_input = keras.Input(shape=(Constants.N, Constants.N, 1))
    Hx3_input = keras.Input(shape=(Constants.N - 2, Constants.N - 1, 1))
    Hy3_input = keras.Input(shape=(Constants.N - 1, Constants.N - 2, 1))

    layer1 = DRP_LAYER()
    output = layer1([E1_input, Hx1_input, Hy1_input, E2_input, Hx2_input, Hy2_input, E3_input, Hx3_input, Hy3_input])

    E1_output = output[0]
    Hx1_output = output[1]
    Hy1_output = output[2]

    E2_output = output[3]
    Hx2_output = output[4]
    Hy2_output = output[5]

    E3_output = output[6]
    Hx3_output = output[7]
    Hy3_output = output[8]

    # div_output=output[6]
    # energy_output = output[6]

    model = keras.Model(
        inputs=[E1_input, Hx1_input, Hy1_input, E2_input, Hx2_input, Hy2_input, E3_input, Hx3_input, Hy3_input],
        outputs=[E1_output, Hx1_output, Hy1_output, E2_output, Hx2_output, Hy2_output
           # ,E3_output, Hx3_output, Hy3_output
                 ]
        # outputs = [E_output, Hx_output, Hy_output, energy_output]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        # loss=[custom_loss, custom_loss, custom_loss],
        loss=[custom_loss, custom_loss, custom_loss,
              custom_loss, custom_loss,custom_loss
            #  ,custom_loss, custom_loss,custom_loss
              ]
    )
    if k==0:
       model.save(saving_path + 'model.pkl')

    # model.load_weights(path + 'mymodel_weights2.pkl').expect_partial()

    earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                            mode="min", patience=5,
                                            restore_best_weights=False)

    checkpoint_filepath = saving_path + 'model_weights_val_number_' + str(k) + '.pkl'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    # csv loger
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    history = model.fit(
        net_input, net_output[:-4],
        callbacks=[earlystopping, model_checkpoint_callback, reduce_lr],
        epochs=Constants.EPOCHS,
        batch_size=Constants.BATCH_SIZE,
        shuffle=True, validation_split=0.2, verbose=2)

print("--- %s seconds ---" % (time.time() - start_time))

model.load_weights(saving_path + 'model_weights_val_number_' + str(0) + '.pkl').expect_partial()
print(model.trainable_weights)
print(calculate_DRP())

# if __name__ == "__main__":
#     start_time = time.time()
#
#     history = model.fit(
#         [train_data['ex'], train_data['hx_x'], train_data['hy_x']],
#          [train_data['ey1'], train_data['hx_y1'], train_data['hy_y1'], train_data['ey2'], train_data['hx_y2'], train_data['hy_y2']],
#         callbacks=[earlystopping, model_checkpoint_callback],
#         # [ex, hx_x, hy_x], [ey, hx_y, hy_y, energy_y],
#         epochs=100,
#         batch_size=32,
#         shuffle=True, validation_split=0.2, verbose=2)
#
#     print("--- %s seconds ---" % (time.time() - start_time))
#
#     # pickle.dump(history.history, open(path + 'multiple_history.pkl', "wb"))
#     # plt.plot(history.history['loss'])
#     # plt.plot(history.history['val_loss'])
#     # plt.show()
#
#     # trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
#     # non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
#     # print('Total params: {:,}'.format(trainable_count + non_trainable_count))
#     # print('Trainable params: {:,}'.format(trainable_count))
#     # print('Non-trainable params: {:,}'.format(non_trainable_count))
#     print(model.trainable_weights)

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

# raise Exception("The number shouldn't be an odd integer")
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
#print(model.trainable_weights)
#
