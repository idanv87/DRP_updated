import numpy as np
import tensorflow as tf
from tensorflow import keras


from constants import Constants


def trapz(f,x):

    if len(f.shape) == 1:
        elem = tf.cast(tf.range(0), tf.dtypes.float64)
    else:
        elem = tf.cast(tf.range(f.shape[1]), tf.dtypes.float64)
    dx = Constants.DX
    T = tf.map_fn(fn=lambda k: (dx / 2) * tf.math.reduce_sum(f[1:, int(k)] + f[:-1, int(k)], axis=0), elems=elem)
    return T


def trapz2(f, x,y):
    return trapz(tf.reshape(trapz(f, x), [f.shape[1], 1]), y)


def trapz2_batch(f, x,y):
    if f.shape[0] == None:
        T = trapz2(f[-1, :, :, 0], x,y)
    else:
        #elem = tf.cast(np.arange(f.shape[0]), tf.dtypes.float64)
        elem = np.arange(f.shape[0]).astype('float64')
        T = tf.map_fn(fn=lambda k: trapz2(f[int(k), :, :, 0], x,y), elems=elem)
    return T


def amper(E, Hx, Hy, par1, par2):

    pad1 = pad_function([2, 2, 2, 2])
    pad5 = pad_function([Constants.N - 2, 1, 2, 2])
    pad6 = pad_function([2, 2, 1, Constants.N - 2])
    pad7 = pad_function([2, 2, Constants.N - 2, 1])
    pad4 = pad_function([1, Constants.N - 2, 2, 2])

    x1=tf.math.multiply(par1, Dx(Hy, tf.transpose(Constants.FILTER_BETA, perm=[1, 0, 2, 3])))
    x2 = tf.math.multiply(par2, Dx(Hy, tf.transpose(Constants.FILTER_DELTA, perm=[1, 0, 2, 3])))
    x3 = Dx(Hy, tf.transpose(Constants.FILTER_YEE, perm=[1, 0, 2, 3]))

    s1 = tf.pad(x1+x2+x3, pad1) + \
         tf.pad(Dx(Hy, tf.transpose(Constants.KERNEL_FORWARD, perm=[1, 0, 2, 3])), Constants.PADY_FORWARD) + \
         tf.pad(Dx(Hy, tf.transpose(Constants.KERNEL_BACKWARD, perm=[1, 0, 2, 3])), Constants.PADY_BACWARD) + \
         tf.pad(Dx(Hy, tf.transpose(Constants.FOURTH_UP, perm=[1, 0, 2, 3])), pad6) + \
         tf.pad(Dx(Hy, tf.transpose(Constants.FOURTH_DOWN, perm=[1, 0, 2, 3])), pad7)

    x1=tf.math.multiply(par1, Dy(Hx, Constants.FILTER_BETA))
    x2=tf.math.multiply(par2, Dy(Hx, Constants.FILTER_DELTA))
    x3=Dy(Hx, Constants.FILTER_YEE)

    s2 = tf.pad(x1+x2+x3, pad1) + \
         tf.pad(Dy(Hx, Constants.KERNEL_FORWARD), Constants.PADX_FORWARD) + \
         tf.pad(Dy(Hx, Constants.KERNEL_BACKWARD), Constants.PADX_BACWARD) + \
         tf.pad(Dy(Hx, Constants.FOURTH_UP), pad4) + \
         tf.pad(Dy(Hx, Constants.FOURTH_DOWN), pad5)
    return E + Constants.DT * (s1 - s2)


def faraday(E, Hx, Hy, beta, delta):
    pad2 = pad_function([0, 0, 1, 1])
    pad3 = pad_function([1, 1, 0, 0])

    x1 = tf.math.multiply(beta, Dy(E, Constants.FILTER_BETA))
    x2=tf.math.multiply(delta, Dy(E, Constants.FILTER_DELTA))
    x3= Dy(E, Constants.FILTER_YEE)

    s3 = tf.pad(x1+x2+x3, pad2) + \
         tf.pad(Dy(E, Constants.KERNEL_E_FORWARD), Constants.PADEX_FORWARD)[:, 1:-1, :, :] + \
         tf.pad(Dy(E, Constants.KERNEL_E_BACKWARD), Constants.PADEX_BACKWARD)[:, 1:-1, :, :]

    x1=tf.math.multiply(beta, Dx(E, tf.transpose(Constants.FILTER_BETA, perm=[1, 0, 2, 3])))
    x2 = tf.math.multiply(delta, Dx(E, tf.transpose(Constants.FILTER_DELTA, perm=[1, 0, 2, 3])))
    x3 = Dx(E, tf.transpose(Constants.FILTER_YEE, perm=[1, 0, 2, 3]))

    s4 = tf.pad(x1+x2+x3, pad3) + \
         tf.pad(Dx(E, tf.transpose(Constants.KERNEL_E_FORWARD, perm=[1, 0, 2, 3])), Constants.PADEY_FORWARD)[:,
         :, 1:-1, :] + \
         tf.pad(Dx(E, tf.transpose(Constants.KERNEL_E_BACKWARD, perm=[1, 0, 2, 3])), Constants.PADEY_BACKWARD)[
         :, :, 1:-1, :]

    return Hx - Constants.DT * s3, Hy + Constants.DT * s4


def Dy(B, kernel):
    return tf.nn.conv2d(tf.cast(B, tf.dtypes.float64), kernel, strides=1, padding='VALID')


def Dx(B, kernel):
    return tf.nn.conv2d(tf.cast(B, tf.dtypes.float64), kernel, strides=1, padding='VALID')


def f_a(c, n, k1, k2):
    e = c * np.cos(c * n * Constants.DT) * (
            np.sin(Constants.PI * k1 * Constants.X) * np.sin(Constants.PI * k2 * Constants.Y) +
            np.sin(Constants.PI * k2 * Constants.X) * np.sin(
        Constants.PI * k1 * Constants.Y))

    hx = np.sin(c * (Constants.DT / 2) * (2 * n + 1)) * (
            -Constants.PI * k2 * np.sin(Constants.PI * k1 * Constants.X) * np.cos(
        Constants.PI * k2 * (Constants.Y + Constants.DX / 2)) - Constants.PI * k1 * np.sin(
        Constants.PI * k2 * Constants.X) * np.cos(Constants.PI * k1 * (Constants.Y + Constants.DX / 2)))

    hy = np.sin(c * (Constants.DT / 2) * (2 * n + 1)) * (
            Constants.PI * k1 * np.cos(Constants.PI * k1 * (Constants.X + Constants.DX / 2)) * np.sin(
        Constants.PI * k2 * Constants.Y) + Constants.PI * k2 * np.cos(
        Constants.PI * k2 * (Constants.X + Constants.DX / 2)) * np.sin(Constants.PI * k1 * Constants.Y))

    if k1 == k2:
        err2 = c ** 2 * (np.sin(c * (2 * n + 1) * Constants.DT / 2) ** 2)
        err1 = c ** 2 * (np.cos(c * n * Constants.DT) ** 2)
    else:
        err2 = c ** 2 * (np.sin(c * (2 * n + 1) * Constants.DT / 2) ** 2) / 2
        err1 = c ** 2 * (np.cos(c * n * Constants.DT) ** 2) / 2

    return e, hx[1:-1, :-1], hy[:-1, 1:-1], err1, err2


def pad_function(input):
    return tf.constant([[0, 0], [input[0], input[1]], [input[2], input[3]], [0, 0]], shape=[4, 2])


def loss_yee(name,beta, delta, E1, Hx1, Hy1, e_true, hx_true, hy_true, i):
    l = 0
    for n in range(Constants.TIME_STEPS - 1):
        E1 = amper(E1, Hx1, Hy1, beta, delta)
        if name=='DRP':
           Hx1, Hy1 = faraday(E1, Hx1, Hy1, 0., 0.)
        else:
            Hx1, Hy1 = faraday(E1, Hx1, Hy1, beta, delta)
        l += tf.reduce_max(abs(E1[0, :, :, 0] - e_true[i * i * Constants.TIME_STEPS + (n + 1), :, :, 0])) + \
             tf.reduce_max(abs(Hx1[0, :, :, 0] - hx_true[i * i * Constants.TIME_STEPS + (n + 1), :, :, 0])) + \
             tf.reduce_max(abs(Hy1[0, :, :, 0] - hy_true[i * Constants.TIME_STEPS + (n + 1), :, :, 0]))
    return l / (3 * (Constants.TIME_STEPS - 1))


def loss_model(model, E1, Hx1, Hy1, e_true, hx_true, hy_true, i):
    l = 0
    for n in range(Constants.TIME_STEPS - 1):
        # E1 = amper(E1, Hx1, Hy1, w)
        # Hx1, Hy1 = faraday(E1, Hx1, Hy1, w)
        E1, Hx1, Hy1, inte, inth = model([E1, Hx1, Hy1])
        E1 = E1[:, 0:Constants.N, :, :]
        Hx1 = Hx1[:, 0:Constants.N - 2, :, :]
        Hy1 = Hy1[:, 0:Constants.N - 1, :, :]
        l += tf.reduce_max(abs(E1[0, :, :, 0] - e_true[i * i * Constants.TIME_STEPS + (n + 1), :, :, 0])) + \
             tf.reduce_max(abs(Hx1[0, :, :, 0] - hx_true[i * i * Constants.TIME_STEPS + (n + 1), :, :, 0])) + \
             tf.reduce_max(abs(Hy1[0, :, :, 0] - hy_true[i * Constants.TIME_STEPS + (n + 1), :, :, 0]))
        return l / (3 * (Constants.TIME_STEPS - 1))


def custom_loss(y_true, y_pred):
    loss = tf.reduce_mean(abs(y_true - y_pred))
    return loss / Constants.DT


class MAIN_LAYER(keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.pars1 = tf.Variable(2., trainable=True, dtype=tf.dtypes.float64, name='beta')
        self.pars2 = tf.Variable(2., trainable=True, dtype=tf.dtypes.float64, name='delta')

    def call(self, input):
        E, Hx, Hy = input
        E_n = amper(tf.cast(E, tf.dtypes.float64), tf.cast(Hx, tf.dtypes.float64), tf.cast(Hy, tf.dtypes.float64),
                    self.pars1, self.pars2)

        Hx_n, Hy_n = faraday(tf.cast(E_n, tf.dtypes.float64), tf.cast(Hx, tf.dtypes.float64),
                             tf.cast(Hy, tf.dtypes.float64), self.pars1, self.pars2)

        E_m = amper(tf.cast(E_n, tf.dtypes.float64), tf.cast(Hx_n, tf.dtypes.float64), tf.cast(Hy_n, tf.dtypes.float64),
                    self.pars1, self.pars2)

        Hx_m, Hy_m = faraday(E_m, Hx_n, Hy_n, self.pars1, self.pars2)

        inte = trapz2_batch(E_n ** 2, Constants.X,Constants.X)

        inthx = trapz2_batch(Hx_n ** 2, Constants.X[-1:1],Constants.X[:-1])
        inthy = trapz2_batch(Hy_n ** 2, Constants.X[:-1], Constants.X[1:-1])

        # int2 = simps(simps((Hx_n[0,:,:,0]) ** 2, Constants.X1), Constants.X2)
        # int3 = simps(simps((Hy_n[0,:,:,0]) ** 2, Constants.X1), Constants.X2)

        return tf.concat([E_n, E_m], 1), tf.concat([Hx_n, Hx_m], 1), tf.concat([Hy_n, Hy_m], 1), inte, inthx+inthy
