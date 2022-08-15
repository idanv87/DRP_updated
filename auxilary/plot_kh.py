import math

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import *
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches



from DRP_multiple_networks.constants import Constants
from DRP_multiple_networks.utils import custom_loss, custom_loss3
from DRP_multiple_networks.drp import calculate_DRP

path = Constants.PATH


# name = 'DL2i_N='+str(Constants.N)
name = 'DL3i_N='+str(Constants.N)
saving_path = path + 'Experiment_' + name + '_details/'
model1 = keras.models.load_model(saving_path + 'model.pkl',
                                     custom_objects={'custom_loss': custom_loss, 'custom_loss3': custom_loss3})
model1.load_weights(saving_path + 'model_weights_val_number_' + str(0) + '.pkl').expect_partial()




def loss(h,dt,k1, k2, a):
    k = np.sqrt(k1 ** 2 + k2 ** 2)

    f = ((1 - a) ** 2) * (np.cos(3 * k1 * h) + np.cos(3 * k2 * h)) + \
        (6 * a - 6 * a ** 2) * (np.cos(2 * k1 * h) + np.cos(2 * k2 * h)) + \
        (15 * a ** 2 - 6 * a) * (np.cos(k1 * h) + np.cos(k2 * h))
    omega = (2 / dt) * np.arcsin((dt/h) * np.sqrt((1 / 18) * (20 * a ** 2 - 4 * a + 2 - f)))

    return abs(omega/k-1)

a=[9/8, 1-3*model1.trainable_weights[0],1-3*calculate_DRP()]
k1, k2 = np.meshgrid(math.pi*Constants.K1_TRAIN, math.pi*Constants.K2_TRAIN, indexing='ij')

com=k1*0
for i ,kx in enumerate(Constants.K1_TRAIN):
    for j, ky in enumerate(Constants.K2_TRAIN):
        Z=list([loss(Constants.DX, Constants.DT, math.pi*kx, math.pi*ky, a[k]) for k in range(3)])
        com[i,j]=Z.index(min(Z))


cMap = ListedColormap(['red', 'green', 'blue'])
im=pcolormesh(k1 * Constants.DX/math.pi, k2 * Constants.DX/math.pi, com, cmap=cMap)

pop_a = mpatches.Patch(color='red', label='Yee4')
pop_b = mpatches.Patch(color='green', label='DL2')
pop_c = mpatches.Patch(color='blue', label='DRP')
plt.legend(handles=[pop_a, pop_b, pop_c])
plt.xlabel(r'${ \frac{hk_x}{\pi} }$', fontsize=12)
plt.ylabel(r'${ \frac{hk_y}{\pi} }$', rotation=0, fontsize=12, labelpad=14)
plt.suptitle("Dispersion relation minimizers", size=12, y=0.98)

#plt.xticks([Constants.K1_DRP[20]*Constants.DX,
#           Constants.K1_DRP[-1]*Constants.DX, Constants.K1_TRAIN[-1]*Constants.DX],
#[str(Constants.K1_DRP[20]*Constants.DX),r'${ k_{\max}}$',str("{:.1f}".format(Constants.K1_TRAIN[-1]*Constants.DX))])

#plt.yticks([Constants.K1_DRP[20]*Constants.DX,
#           Constants.K1_DRP[-1]*Constants.DX, Constants.K1_TRAIN[-1]*Constants.DX],
#[str(Constants.K1_DRP[20]*Constants.DX),r'${ k_{\max}}$',str("{:.1f}".format(Constants.K1_TRAIN[-1]*Constants.DX))])

# Adding grid to plot
# plt.savefig('/Users/idanversano/documents/papers/drp/figures/phase_41.eps', format='eps',
#             bbox_inches='tight')


plt.show()
print(q)
############################################################################
names=['Yee4', 'DL2', 'DRP']
k1, k2 = np.meshgrid(math.pi*Constants.K1_TRAIN, math.pi*Constants.K2_TRAIN, indexing='ij')

fig = plt.figure(figsize=(9.75, 5))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,3),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
for i, ax in enumerate(grid):

    Z = loss(Constants.DX, Constants.DT, k1 , k2 , a[i])
    im=ax.pcolormesh(k1 * Constants.DX/math.pi, k2 * Constants.DX/math.pi, Z)
    ax.set_title(names[i])
    ax.set_xlabel(r'${ \frac{hk_x}{\pi} }$', fontsize=16)
    if i==0:
        ax.set_ylabel(r'${ \frac{hk_y}{\pi} }$', rotation=0, fontsize=16, labelpad=14)
    ax.set_aspect('equal', 'box')

fig.suptitle("Dispersion  error", size=16, y=0.9)
ax.cax.colorbar(im)
ax.cax.toggle_label(True)

plt.savefig('/Users/idanversano/documents/papers/drp/figures/phase.eps', format='eps',
            bbox_inches='tight')

plt.show()
##################################################################################
print(q)

l_model = []
l_drp=[]
l_4=[]
l_k=[]





for k1, k2   in zip(np.arange(10,30,2), np.arange(10,30,2)):

    name = 'DL2i_N=31'
    saving_path = path + 'Experiment_' + name + '_details/'
    model1 = keras.models.load_model(saving_path + 'model.pkl',
                                     custom_objects={'custom_loss': custom_loss, 'custom_loss3': custom_loss3})
    model1.load_weights(saving_path + 'model_weights_val_number_' + str(0) + '.pkl').expect_partial()
    a = 1-3*model1.trainable_weights[0]
    h = Constants.DX
    l_k.append(h*np.sqrt((Constants.PI*k1) ** 2 + (Constants.PI*k2) ** 2 ))
    l_model.append(loss(h,Constants.DT, Constants.PI*k1, Constants.PI*k2, a))
    l_drp.append(loss(h,Constants.DT, Constants.PI*k1, Constants.PI*k2, 1.35680506))
    l_4.append(loss(h, Constants.DT, Constants.PI*k1, Constants.PI*k2, 9/8))



plt.plot(l_k, l_4, 'r', label="Yee4")
plt.plot(l_k, l_drp, '-d', label="DRP")
plt.plot(l_k, l_model, 'g', label='DL2', linestyle='dashed')

plt.legend(loc="upper left")

plt.title('Dispersion error')

plt.xlabel(r'$\vec{k}h$')
plt.ylabel(r'$\mathrm{Error}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.savefig('/Users/idanversano/documents/papers/drp/figures/dispersion2.eps', format='eps',
            bbox_inches='tight')
plt.show()