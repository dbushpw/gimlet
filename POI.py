#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system(' pip install tensorflow==2.0.0')
#get_ipython().system(' pip uninstall gin-config -y')
#! set path=%path%;"C:\Program Files (x86)\GnuWin32\bin"
#get_ipython().system(' rm -rf gimlet')
#! npm uninstall gimlet
#! git clone https://github.com/choderalab/gimlet.git


# In[14]:


import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(3)
# import tensorflow_probability as tfp
# tf.enable_eager_execution()

#sys.path.append('/content/gimlet')
sys.path.append('gimlet-master')

import gin
import lime
import pandas as pd
import numpy as np


#import read_tagged_sdf2
import read_tagged_sdf

import matplotlib.pyplot as plt
import seaborn as sns
#from IPython.display import SVG

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import pickle



df="mp_df_final_acidH_upd.sdf"
dfX="mp_df_strict_normal.sdf"
choice=1

if choice==1:
  ds_all=read_tagged_sdf2.read_tagged_sdf(df)
  n_global_te = int(79)
if choice==0:
  ds_all=read_tagged_sdf.read_tagged_sdf(dfX)
  n_global_te = int(82)

# In[4]:


for x in ds_all:
  print(x)
  break


# In[5]:


#! more /content/temp.sdf


# by default, there is coordinates in dataset created from sdf
# now we get rid of it
if choice==1:
  ds_all = ds_all.map(lambda mol, attr: (mol[0], mol[1], attr[0]))
if choice==0:
  ds_all = ds_all.map(lambda mol, attr: (mol[0], mol[1], attr[3]))#atoms, map, values# three arrays basicly
#atoms, adjacency_map, (atom_in_mol, bond_in_mol, q_i, attr_in_mol)


# In[8]:


ds_all = gin.probabilistic.gn.GraphNet.batch(ds_all, 128)


# In[9]:


# get the number of samples
# NOTE: there is no way to get the number of samples in a dataset
# except loop through one time, unfortunately
n_batches = gin.probabilistic.gn.GraphNet.get_number_batches(ds_all)
n_global_te = int(0.2 * n_batches.numpy())


# In[10]:


# now we split them into global test and the rest

#n_global_te = int(0.2 * n_batches.numpy())
#n_global_te = int(87)
#n_global_te = int(79)
ds_tr = ds_all.skip(n_global_te)
ds_te = ds_all.take(n_global_te)
ds_vl = ds_all.skip(n_global_te).take(n_global_te)
#print(ds_tr,n_batches,"global te",n_global_te):
#<SkipDataset shapes: (<unknown>, (None, None), (None, None), (None, None), <unknown>, (None,)), types: (tf.int64, tf.float32, tf.bool, tf.bool, tf.float32, tf.bool)> tf.Tensor(437, shape=(), dtype=int64) global te 87


# In[11]:


debug=False
debugI=False


if debugI:
    print("toto")
if debug:
    print("moto")
"""'phi_e_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_e_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_v_0': [32, 64, 128],
    'phi_v_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_v_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_u_0': [32, 64, 128],
    'phi_u_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_u_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'f_r_0': [32, 64, 128],
    'f_r_1': [32, 64, 128],
    'f_r_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'f_r_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],"""


config_space = {
    'D_V': [16, 32, 64, 128, 256],
    'D_E': [16, 32, 64, 128, 256],
    'D_U': [16, 32, 64, 128, 256],
    
    'phi_e_0': [32, 64, 128],
    'phi_e_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_e_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_v_0': [32, 64, 128],
    'phi_v_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_v_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_u_0': [32, 64, 128],
    'phi_u_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_u_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'f_r_0': [32, 64, 128],
    'f_r_1': [32, 64, 128],
    'f_r_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'f_r_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'learning_rate': [1e-5, 1e-4, 1e-3]

}

#debug=True
def init(point):
    global gn
    global optimizer

    class f_v(tf.keras.Model):
        def __init__(self, units=point['D_V']):
            super(f_v, self).__init__()
            self.d = tf.keras.layers.Dense(units)

        @tf.function
        def call(self, x):

            x = tf.one_hot(
                x,
                10)

            x.set_shape([None, 10])
            return self.d(x)

    f_e = tf.keras.layers.Dense(point['D_E'])

    f_u = lambda atoms, adjacency_map, batched_attr_in_mol: tf.tile(
            tf.zeros((1, point['D_U'])),
            [
                 tf.math.count_nonzero(
                     batched_attr_in_mol),
                1
            ]
        )

    phi_v = lime.nets.for_gn.ConcatenateThenFullyConnect(
        (point['phi_v_0'],
         point['phi_v_a_0'],
         point['D_V'],
         point['phi_v_a_1']))

    phi_e = lime.nets.for_gn.ConcatenateThenFullyConnect(
        (point['phi_e_0'],
         point['phi_e_a_0'],
         point['D_E'],
         point['phi_e_a_1']))

    phi_u = lime.nets.for_gn.ConcatenateThenFullyConnect(
        (point['phi_u_0'],
         point['phi_u_a_0'],
         point['D_U'],
         point['phi_u_a_1']))

    class f_r(tf.keras.Model):
        def __init__(self, config=[
          point['f_r_0'],
          point['f_r_a_0'],
          point['f_r_1'],
          point['f_r_a_1'], 1],

          d_e=point['D_E'],
          d_u=point['D_U'],
          d_v=point['D_V']):
            super(f_r, self).__init__()
            self.d = lime.nets.for_gn.ConcatenateThenFullyConnect(config)
            self.f_r_1 = config[2]
            self.d_e = d_e
            self.d_u = d_u
            self.d_v = d_v

        @tf.function
        def call(self, h_e, h_v, h_u,
                h_e_history, h_v_history, h_u_history,
                atom_in_mol, bond_in_mol):

            h_e_history.set_shape([None, 6, self.d_e])
            h_u_history.set_shape([None, 6, self.d_u])
            h_v_history.set_shape([None, 6, self.d_v])

            h_e_bar_history = tf.reduce_sum( # (n_mols, t, d_e)
                            tf.multiply(
                                tf.tile(
                                    tf.expand_dims(
                                        tf.expand_dims(
                                            tf.where( # (n_bonds, n_mols)
                                                tf.boolean_mask(
                                                    bond_in_mol,
                                                    tf.reduce_any(
                                                        bond_in_mol,
                                                        axis=1),
                                                    axis=0),
                                                tf.ones_like(
                                                    tf.boolean_mask(
                                                        bond_in_mol,
                                                        tf.reduce_any(
                                                            bond_in_mol,
                                                            axis=1),
                                                        axis=0),
                                                    dtype=tf.float32),
                                                tf.zeros_like(
                                                    tf.boolean_mask(
                                                        bond_in_mol,
                                                        tf.reduce_any(
                                                            bond_in_mol,
                                                            axis=1),
                                                        axis=0),
                                                    dtype=tf.float32)),
                                            2),
                                        3),
                                    [
                                        1,
                                        1,
                                        tf.shape(h_e_history)[1],
                                        tf.shape(h_e)[1]
                                    ]),
                                tf.tile( # (n_bonds, n_mols, t, d_e)
                                    tf.expand_dims(
                                        h_e_history, # (n_bonds, t, d_e)
                                        1),
                                    [1, tf.shape(bond_in_mol)[1], 1, 1])),
                            axis=0)

            h_e_bar_history = tf.math.divide_no_nan(
                h_e_bar_history, # (n_bonds, t, d_e)
                    tf.tile(
                    tf.expand_dims(
                        tf.expand_dims(
                            tf.cast(
                                tf.math.count_nonzero(
                                    bond_in_mol,
                                    axis=0),
                                tf.float32),
                            1),
                        2),
                    [1, tf.shape(h_e_bar_history)[1], tf.shape(h_e)[1]]))

            h_v_bar_history = tf.reduce_sum( # (n_mols, t, d_e)
                    tf.multiply(
                        tf.tile(
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.where( # (n_atoms, n_mols)
                                        atom_in_mol,
                                        tf.ones_like(
                                            atom_in_mol,
                                            dtype=tf.float32),
                                        tf.zeros_like(
                                            atom_in_mol,
                                            dtype=tf.float32)),
                                    2),
                                3),
                            [1, 1, tf.shape(h_v_history)[1], tf.shape(h_v)[1]]),
                        tf.tile( # (n_atoms, n_mols, t, d_e)
                            tf.expand_dims(
                                h_v_history, # (n_atoms, t, d_e)
                                1),
                            [1, tf.shape(atom_in_mol)[1], 1, 1])),
                    axis=0)

            h_v_bar_history = tf.math.divide_no_nan(
                h_v_bar_history, # (n_bonds, t, d_e)
                    tf.tile(
                    tf.expand_dims(
                        tf.expand_dims(
                            tf.cast(
                                tf.math.count_nonzero(
                                    atom_in_mol,
                                    axis=0),
                                tf.float32),
                            1),
                        2),
                    [1, tf.shape(h_v_bar_history)[1], tf.shape(h_v)[1]]))

            y = self.d(
                tf.reshape(
                    h_v_bar_history,
                    [-1, 6 * self.d_v]),
                tf.reshape(
                    h_e_bar_history,
                    [-1, 6 * self.d_e]),
                tf.reshape(
                    h_u_history,
                    [-1, 6 * self.d_u]))

            y = tf.reshape(y, [-1])

            return y


    gn = gin.probabilistic.gn.GraphNet(
        f_e=f_e,

        f_v=f_v(),

        f_u=f_u,

        phi_e=phi_e,

        phi_v=phi_v,

        phi_u=phi_u,

        f_r=f_r(),

        repeat=5)

    optimizer = tf.keras.optimizers.Adam(1e-5)#3

counter=1
n=50
para=0
if debug ==True:
  n=1
  print("Debug")
def obj_fn(point):
    global para
    #point = dict(zip(config_space.keys(), point))
    init(point)
    #print(point)
    
    for dummy_idx in range(n):#30
        #mol, attr: (mol[0], mol[1], attr[1]
        #for mol[0], mol[1], attr[1] in ds_tr:
        for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
                in ds_tr:

            with tf.GradientTape() as tape:
                y_hat = gn(
                    atoms,
                    adjacency_map,
                    atom_in_mol=atom_in_mol,
                    bond_in_mol=bond_in_mol,
                    batched_attr_in_mol=y_mask)

                
                y = tf.boolean_mask(
                    y,
                    y_mask)

                loss = tf.losses.mean_squared_error(y, y_hat)
                if debug:
                    break
            variables = gn.variables
            grad = tape.gradient(loss, variables)
            optimizer.apply_gradients(
                zip(grad, variables))
            if debugI:
                print(y_hat,"Hat, trI",y_mask,"maskI")
                print(y,"y,trI")
            if debug == True:
              print("ds_tr")
              break           
      


    y_true_tr = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_tr = -1. * tf.ones([1, ], dtype=tf.float32)

    y_true_vl = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_vl = -1. * tf.ones([1, ], dtype=tf.float32)

    y_true_te = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_te = -1. * tf.ones([1, ], dtype=tf.float32)

    for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
            in ds_tr:

        y_hat = gn(
            atoms,
            adjacency_map,
            atom_in_mol=atom_in_mol,
            bond_in_mol=bond_in_mol,
            batched_attr_in_mol=y_mask)

        y = tf.boolean_mask(
            y,
            y_mask)

        y_true_tr = tf.concat([y_true_tr, y], axis=0)
        y_pred_tr = tf.concat([y_pred_tr, y_hat], axis=0)
        if debugI:
                print(y_hat,"Hat, trII",y_mask,"masko")
                print(y,"y,trII")
        if debugI:
                print(y_pred_tr,"y_pred, trII")
                print(y_true_tr,"y_true,trII")
        if debug ==True:
          print("tr2")
          break
    for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
            in ds_vl:

        y_hat = gn(
            atoms,
            adjacency_map,
            atom_in_mol=atom_in_mol,
            bond_in_mol=bond_in_mol,
            batched_attr_in_mol=y_mask)

        y = tf.boolean_mask(
            y,
            y_mask)

        y_true_vl = tf.concat([y_true_vl, y], axis=0)
        y_pred_vl = tf.concat([y_pred_vl, y_hat], axis=0)
        if debugI:
                print(y_hat,"Hat, vl")
                print(y_mask,"vl mask")
                print(y,"y,vl")
        if debugI:
                print(y_pred_vl,"y_pred, vl")
                print(y_true_vl,"y_true,vl")
        if debug:
            break
    for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
            in ds_te:

        y_hat = gn(
            atoms,
            adjacency_map,
            atom_in_mol=atom_in_mol,
            bond_in_mol=bond_in_mol,
            batched_attr_in_mol=y_mask)

        y = tf.boolean_mask(
            y,
            y_mask)
        if debugI:
                print(y_hat,"Hat, te")
                print(y,"y,te",y_mask,"mask")
        y_true_te = tf.concat([y_true_te, y], axis=0)
        y_pred_te = tf.concat([y_pred_te, y_hat], axis=0)
        if debugI:
                print(y_pred_te,"y_pred, te")
                print(y_true_te,"y_true,te")
        if debug ==True:
          print("te")
          break
    w_save="weights"
    gn.save_weights("all_w/"+(w_save+str(counter))+(str(para)))
    r2_tr = r2_score(y_true_tr[1:].numpy(), y_pred_tr[1:].numpy())
    rmse_tr = mean_absolute_error(y_true_tr[1:].numpy(), y_pred_tr[1:].numpy())

    r2_vl = metrics.r2_score(y_true_vl[1:].numpy(), y_pred_vl[1:].numpy())
    rmse_vl = metrics.mean_squared_error(y_true_vl[1:].numpy(), y_pred_vl[1:].numpy())

    r2_te = r2_score(y_true_te[1:].numpy(), y_pred_te[1:].numpy())
    rmse_te = mean_absolute_error(y_true_te[1:].numpy(), y_pred_te[1:].numpy())


    print(point, flush=True)
    print(r2_tr, flush=True)
    print(rmse_tr, flush=True)
    print(r2_vl, flush=True)
    print(rmse_vl, flush=True)
    print(r2_te, flush=True)
    print(rmse_te, flush=True)

    return rmse_vl
N_X=1000
if debug:
    N_X=2
#lime.optimize.dummy.optimize(obj_fn, config_space.values(), N_X)
obj_fn({'D_V': 32, 'D_E': 256, 'D_U': 128, 'phi_e_0': 64, 'phi_e_a_0': 'relu', 'phi_e_a_1': 'tanh', 'phi_v_0': 128, 'phi_v_a_0': 'tanh', 'phi_v_a_1': 'elu', 'phi_u_0': 32, 'phi_u_a_0': 'leaky_relu', 'phi_u_a_1': 'tanh', 'f_r_0': 128, 'f_r_1': 32, 'f_r_a_0': 'tanh', 'f_r_a_1': 'elu', 'learning_rate': 0.001})
