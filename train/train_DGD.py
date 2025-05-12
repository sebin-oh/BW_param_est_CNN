#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Last Updated:   04/24/2025
@Author      :   Sebin Oh
@Contact     :   sebin.oh@berkeley.edu
@Description : 
This script trains a Convolutional Neural Network (CNN) to estimate DGD (Degradation) parameters from hysteresis data derived from cyclic loading histories of structures. It loads preprocessed datasets of displacement and force time series, introduces varying levels of noise to the force component to improve generalizability, and normalizes the input and output for training.

The CNN architecture is designed to process 2D representations of hysteresis curves, applying convolutional, pooling, and dense layers. The training targets are normalized BSC parameters, and the output layer uses a sigmoid activation function to ensure outputs remain in [0, 1].

The script:
- Loads training and test datasets for a specified loading history
- Augments training data by injecting Gaussian noise at multiple levels
- Normalizes displacement and force inputs using their respective maxima
- Normalizes the target outputs (parameter values) for compatibility with sigmoid activation
- Defines a CNN architecture using Keras with optional data augmentation layers
- Compiles and trains the model using the mean squared error loss
- Visualizes training loss and accuracy over epochs

@Reference   : Oh, S., Song, J., & Kim, T. (2024). Deep learning-based modularized loading protocol for parameter estimation of Bouc-Wen class models. Engineering Structures
"""
# %%
# Import the necessary libraries
import random
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.layers import MaxPooling2D
from keras.layers import RandomContrast

# Plotting settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "cm"

# %%
# Import the data

LH = "Ref" # Ref: Reference loading history, Opt: Optimal loading history

Input_train = pickle.load(open(os.path.dirname(os.path.abspath('__file__')) + "./Data/input_train_"+str(LH)+".pkl", "rb"))
Input_test = pickle.load(open(os.path.dirname(os.path.abspath('__file__')) + "./Data/input_test_"+str(LH)+".pkl", "rb"))
Output_train = pickle.load(open(os.path.dirname(os.path.abspath('__file__')) + "./Data/output_train_"+str(LH)+".pkl", "rb"))
Output_test = pickle.load(open(os.path.dirname(os.path.abspath('__file__')) + "./Data/output_test_"+str(LH)+".pkl", "rb"))

# Memory optimization
Input_train = np.float16(Input_train)
Input_test = np.float16(Input_test)
Output_train = np.float16(Output_train)
Output_test = np.float16(Output_test)

idx_DGD = [5,6]

Input_train_disp_max = np.max(Input_train[:,:,0,:], axis=1)
Input_train_force_max = np.max(Input_train[:,:,1,:], axis=1)
Input_test_disp_max = np.max(Input_test[:,:,0,:], axis=1)
Input_test_force_max = np.max(Input_test[:,:,1,:], axis=1)

Input_train_disp_max = np.max(Input_train[:,:,0,:], axis=1)
Input_train_force_max = np.max(Input_train[:,:,1,:], axis=1)
Input_test_disp_max = np.max(Input_test[:,:,0,:], axis=1)
Input_test_force_max = np.max(Input_test[:,:,1,:], axis=1)

# %%
# Introduce noise to the training input (force)
cov_low = 0.002
cov_med = 0.005
cov_high = 0.008

sigma_low = cov_low*np.ones(Input_train[:,:,0,:].shape) # because the input has been max-normalized
sigma_med = cov_med*np.ones(Input_train[:,:,0,:].shape)
sigma_high = cov_high*np.ones(Input_train[:,:,0,:].shape)

noise_low = np.random.normal(0, sigma_low, Input_train[:,:,1,:].shape)
noise_med = np.random.normal(0, sigma_med, Input_train[:,:,1,:].shape)
noise_high = np.random.normal(0, sigma_high, Input_train[:,:,1,:].shape)

Input_train_noised_zero = np.copy(Input_train)
Input_train_noised_low = np.copy(Input_train)
Input_train_noised_med = np.copy(Input_train)
Input_train_noised_high = np.copy(Input_train)

Input_train_noised_zero[:,0,1,:] = 0
Input_train_noised_low[:,0,1,:] = 0
Input_train_noised_med[:,0,1,:] = 0
Input_train_noised_high[:,0,1,:] = 0

Input_train_noised_low[:,1:,1,:] = Input_train_noised_low[:,1:,1,:] + noise_low[:,1:,:]
Input_train_noised_med[:,1:,1,:] = Input_train_noised_med[:,1:,1,:] + noise_med[:,1:,:]
Input_train_noised_high[:,1:,1,:] = Input_train_noised_high[:,1:,1,:] + noise_high[:,1:,:]

num_train = 1e6
num_train_zero = 3e5
num_train_low = 3e5
num_train_med = 3e5
num_train_high = 1e5

idx_zero = random.sample(range(0, Input_train.shape[0]), int(num_train_zero))
idx_low = random.sample(range(0, Input_train.shape[0]), int(num_train_low))
idx_med = random.sample(range(0, Input_train.shape[0]), int(num_train_med))
idx_high = random.sample(range(0, Input_train.shape[0]), int(num_train_high))

Input_train_noised_zero = Input_train_noised_zero[idx_zero, :, :, :]
Input_train_noised_low = Input_train_noised_low[idx_low, :, :, :]   
Input_train_noised_med = Input_train_noised_med[idx_med, :, :, :]
Input_train_noised_high = Input_train_noised_high[idx_high, :, :, :]

Input_train = np.concatenate((Input_train_noised_zero, Input_train_noised_low, Input_train_noised_med, Input_train_noised_high), axis=0)

# Adjust the training output
Output_train = np.concatenate((Output_train[idx_zero, :],Output_train[idx_low, :],Output_train[idx_med, :],Output_train[idx_high, :],))

# Adjust the max/min values
Input_train_disp_max = np.concatenate((Input_train_disp_max[idx_zero], Input_train_disp_max[idx_low], Input_train_disp_max[idx_med], Input_train_disp_max[idx_high]))
Input_train_force_max = np.concatenate((Input_train_force_max[idx_zero], Input_train_force_max[idx_low], Input_train_force_max[idx_med], Input_train_force_max[idx_high]))

# %%
# Normalize the input and output of the training and test datasets
Input_train[:,:,0,:] = Input_train[:,:,0,:]/Input_train_disp_max[:,None]
Input_train[:,:,1,:] = Input_train[:,:,1,:]/Input_train_force_max[:,None]
Input_test[:,:,0,:] = Input_test[:,:,0,:]/Input_test_disp_max[:,None]
Input_test[:,:,1,:] = Input_test[:,:,1,:]/Input_test_force_max[:,None]

# %%
# Normalize the training output so that the values are between 0 and 1 (for the sigmoid activation function)

Output_train_normed = np.copy(Output_train)
params_min = []
params_max = []
for i in range(Output_train_normed.shape[1]):
    params_min.append(np.min(Output_train_normed[:,i]))
    params_max.append(np.max(Output_train_normed[:,i]))
params_min = np.asarray(params_min)
params_max = np.asarray(params_max)
params_range = params_max - params_min
Output_train_normed = (Output_train_normed - params_min)/params_range

# %%
# Define the CNN architecture

input_DGD = Input(shape=(Input_train.shape[1], Input_train.shape[2], 1))

input_DGD = RandomContrast(factor=0.5)(input_DGD)

Layer_conv = Convolution2D(4, (2, 2), padding='same',activation='relu')(input_DGD)
Layer_conv = MaxPooling2D(pool_size=(2, 1),padding='same')(Layer_conv)
Layer_conv = Convolution2D(8, (4, 2), padding='same',activation='relu')(Layer_conv)
Layer_conv = MaxPooling2D(pool_size=(2, 1),padding='same')(Layer_conv)

Layer_flat = Flatten()(Layer_conv)
Layer_flat = Dense(256, activation='relu')(Layer_flat)
Layer_flat = Dense(32, activation='relu')(Layer_flat)

output_DGD = Dense(2, activation='sigmoid')(Layer_flat)

model_DGD = Model(inputs=input_DGD, outputs=output_DGD)

# %%
# Compile the model

model_DGD.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# %%
# Train the model

history = model_DGD.fit(Input_train,
                        Output_train_normed[:, idx_DGD],
                        epochs=300,
                        batch_size=64)

plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
