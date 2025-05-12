#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Last Updated:   04/24/2025
@Author      :   Sebin Oh
@Contact     :   sebin.oh@berkeley.edu
@Description :   

This scripts tests the trained CNN model

"""

# %%
# Import the necessary libraries
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from tensorflow import keras

# Plotting settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "cm"

# %%
# Select the loading history
LH = "Opt" # Ref: Reference loading history, Opt: Optimal loading history

# Load the data
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "Data")

Input_train = pickle.load(open(os.path.join(data_dir, f"input_train_{LH}.pkl"), "rb"))
Input_test = pickle.load(open(os.path.join(data_dir, f"input_test_{LH}.pkl"), "rb"))
Output_train = pickle.load(open(os.path.join(data_dir, f"output_train_{LH}.pkl"), "rb"))
Output_test = pickle.load(open(os.path.join(data_dir, f"output_test_{LH}.pkl"), "rb"))

# Optimal loading history for the DGD model includes only 280 time steps
if LH == "Opt":
    Input_train = Input_train[:,:280,:,:]
    Input_test = Input_test[:,:280,:,:] 

# Memory optimization
Input_train = np.float16(Input_train)
Input_test = np.float16(Input_test)
Output_train = np.float16(Output_train)
Output_test = np.float16(Output_test)

idx_DGD = [5,6] 

# %%
# Normalize the input and output of the training and test datasets 

Input_train_disp_max = np.max(Input_train[:,:,0,:], axis=1)
Input_train_force_max = np.max(Input_train[:,:,1,:], axis=1)
Input_test_disp_max = np.max(Input_test[:,:,0,:], axis=1)
Input_test_force_max = np.max(Input_test[:,:,1,:], axis=1)

Output_train[:,0] = np.squeeze(np.sqrt(Input_train_force_max/Input_train_disp_max))*Output_train[:,0]
Output_train[:,1] = Output_train[:,1]/np.squeeze(Input_train_force_max)

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
# Models loading
data_dir = os.path.join(base_dir, "..", "trained models")
CNN_model = keras.models.load_model(os.path.join(data_dir, f"CNN_DGD_{LH}.pkl"),compile=False)

# %%
# Predict the parameters using the trained model
param_DGD_test = CNN_model.predict(Input_test, batch_size=64)
param_DGD_test = np.multiply(params_range[idx_DGD],param_DGD_test)+params_min[idx_DGD]

param_DGD_test[:,0] = param_DGD_test[:,0]/np.squeeze(np.sqrt(Input_test_force_max/Input_test_disp_max))
param_DGD_test[:,1] = param_DGD_test[:,1]*np.squeeze(Input_test_force_max)

# %%
# Calculate the correlation coefficients between the predicted and true parameters
corrcoef_values = np.zeros((len(idx_DGD)))
for i in range(len(idx_DGD)):
    corrcoef_values[i] = np.corrcoef(Output_test[:,idx_DGD[i]], param_DGD_test[:,i])[0,1]

# %%
# Plot the results

plt.figure(figsize=[15,5])
ax1 = plt.subplot2grid((1,2),(0,0))
ax2 = plt.subplot2grid((1,2),(0,1))

ax1.plot(Output_test[:,idx_DGD[0]],param_DGD_test[:,0],'k.',markersize=1)
ax1.annotate("$\\rho$ = {:.5f}".format(corrcoef_values[0]), (0.035, 0.315), fontsize=15)
ax1.plot([0,0.36],[0,0.36],'--',color='red')
ax1.set_xlabel('True', fontsize = 18)
ax1.set_ylabel('Prediction', fontsize = 18)
ax1.tick_params(axis='both', labelsize=14)
ax1.set_aspect('equal')
ax1.set_xlim([0, 0.36])
ax1.set_ylim([0, 0.36])
ax1.set_title('$\mathdefault{\\delta_\\nu}$', fontsize = 18)
plt.tight_layout()

ax2.plot(Output_test[:,idx_DGD[1]],param_DGD_test[:,1],'k.',markersize=1)
ax2.annotate("$\\rho$ = {:.5f}".format(corrcoef_values[1]), (0.04, 0.35), fontsize=15)
ax2.plot([0,0.39],[0,0.39],'--',color='red')
ax2.set_xlabel('True', fontsize = 18)
ax2.set_ylabel('Prediction', fontsize = 18)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_aspect('equal')
ax2.set_xlim([0, 0.39])
ax2.set_ylim([0, 0.39])
ax2.set_title('$\mathdefault{\\delta_\\eta}$', fontsize = 18)
plt.tight_layout()

### Bar plot for the correlation coefficients
plt.figure(figsize=[3,5])
plt.bar([r'$\delta_{\nu}$',r'$\delta_{\eta}$'],corrcoef_values, width=0.6, color='lightsteelblue', edgecolor='black')
plt.ylabel("Correlation coefficient", fontsize=18)
plt.ylim([0.9,1])
plt.tight_layout()
plt.show()
