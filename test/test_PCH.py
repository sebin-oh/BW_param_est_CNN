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

# Memory optimization
Input_train = np.float16(Input_train)
Input_test = np.float16(Input_test)
Output_train = np.float16(Output_train)
Output_test = np.float16(Output_test)

idx_PCH = [7,8,9,10,11,12]

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
CNN_model = keras.models.load_model(os.path.join(data_dir, f"CNN_PCH_{LH}.pkl"),compile=False)

# %%
# Predict the parameters using the trained model
param_PCH_test = CNN_model.predict(Input_test, batch_size=64)
param_PCH_test = np.multiply(params_range[idx_PCH],param_PCH_test)+params_min[idx_PCH]

param_PCH_test[:,0] = param_PCH_test[:,0]/np.squeeze(np.sqrt(Input_test_force_max/Input_test_disp_max))
param_PCH_test[:,1] = param_PCH_test[:,1]*np.squeeze(Input_test_force_max)

# %%
# Calculate the correlation coefficients between the predicted and true parameters
corrcoef_values = np.zeros((len(idx_PCH)))
for i in range(len(idx_PCH)):
    corrcoef_values[i] = np.corrcoef(Output_test[:,idx_PCH[i]], param_PCH_test[:,i])[0,1]

# %%
# Plot the results

plt.figure(figsize=[15,10])
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,0), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,2), colspan=2)
ax6 = plt.subplot2grid((2,6), (1,4), colspan=2)

ax1.plot(Output_test[:,idx_PCH[0]],param_PCH_test[:,0],'k.',markersize=1)
ax1.annotate("$\\rho$ = {:.5f}".format(corrcoef_values[0]), (0.1, 0.9), fontsize=15)
ax1.plot([0,1],[0,1],'--',color='red')
ax1.set_xlabel('True', fontsize = 18)
ax1.set_ylabel('Prediction', fontsize = 18)
ax1.tick_params(axis='both', labelsize=14)
ax1.set_aspect('equal')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_title("$\mathdefault{\\zeta_0}$", fontsize = 18)
plt.tight_layout()

ax2.plot(Output_test[:,idx_PCH[1]],param_PCH_test[:,1],'k.',markersize=1)
ax2.annotate("$\\rho$ = {:.5f}".format(corrcoef_values[1]), (0.138, 1.242), fontsize=15)
ax2.plot([0,1.38],[0,1.38],'--',color='red')
ax2.set_xlabel('True', fontsize = 18)
ax2.set_ylabel('Prediction', fontsize = 18)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_aspect('equal')
ax2.set_xlim([0, 1.38])
ax2.set_ylim([0, 1.38])
ax2.set_title("$\mathdefault{p}$", fontsize = 18)
plt.tight_layout()

ax3.plot(Output_test[:,idx_PCH[2]],param_PCH_test[:,2],'k.',markersize=1)
ax3.annotate("$\\rho$ = {:.5f}".format(corrcoef_values[2]), (0.052, 0.388), fontsize=15)
ax3.plot([0.01,0.43],[0.01,0.43],'--',color='red')
ax3.set_xlabel('True', fontsize = 18)
ax3.set_ylabel('Prediction', fontsize = 18)
ax3.tick_params(axis='both', labelsize=14)
ax3.set_aspect('equal')
ax3.set_xlim([0.01, 0.43])
ax3.set_ylim([0.01, 0.43])
ax3.set_title("$\mathdefault{q}$", fontsize = 18)
plt.tight_layout()

ax4.plot(Output_test[:,idx_PCH[3]],param_PCH_test[:,3],'k.',markersize=1)
ax4.annotate("$\\rho$ = {:.5f}".format(corrcoef_values[3]), (0.175, 0.775), fontsize=15)
ax4.plot([0.1,0.85],[0.1,0.85],'--',color='red')
ax4.set_xlabel('True', fontsize = 18)
ax4.set_ylabel('Prediction', fontsize = 18)
ax4.tick_params(axis='both', labelsize=14)
ax4.set_aspect('equal')
ax4.set_xlim([0.1, 0.85])
ax4.set_ylim([0.1, 0.85])
ax4.set_title("$\mathdefault{\\psi}$", fontsize = 18)
plt.tight_layout()

ax5.plot(Output_test[:,idx_PCH[4]],param_PCH_test[:,4],'k.',markersize=1)
ax5.annotate("$\\rho$ = {:.5f}".format(corrcoef_values[4]), (0.009, 0.081), fontsize=15)
ax5.plot([0,0.09],[0,0.09],'--',color='red')
ax5.set_xlabel('True', fontsize = 18)
ax5.set_ylabel('Prediction', fontsize = 18)
ax5.tick_params(axis='both', labelsize=14)
ax5.set_aspect('equal')
ax5.set_xlim([0, 0.09])
ax5.set_ylim([0, 0.09])
ax5.set_title("$\mathdefault{\\delta_\\psi}$", fontsize = 18)
plt.tight_layout()

ax6.plot(Output_test[:,idx_PCH[5]],param_PCH_test[:,5],'k.',markersize=1)
ax6.annotate("$\\rho$ = {:.5f}".format(corrcoef_values[5]), (0.089, 0.721), fontsize=15)
ax6.plot([0.01,0.8],[0.01,0.8],'--',color='red')
ax6.set_xlabel('True', fontsize = 18)
ax6.set_ylabel('Prediction', fontsize = 18)
ax6.tick_params(axis='both', labelsize=14)
ax6.set_aspect('equal')
ax6.set_xlim([0.01, 0.8])
ax6.set_ylim([0.01, 0.8])
ax6.set_title("$\mathdefault{\\lambda}$", fontsize = 18)
plt.tight_layout()

### Bar plot for the correlation coefficients
plt.figure(figsize=[6,5])
plt.bar([r'$\zeta_0$',r'$p$',r'$q$',r'$\psi$',r'$\delta_{\psi}$',r'$\lambda$'],corrcoef_values, width=0.6, color='lightsteelblue', edgecolor='black')
plt.ylabel("Correlation coefficient", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim([0,1])
plt.tight_layout()
plt.show()
