"""
Author: Patrick Brosseau
Date: 02-11-2022

Detection of outlier two-dimensional electronic spectra (2DES). 
Compares repetitions of 2DES experiment using Principal Components Analysis (PCA). 
Calculates Mahalanobis distance between experimental repetitions in PCA basis.
Identifies outliers based off a set threshhold value for Mahalanobis distance.

Input: npz file
stack_input = four dimensional array, the axes in order are: [e_x, t2, e_em, repetitions]
e_x: 1D array, excitation energy, for plotting
e_em: 1D array, emission energy, for plotting
t2: 1D array, population time delays, for plotting
"""

import sys
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)

mpl.rcParams['contour.negative_linestyle'] = 'solid'

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data,axis=0)
    #if not cov:
    #    cov = np.cov(data.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="2DES outlier detection")
parser.add_argument('-f', '--file')
parser.add_argument('-ps', '--plotspectra', action='store_true', default=False)
parser.add_argument('-po', '--plotoutliers', action='store_true', default=False)
args = parser.parse_args()

input_file = args.file
switch_plot_spectra = args.plotspectra
switch_plot_outliers = args.plotoutliers

if switch_plot_spectra or switch_plot_outliers:
    if not os.path.exists('rep_images'):
       os.makedirs('rep_images')

#Load data
inp = np.load(input_file)
stack_input = inp["stack_reps"] #stack_input = four dimensional array, the axes in order are: [e_x, t2, e_em, repetitions]
e_x = inp["e_x"]
e_em = inp["e_em"]
t2 = inp["t2"]

nreps = stack_input.shape[-1]
rep_list = np.arange(0,nreps)

thresh = 12 #Mahalanobis threshhold

#Plot 2DE spectra repetitions for each t2
if switch_plot_spectra:
    for t2_step in tqdm(t2):
        time_index = np.argmin(np.abs(t2-t2_step))
        
        fig = plt.figure(figsize=(5,8))
        grid = plt.GridSpec(8, 5, hspace=0.1, wspace=0.1)

        i = 0
        j = 0
        for rep in np.linspace(0,nreps-1,nreps).astype(int):
            
            ax = fig.add_subplot(grid[i,j])
            plt.sca(ax)
            
            spec = stack_input[:,:,:,rep]
            scale = np.max(np.abs(spec[:, time_index, :]))
            plot_spec = np.real(spec[:, time_index, :].T)/scale

            plt.pcolormesh(e_x,e_em,plot_spec,cmap="seismic_r",vmin=-1,vmax=1)
            plt.contour(e_x, e_em,plot_spec,colors='w', levels=np.linspace(-1, 1, 21),linewidths=0.5)
            plt.xlim(1.95,2.25)
            plt.ylim(1.95,2.25)
            plt.xticks([])
            plt.yticks([])
            ax.set_aspect("equal")
            
            plt.text(1.95,2.175,rep,c="k")
            
            j += 1
            if j == 5:
                j = 0
                i += 1
                
        plt.savefig(r"rep_images\t2 = "+str(int(t2[time_index]))+" fs.png")
        plt.close()
    
       
#Detect outliers
outliers = []

for time_index in tqdm(range(0,stack_input.shape[1])):
    stack = np.zeros((stack_input.shape[0]*stack_input.shape[2],stack_input.shape[3]))
    for i in range(0,stack_input.shape[3]):
        stack[:,i] = stack_input[:,time_index,:,i].flatten()
    stack = stack.T

    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(stack)

    cov = np.cov(X_pca.T)
    mahala = mahalanobis(x=X_pca, data=X_pca, cov=cov)
    super_threshold_indices = np.argwhere(mahala > thresh).tolist()

    nreps= len(X_pca)
    colors = np.linspace(0,nreps,nreps)
    
    for s in super_threshold_indices:
        outliers.append((time_index,s[0]))
        
    if switch_plot_outliers:
        plt.figure(figsize=(4,3))
        plt.title("$t_2$ = %0.0f fs"%(t2[time_index]))
        plt.plot(mahala,marker="o",linestyle="none")
        plt.plot(super_threshold_indices,mahala[super_threshold_indices],marker="o",c="r",linestyle="none")
        plt.hlines(thresh,-10,200,linestyle="--",color="r")
        plt.xlim(-1,stack_input.shape[-1])
        plt.ylim(-1,40)
        plt.xlabel("Repetition #")
        plt.ylabel("Mahalanobis Distance")
        plt.tight_layout()
        
        plt.savefig(r"rep_images\Outliers, t2 = "+str(int(t2[time_index]))+" fs.png")
        plt.close()

print("Outliers, (t2_index,repetition): \n", outliers)
print("Percentage of 2DES outliers: %0.2f"%(100*len(outliers)/(stack_input.shape[-1]*stack_input.shape[1])))
print("Total number of 2DES outliers: %0.0f"%(len(outliers)))
        