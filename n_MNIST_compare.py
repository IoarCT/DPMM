#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:53:47 2022

@author: icasado
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import scipy.io
import matplotlib.pyplot as plt
from HPP_CVI_DPMM import HPP_DPMM
from MHPP_CVI_DPMM import MHPP_DPMM
from CVI_for_DPMM import CV_DPMM
from Auxiliary_functions_for_TFCVI import compute_test_perplexity
import struct

# Load datasets
# Original MNIST data
# Train data
with open('/home/icasado/Desktop/n_mnist/train-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    train_mnist = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    train_mnist = train_mnist.reshape((size, nrows*ncols))
# Test data
with open('/home/icasado/Desktop/n_mnist/t10k-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    test_mnist = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    test_mnist = test_mnist.reshape((size, nrows*ncols))
# Train labels
with open('/home/icasado/Desktop/n_mnist/train-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack('>II', f.read(8))
    train_mnist_labels = np.fromfile(f, dtype=np.dtype(np.uint8)).newbyteorder(">")   
# Test labels    
with open('/home/icasado/Desktop/n_mnist/t10k-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack('>II', f.read(8))
    test_mnist_labels = np.fromfile(f, dtype=np.dtype(np.uint8)).newbyteorder(">")   

# n_MNIST variants
mnist_awgn = scipy.io.loadmat('/home/icasado/Desktop/n_mnist/mnist-with-awgn.mat')
mnist_blur = scipy.io.loadmat('/home/icasado/Desktop/n_mnist/mnist-with-motion-blur.mat')
mnist_contrast_awgn = scipy.io.loadmat('/home/icasado/Desktop/n_mnist/mnist-with-reduced-contrast-and-awgn.mat')

# Gaussian noise train/test
train_awgn = mnist_awgn['train_x']
train_awgn_labels = np.zeros(train_awgn.shape[0], dtype=int)
for i in range(train_awgn.shape[0]):
    train_awgn_labels[i] = list(mnist_awgn['train_y'][i]).index(1)

test_awgn = mnist_awgn['test_x']
test_awgn_labels = np.zeros(test_awgn.shape[0], dtype=int)
for i in range(test_awgn.shape[0]):
    test_awgn_labels[i] = list(mnist_awgn['test_y'][i]).index(1)
 
# Blur train/test
train_blur = mnist_blur['train_x']
train_blur_labels = np.zeros(train_blur.shape[0], dtype=int)
for i in range(train_blur.shape[0]):
    train_blur_labels[i] = list(mnist_blur['train_y'][i]).index(1)

test_blur = mnist_blur['test_x']
test_blur_labels = np.zeros(test_blur.shape[0], dtype=int)
for i in range(test_blur.shape[0]):
    test_blur_labels[i] = list(mnist_blur['test_y'][i]).index(1)

# Contrast + gaussian noise train/test
train_con_awgn = mnist_contrast_awgn['train_x']
train_con_awgn_labels = np.zeros(train_con_awgn.shape[0], dtype=int)
for i in range(train_con_awgn.shape[0]):
    train_con_awgn_labels[i] = list(mnist_contrast_awgn['train_y'][i]).index(1)

test_con_awgn = mnist_contrast_awgn['test_x']
test_con_awgn_labels = np.zeros(test_con_awgn.shape[0], dtype=int)
for i in range(test_con_awgn.shape[0]):
    test_con_awgn_labels[i] = list(mnist_contrast_awgn['test_y'][i]).index(1)


n_batches = 15
n_samples = 10000
n_test_samples = 2000
n_comp = 50

# Preprocess
train_total = np.vstack((train_mnist, train_blur, train_awgn))
test_total = np.vstack((test_mnist, test_blur, test_awgn))
pca = PCA(n_components=n_comp)
minmax = MinMaxScaler()
train_total = minmax.fit_transform(train_total)
train_total = pca.fit(train_total)

train_mnist = pca.transform(minmax.fit_transform(train_mnist))
test_mnist = pca.transform(minmax.fit_transform(test_mnist))

train_blur = pca.transform(minmax.fit_transform(train_blur))
test_blur = pca.transform(minmax.fit_transform(test_blur))

train_awgn = pca.transform(minmax.fit_transform(train_awgn))
test_awgn = pca.transform(minmax.fit_transform(test_awgn))


batches = np.zeros((n_batches, n_samples, n_comp))
test_data = np.zeros((n_batches, n_test_samples, n_comp))


# Create data batches with drift
for i in range(5):
    data = train_mnist[[j for j, num in enumerate(train_mnist_labels.astype(int)) if num <= 5+i]]
    test_d = test_mnist[[j for j, num in enumerate(test_mnist_labels.astype(int)) if num <= 5+i]]

    batches[i] = data[np.random.choice(data.shape[0], n_samples, replace=False), :]
    test_data[i] = test_d[np.random.choice(test_d.shape[0], n_test_samples, replace=False), :]

for i in range(5, 10):
    data = train_blur
    test_d = test_blur

    batches[i] = data[np.random.choice(data.shape[0], n_samples, replace=False), :]
    test_data[i] = test_d[np.random.choice(test_d.shape[0], n_test_samples, replace=False), :]

for i in range(10, n_batches):
    data = train_awgn[[j for j, num in enumerate(train_awgn_labels.astype(int)) if num <= 19-i]]
    test_d = test_awgn[[j for j, num in enumerate(test_awgn_labels.astype(int)) if num <= 19-i]]

    batches[i] = data[np.random.choice(data.shape[0], n_samples, replace=False), :]
    test_data[i] = test_d[np.random.choice(test_d.shape[0], n_test_samples, replace=False), :]



params, clusters, N_t, iteration, log_lik, cluster_centers, cluster_covs = CV_DPMM(batches, 
                                                                            alpha=2.,
                                                                            thresh=1.e-4,
                                                                            max_iter=100,
                                                                            T=30)

params_1, clusters, N_t, iteration, log_lik, cluster_centers, cluster_covs = HPP_DPMM(batches, 
                                                                            alpha=2.,
                                                                            thresh=1.e-4,
                                                                            max_iter=100,
                                                                            T=30)
params_2, clusters, N_t, iteration, log_lik, cluster_centers, cluster_covs = MHPP_DPMM(batches, 
                                                                            alpha=2.,
                                                                            thresh=1.e-4,
                                                                            max_iter=100,
                                                                            T=30)


# for i in range(n_batches):
#     fig, ax = plt.subplots(6, 5, figsize=(20, 20))
#     cluster_centers[i] = pca.inverse_transform(cluster_centers[i])
#     n_clust = len(np.unique(clusters[i]))
#     for j in range(n_clust):
#         ax[j//6, j%5].matshow(cluster_centers[i][j].reshape((28, 28)), cmap='Greys')
#     plt.show()

perp = np.zeros((3, n_batches))
fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

for i in range(n_batches):
    perp[0][i] = compute_test_perplexity(test_data[i], params[i][4], params[i][0], params[i][1], params[i][2], params[i][3], 2)
    perp[1][i] = compute_test_perplexity(test_data[i], params_1[i][4], params_1[i][0], params_1[i][1], params_1[i][2], params_1[i][3], 2)
    perp[2][i] = compute_test_perplexity(test_data[i], params_2[i][4], params_2[i][0], params_2[i][1], params_2[i][2], params_2[i][3], 2)


ax1.plot(perp[0], 'bx--', linewidth=4, label='SVB-DPM')
ax1.plot(perp[1], 'rx-', linewidth=4, label='HPP-DPM')
ax1.plot(perp[2], 'gx-', linewidth=4, label='MHPP-DPM')
ax1.axvline(x = 5, color='black', linestyle='dashed')
ax1.axvline(x = 10, color='black', linestyle='dashed')
ax1.text(7, 57, 'Blur', fontsize=12)
ax1.text(11, 57, 'Gaussian noise', fontsize=12)
plt.xlabel('Batch nº', fontsize=18)
plt.ylabel('log(perplexity)', fontsize=18)
ax1.legend(frameon=False, fontsize=17, loc='lower left')
plt.show()
fig.savefig('n_MNIST.pdf', format='pdf', bbox_inches='tight')
