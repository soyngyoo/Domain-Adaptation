#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pickle as pk
import tensorflow as tf


# In[20]:


def get_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')
    # Process MNIST
    mnist_train = (x_train > 0).reshape(60000, 28, 28, 1).astype(np.uint8) * 255
    mnist_train = mnist_train[:10000]
    trainX = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = (x_test > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
    mnist_test = mnist_test[:500]
    testX = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    src = '/home/lab1_ysy/Domain Adaptation/MNIST_data/'
    # Load MNIST-M dataset as out-of-domain and unlabeled data
    with open(src+'mnistm_data.pkl', 'rb') as f:
        mnistm = pk.load(f)
    trainDX = mnistm['train'][:10000]
    testDX = mnistm['test'][:500]

    trainY = y_train[:10000]
    testY = y_test[:500]
    # Rescale -1 to 1
    trainX = trainX.astype(np.float32)/255.
    trainDX = trainDX.astype(np.float32)/255.
    testX = testX.astype(np.float32)/255.
    testDX = testDX.astype(np.float32)/255.

    trainX_domain = np.tile([1., 0.], [len(trainX), 1])
    trainDX_domain = np.tile([0., 1.], [len(trainDX),1])

    testX_domain = np.tile([1., 0.], [len(testX), 1])
    testDX_domain = np.tile([0., 1.], [len(testDX),1])

    print(trainX.shape,trainDX.shape,testX.shape,testDX.shape)

    combined_train_imgs = np.concatenate([trainX, trainDX])
    combined_train_labels = np.concatenate([trainY, trainY])
    combined_train_domain = np.concatenate([np.tile([1., 0.], [len(trainX), 1]),
                                            np.tile([0., 1.], [len(trainDX), 1])])
    print(combined_train_imgs.shape,combined_train_labels.shape,combined_train_domain.shape)

    combined_test_imgs = np.concatenate([testX, testDX])
    combined_test_labels = np.concatenate([testY, testY])
    combined_test_domain = np.concatenate([np.tile([1., 0.], [len(testX), 1]),
                                            np.tile([0., 1.], [len(testDX), 1])])

    print(combined_test_imgs.shape,combined_test_labels.shape,combined_test_domain.shape)
    
    source_data = (trainX,trainY,trainX_domain),(testX,testY,testX_domain)
    target_data = (trainDX,trainY,trainDX_domain),(testDX,testY,testDX_domain)
    combine_data = (combined_train_imgs,combined_train_labels,combined_train_domain),(combined_test_imgs,combined_test_labels,combined_test_domain)
    return source_data,target_data,combine_data


# In[17]:


def dataset_vis(trainX,trainY,trainDX,trainDY):
    c=0
    plt.figure(figsize=(20,3))
    for i in range(20,30):
        plt.subplot(1,10,c+1)
        plt.imshow(trainX[i])
        plt.title(trainY[i].argmax(),fontsize=30)
        plt.tight_layout()
        plt.axis('off')
        c+=1
    plt.show()
    plt.figure(figsize=(20,3))
    c=0
    for i in range(20,30):
        plt.subplot(1,10,c+1)
        plt.imshow(trainDX[i])
        plt.title(trainDY[i].argmax(),fontsize=30)
        plt.tight_layout()
        plt.axis('off')
        c+=1

