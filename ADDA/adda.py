#!/usr/bin/env python
# coding: utf-8

# In[41]:


import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Reshape,Flatten,Dropout,BatchNormalization,Activation,Conv2D,ZeroPadding2D,MaxPooling2D,LeakyReLU
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import numpy as np
import os,sys,argparse

from dataset import get_dataset


# In[42]:


source_data, target_data, combine_data = get_dataset()
(trainX, trainY,trainX_domain),(testX,testY,testX_domain) = source_data
(trainDX, trainDY,trainDX_domain),(testDX,testDY,testDX_domain) = target_data
(combined_train_imgs,combined_train_labels,combined_train_domain),(combined_test_imgs,combined_test_labels,combined_test_domain) = combine_data


# In[56]:


class ADDA():
    def __init__(self,lr):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 3
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        
        self.src_flag = False
        self.disc_flag = False
        
        self.discriminator_decay_rate = 3 # iterations
        self.discriminator_decay_factor = 0.5
        self.src_optimizer = Adam(lr,beta_1 = 0.5)
        self.tgt_optimizer = Adam(lr,beta_1 = 0.5)
    
    def define_source_encoder(self,weights=None):
        inp = Input(shape=self.img_shape)
        x = Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = self.img_shape, padding = 'same')(inp)
        x = Conv2D(64, kernel_size = (3,3), activation = 'relu',padding='same')(x)
        x = MaxPooling2D((2,2))(x)
        x = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same')(x)
        x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
        x = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same')(x)
        x = MaxPooling2D((2,2))(x)
        x = Conv2D(128, kernel_size=(3,3), activation='relu', input_shape = self.img_shape,padding='same')(inp)
        x = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same')(x)
        x = Conv2D(32,kernel_size=(3,3),activation='relu',padding='same')(x)
        out = MaxPooling2D((2,2))(x)
        self.source_encoder = Model(inputs=inp, outputs=out)
        
        self.src_flag = True
        if weights is not None:
            self.source_encoder.load_weight(weights, by_name =True)
        
    def define_target_encoder(self, weights=None):
        if not self.src_flag:
            self.define_source_encoder()
        
        with tf.device('/cpu:0'):
            self.target_encoder = clone_model(self.source_encoder)
        
        if weights is not None:
            self.target_encoder.load_weights(weights, by_name=True)
    
    def get_source_classifier(self,model,weights=None):
        x = Flatten()(model.output)
        x = Dense(128, activation = 'relu')(x)
        x = Dense(10, activation = 'softmax')(x)
        
        source_classifier_model = Model(inputs = model.input,outputs=x)
        
        return source_classifier_model
    
    def define_discriminator(self, shape):
        inp = Input(shape = shape)
        x = Flatten()(inp)
        x = Dense(128, activation = LeakyReLU(alpha=0.3),kernel_regularizer = regularizers.l2(0.01), name = 'discriminator1')(x)
        x = Dense(2, activation = 'sigmoid', name = 'discriminator2')
        self.disc_flag = True
        self.discriminator_model = Model(inputs=inp,outputs=x,name='discriminator')
    
    def get_discriminator(self,model,weights=None):
        if not self.disc_flag: 
            self.define_discriminator(model.output_shape[1:])
        
        disc = Model(inputs=model.input, outputs=self.discriminator_model(model.output))
        
        if weights is not None:
            disc.load_weights(weights, by_name=True)
        
        return disc
    
    def train_source_model(self,model,epochs=2000,batch_size=128,save_interval=1, start_epoch=0):
        # Load Dataset       
        model.compile(loss='categorical_crossentropy', optimizer=self.src_optimizer, metrics=['acc'])
        
        if not os.path.isdir('model_checkpoint'):
            os.mkdir('model_checkpoint')
        
        model_name = 'model' 
        saver = tf.keras.callbacks.ModelCheckpoint(model_name+'.h5'
                                                   ,monitor='val_loss'
                                                   ,verbose=1 ,save_best_only=True
                                                   ,save_weights_only=True
                                                   ,mode='auto'
                                                   ,period=save_interval)
        #cal Minima에 빠져버린 경우, 쉽게 빠져나오지 못하고 갇혀버리게 되는데, 이때 learning rate를 늘리거나 줄여주는 방법으로 빠져나오는 효과를 기대할 수 있음.
        scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.75,patience=10,verbose=1,mode='min')
        
        model.fit(x=trainX,y=trainY,
                 batch_size=batch_size,
                 epochs=epochs,
                 callbacks=[saver,scheduler],
                 validation_data =(testX,testY),
                 initial_epoch = start_epoch)
        
    def train_target_discriminator(self,source_model=None,src_discriminator=None,tgt_discriminaotr=None,epochs=2000,batch_size=100,save_interval=1,start_epoch=0, num_batches=100):
        
        self.define_source_encoder(source_model)
        
        for layer in self.source_encoder.layers:
            layer.trainable = False
        
        source_discriminator = self.get_discriminator(self.source_encoder,src_discriminator)
        target_discriminator = self.get_discriminator(self.target_encoder,tgt_discriminaotr)
        
        if src_discriminator is not None:
            source_discriminator.load_weights(src_discriminator)
        if tgt_discriminator is not None:
            target_discriminator.load_weights(tgt_discriminaotr)
        
        source_discriminator.compile(loss='binary_crossentropy', optimizer=self.tgt_optimizer,metrics=['acc'])
        target_discriminator.compile(loss='binary_crossentropy', optimizer=self.tgt_optimizer,metrics=['acc'])
        
    def eval_source_classifier(self, model, batch_size=128,domain='source'):
        
        model.compile(loss='categorical_crossentropy',optimizer=self.src_optimizer, metrics=['acc'])
        scores = model.evaluation(testX,testY,batch_size=batch_size)
        print(' %s classifier test loss : %.5f'%(domain, scores[0]))
        print(' %s classifier test acc : %.5f'%(domain, scores[1]))
    
    def eval_target_classifier(self, source_model, target_discriminator):
        self.define_target_encoder()
        model = self.get_source_classifier(self.target_encoder, source_model)
        model.load_weights(target_discriminator, by_name=True)
        model.summary()
        self.eval_source_classifier(model, dataset=(testDX,testDY),domain='Target')       
        


# In[67]:


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--source_weights', required=False, help="Path to weights file to load source model for training classification/adaptation")
    ap.add_argument('-e', '--start_epoch', type=int,default=1, required=False, help="Epoch to begin training source model from")
    ap.add_argument('-n', '--discriminator_epochs', type=int, default=10000, help="Max number of steps to train discriminator")
    ap.add_argument('-l', '--lr', type=float, default=0.0001, help="Initial Learning Rate")
    ap.add_argument('-f', '--train_discriminator', action='store_true', help="Train discriminator model (if TRUE) vs Train source classifier")
    ap.add_argument('-a', '--source_discriminator_weights', help="Path to weights file to load source discriminator")
    ap.add_argument('-b', '--target_discriminator_weights', help="Path to weights file to load target discriminator")
    ap.add_argument('-t', '--eval_source_classifier', default=None, help="Path to source classifier model to test/evaluate")
    ap.add_argument('-d', '--eval_target_classifier', default=None, help="Path to target discriminator model to test/evaluate")
    args = ap.parse_args()
    
    adda = ADDA(args.lr)
    adda.define_source_encoder()
    
    if not args.train_discriminator:
        if args.eval_source_classifier is None:
            model = adda.get_source_classifier(adda.source_encoder, args.source_weights)
            adda.train_source_model(model, start_epoch=args.start_epoch-1) 
        else:
            model = adda.get_source_classifier(adda.source_encoder, args.eval_source_classifier)
            adda.eval_source_classifier(model)
            adda.eval_source_classifier(model)
    adda.define_target_encoder(args.source_weights)
    
    if args.train_discriminator:
        adda.train_target_discriminator(epochs=args.discriminator_epochs, 
                                        source_model=args.source_weights, 
                                        src_discriminator=args.source_discriminator_weights, 
                                        tgt_discriminator=args.target_discriminator_weights,
                                        start_epoch=args.start_epoch-1)
    if args.eval_target_classifier is not None:
        adda.eval_target_classifier(args.eval_source_classifier, args.eval_target_classifier)


# In[ ]:




