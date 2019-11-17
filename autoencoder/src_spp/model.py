from __future__ import division, print_function

__author__ = """Vassilis Tsiaras (tsiaras@csd.uoc.gr)"""
#    Vassilis Tsiaras <tsiaras@csd.uoc.gr>
#    Computer Science Department, University of Crete.

import sys
import os
import logging
import math
import numpy as np
import tensorflow as tf
from lib.model_io import get_configuration, setup_logger
from lib.optimizers import get_learning_rate, get_optimizer
from lib.precision import _FLOATX
from lib.spp import spatial_pyramid_pool

def get_weight_variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d()):
    if shape is None:
        return tf.get_variable(name)
    else:  
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)

def get_bias_variable(name, shape=None, initializer=tf.constant_initializer(value=0.0, dtype=_FLOATX)): 
    if shape is None:
        return tf.get_variable(name) 
    else:     
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)



class Autoencoder(object):

    def __init__(self, domainString, policyType):
        self.cfg, self.learning_rate_params, self.optim_params = get_configuration(domainString, policyType)
        cfg = self.cfg
        self.l2 = cfg['l2']
        self.layers_shape = cfg['layers_shape']
        self.n_layers = len(self.layers_shape) - 1
        self.input_dim = self.layers_shape[0]
        self.encode_dim = self.layers_shape[-1]
        self.level_n_bins =cfg['level_n_bins']
        self.train_model = cfg['train_model']

        setup_logger('msg_logger', cfg['msg_logging_dir'], level=logging.INFO)

        self.sess = tf.Session()
        with tf.variable_scope("autoencoder", reuse=tf.AUTO_REUSE):
            self.create_variables()  
            self.define_operations()
            if cfg['restore_model']:
                try:
                    self.restore_variables()
                except:
                    pass
      
  

    def create_variables(self):
        #get_weight_variable('W_conv1', (1, 5, 1, 1))

        for i in range(self.n_layers):
            get_weight_variable('W'+str(i), (self.layers_shape[i], self.layers_shape[i+1]))
   

    def encoder(self, X):
        Y = X
        for i in range(self.n_layers):
            W = get_weight_variable('W' + str(i))
            Y = tf.nn.relu(tf.matmul(Y, W))

        return Y

    def decoder(self, Y):
        X = Y
        for i in range(self.n_layers-1, -1, -1):
            W = get_weight_variable('W' + str(i))
            X = tf.nn.relu(tf.matmul(X, tf.transpose(W))) 

        return X 
            

    def define_operations(self):
 
        self.X_data_placeholder = tf.placeholder(dtype=_FLOATX, shape=[None, self.input_dim]) 

        self.X_encoded = self.encoder(self.X_data_placeholder)
        X_reconstructed = self.decoder(self.X_encoded) # Network prediction

        # Loss of train data
        self.train_loss = tf.reduce_mean(tf.pow(self.X_data_placeholder - X_reconstructed, 2)) 
        #self.train_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=X_data, logits=X_reconstructed))   

        # Regularization loss 
        if self.l2 is not None:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not('_b' in v.name)])
            self.train_loss += self.l2*l2_loss

        # define learning rate decay method 
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = get_learning_rate(self.cfg['learning_rate_method'], global_step, self.learning_rate_params)

        # define the optimization algorithm
        opt_name = self.cfg['optimization_algorithm'].lower()
        optimizer = get_optimizer(opt_name, learning_rate, self.optim_params)

        trainable = tf.trainable_variables()
        self.update_ops = optimizer.minimize(self.train_loss, var_list=trainable, global_step=global_step)

        self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1) 

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.sess.run(init_op) 


    def train(self, X):
        X = self.project_input_into_fixed_length_vector(X)

        mean_loss, _ = self.sess.run([self.train_loss, self.update_ops], 
                feed_dict={self.X_data_placeholder: X}) 

        print('Autoencoder mean loss:', mean_loss)
        if math.isnan(mean_loss):
            print('Autoencoder returned cost Nan')



    def encode(self, X_data):
        X_data = self.project_input_into_fixed_length_vector(X_data)
       
        X_encoded_np = self.sess.run(self.X_encoded, feed_dict={self.X_data_placeholder: X_data})

        return X_encoded_np


    def define_decode_operations(self):
        self.X_encoded_placeholder = tf.placeholder(dtype=_FLOATX, shape=(None, self.encode_dim)) 

        self.X_decoded = self.decoder(self.X_encoded_placeholder)


    def decode(self, X_encoded):
        X_decoded = self.sess.run(self.X_decoded, feed_dict={self.X_encoded_placeholder: X_encoded})

        return X_decoded

    def project_input_into_fixed_length_vector(self, X):
        input_shape = X.shape
        Y = np.reshape(X, (input_shape[0], 1, input_shape[1], 1))
        # Add a tf.nn.conv2d layer here. 
        Y = spatial_pyramid_pool(Y, self.level_n_bins, mode='avg')

        #Y_np = self.sess.run(Y)

        return Y
 

    def save_variables(self):
        print('save autoencoder')
        shape_str = ''
        for i in self.cfg['layers_shape']:
            shape_str += str(i) + '_'
 
        if self.cfg['multi_domain']:
            domainString = 'multi_domain'
        else:  
            domainString = self.cfg['domainString']

        model_path = os.path.join(self.cfg['base_dir'], self.cfg['saved_models_dir'], domainString, self.cfg['policyType'],
                                  shape_str[:-1], str(self.cfg['model_id'])) 
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        checkpoint_path = os.path.join(model_path, 'autoencoder')
        self.saver.save(self.sess, checkpoint_path)     

    def restore_variables(self):
        print('restore_variables')
        saver = tf.train.Saver(tf.trainable_variables())

        shape_str = ''
        for i in self.cfg['layers_shape']:
            shape_str += str(i) + '_' 

        model_path = os.path.join(self.cfg['base_dir'], self.cfg['saved_models_dir'], self.cfg['domainString'], self.cfg['policyType'], 
                                  shape_str[:-1], str(self.cfg['model_id']))
        ckpt = tf.train.get_checkpoint_state(model_path)     
        saver.restore(self.sess, ckpt.model_checkpoint_path) 
         
           
        


