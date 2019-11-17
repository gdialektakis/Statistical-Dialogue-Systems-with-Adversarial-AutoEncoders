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
        self.dense_layers_shape = cfg['dense_layers_shape']
        self.n_dense_layers = len(self.dense_layers_shape) - 1
        self.input_dim = cfg['input_dim']
        self.encode_dim = self.dense_layers_shape[-1]
        self.n_bins_list =cfg['n_bins_list']
        self.train_model = cfg['train_model']
        self.n_channels_list = cfg['n_channels_list']
        self.n_conv_layers = len(self.n_channels_list) - 1
        self.non_linearity = cfg['non_linearity_fn'] 
        self.input_height = 1
        self.input_width = self.input_dim
        self.dense_input_dim = cfg['dense_input_dim'] 
        self.pooling_type = cfg['pooling_type']
        self.stride = cfg['stride']

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

        for domain in self.domains:
            # get_weight_variable('W_project_'+domain, (self.input_dim, self.project_dim))

            # projection layer: projects domain layers to projection layer
            get_weight_variable('W_project_' + domain, (self.state_dim[domain], self.project_dim))

        width = self.project_dim

        for i in range(self.n_conv_layers):
            get_weight_variable('W_conv2d_' + str(i + 1), (1, 3, self.n_channels_list[i], self.n_channels_list[i + 1]))
            get_bias_variable('b_conv2d_' + str(i + 1), shape=(self.n_channels_list[i + 1]))

            get_weight_variable('W_deconv2d_' + str(i + 1),
                                (1, 3, self.n_channels_list[i], self.n_channels_list[i + 1]))
            get_bias_variable('b_deconv2d_' + str(i + 1), shape=(self.n_channels_list[i]))
            width = int(np.ceil(width / self.stride))

        dense_input_dim = width * self.n_channels_list[-1]
        self.dense_layers_shape[0] = dense_input_dim

        for i in range(self.n_dense_layers):
            get_weight_variable('W_dense_encoder_' + str(i + 1),
                                (self.dense_layers_shape[i], self.dense_layers_shape[i + 1]))
            get_bias_variable('b_dense_encoder_' + str(i + 1), shape=(self.dense_layers_shape[i + 1]))

            get_weight_variable('W_dense_decoder_' + str(i + 1),
                                (self.dense_layers_shape[i + 1], self.dense_layers_shape[i]))
            get_bias_variable('b_dense_decoder_' + str(i + 1), shape=(self.dense_layers_shape[i]))

    def encoder(self, X):
        W = get_weight_variable('W_project_' + self.domainString)
        Y = tf.nn.relu(tf.matmul(X, W))

        Y = tf.reshape(Y, [tf.shape(Y)[0], 1, self.project_dim, 1])

        self.shapes = []
        for i in range(self.n_conv_layers):
            self.shapes.append(Y.get_shape().as_list())

            W = get_weight_variable('W_conv2d_' + str(i + 1))
            b = get_bias_variable('b_conv2d_' + str(i + 1))

            Y = tf.nn.conv2d(Y, W, strides=[1, 1, 1, 1], padding='SAME') + b
            Y = self.non_linearity(Y)

            # Y = tf.nn.max_pool(Y, ksize=[1, 1, self.stride, 1], strides=[1, 1, self.stride, 1], padding='SAME')
            Y = tf.nn.avg_pool(Y, ksize=[1, 1, self.stride, 1], strides=[1, 1, self.stride, 1], padding='SAME')

        self.shapes.append(Y.get_shape().as_list())
        Y = tf.layers.flatten(Y)

        for i in range(self.n_dense_layers):
            W = get_weight_variable('W_dense_encoder_' + str(i + 1))
            b = get_bias_variable('b_dense_encoder_' + str(i + 1))
            Y = self.non_linearity(tf.matmul(Y, W) + b)

        return Y

    def decoder(self, Y):
        X = Y
        for i in range(self.n_dense_layers - 1, -1, -1):
            W = get_weight_variable('W_dense_decoder_' + str(i + 1))
            b = get_bias_variable('b_dense_decoder_' + str(i + 1))
            X = self.non_linearity(tf.matmul(X, W) + b)

        shape = self.shapes[self.n_conv_layers]
        X = tf.reshape(X, [tf.shape(X)[0], shape[1], shape[2], shape[3]])

        for i in range(self.n_conv_layers - 1, -1, -1):
            shape = self.shapes[i]
            output_shape = tf.stack([tf.shape(Y)[0], shape[1], shape[2], shape[3]])

            W = get_weight_variable('W_deconv2d_' + str(i + 1))
            b = get_bias_variable('b_deconv2d_' + str(i + 1))

            X = tf.nn.conv2d_transpose(X, W, output_shape=output_shape, strides=[1, 1, self.stride, 1],
                                       padding='SAME') + b
            X = self.non_linearity(X)

        X = tf.layers.flatten(X)

        W = get_weight_variable('W_project_' + self.domainString)
        X = tf.nn.relu(tf.matmul(X, tf.transpose(W)))

        return X

    def define_operations(self):
 
        self.X_data_placeholder = tf.placeholder(dtype=_FLOATX, shape=[None, self.input_height, self.input_width, self.n_channels_list[0]]) 

        self.X_encoded = self.encoder(self.X_data_placeholder)
        X_reconstructed = self.decoder(self.X_encoded) # Network prediction

        # Loss of train data
        self.train_loss = tf.reduce_mean(tf.pow(self.X_data_placeholder - X_reconstructed, 2)) 
        #self.train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.X_data_placeholder, logits=X_reconstructed))   

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
        input_shape = X.shape
        X = np.reshape(X, (input_shape[0], 1, input_shape[1], 1))

        mean_loss, _ = self.sess.run([self.train_loss, self.update_ops], 
                feed_dict={self.X_data_placeholder: X}) 

        print('Autoencoder mean loss:', mean_loss)
        if math.isnan(mean_loss):
            print('Autoencoder returned cost Nan')



    def encode(self, X_data):
        input_shape = X_data.shape
        X_data = np.reshape(X_data, (input_shape[0], 1, input_shape[1], 1)) 

        X_encoded_np = self.sess.run(self.X_encoded, feed_dict={self.X_data_placeholder: X_data})

        return X_encoded_np


    def define_decode_operations(self):
        self.X_encoded_placeholder = tf.placeholder(dtype=_FLOATX, shape=(None, self.encode_dim)) 

        self.X_decoded = self.decoder(self.X_encoded_placeholder)


    def decode(self, X_encoded):
        X_decoded = self.sess.run(self.X_decoded, feed_dict={self.X_encoded_placeholder: X_encoded})

        return X_decoded
 

    def save_variables(self):
        print('save autoencoder')
        shape_str = ''
        for i in self.cfg['dense_layers_shape']:
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
         
           
        


