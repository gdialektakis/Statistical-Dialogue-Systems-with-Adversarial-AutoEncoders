from __future__ import division, print_function

__author__ = """George Dialektakis (geo4diale@gmail.com)"""
#    George Dialektakis <geo4diale@gmail.com>
#    Computer Science Department, University of Crete.

import sys
import os
import logging
import math
import tensorflow as tf
import matplotlib.pyplot as plt
# from lib.optimizers import get_learning_rate, get_optimizer
from lib.model_io import get_configuration, setup_logger
from lib.optimizers import get_learning_rate, get_optimizer
from lib.precision import _FLOATX
import numpy as np
import csv

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

def dense(x, n1, n2, name):
        """
        Used to create a dense layer.
        :param x: input tensor to the dense layer
        :param n1: no. of input neurons
        :param n2: no. of output neurons
        :param name: name of the entire dense layer.i.e, variable scope name.
        :return: tensor with shape [batch_size, n2]
        """
        with tf.variable_scope(name, reuse=None):
            weights = tf.get_variable("weights", shape=[n1, n2],
                                      initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
            out = tf.matmul(x, weights)
            return out


"""
Adversarial Autoencoder
"""


class Autoencoder(object):
    def __init__(self,domainString, policyType, variable_scope_name):
        self.cfg, self.learning_rate_params, self.optim_params = get_configuration(domainString, policyType)
        cfg = self.cfg
        self.l2 = float(cfg['l2'])
        self.layers_shape = cfg['layers_shape']
        self.n_layers = len(self.layers_shape) - 1
        self.input_dim = cfg['input_dim']  # self.layers_shape[0]
        self.encode_dim = self.layers_shape[-1]
        self.train_model = cfg['train_model']
        self.domainString = cfg['domainString']
        self.domains = cfg['domains']
        self.state_dim = cfg['state_dim']
        self.isMulti = cfg["multi_domain"]
        self.batch_size = cfg['batch_size']
        self.state_buffer = []
        self.dropout_rate = cfg['dropout_rate']
        self.epochs = cfg['epochs']
        self.trainable = cfg['trainable']
        #discriminator model
        self.discr_layers_shape = cfg['discriminator_layers']

        try:
            self.learning_rate_params = self.load_learning_rate(domainString, self.learning_rate_params)
        except:
            pass

        setup_logger('msg_logger', cfg['msg_logging_dir'], level=logging.INFO)

        """
        Find where to close the Session
        """
        self.sess = tf.Session()
        # device = self.get_tensorflow_device()
        # with tf.device(device):
        with tf.variable_scope("autoencoder_" + variable_scope_name, reuse=tf.AUTO_REUSE):
            self.create_variables()
            self.define_operations()

            if cfg['restore_model']:
                try:
                    self.restore_variables()
                    print("Variables restored for " + domainString + ' AAE ' + str(self.layers_shape))
                except:
                    print("Variables restore Failed for " + domainString + ' AAE!')
                    pass


    def load_learning_rate(self, dstring, learning_rate_params):
        with open(dstring + '_aae_learning_rate_logs.csv', 'r+') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            lr_list = [entry for entry in reader]
            lr = float(lr_list[-1][0])
            if lr != 0.0:
                learning_rate_params['learning_rate'] = lr

                # erase old values from learning rate log files
                # writer = csv.writer(csvfile, delimiter=',')
                # writer.writerow([0.0])

        return learning_rate_params

    def create_variables(self):
        for domain in self.domains:
            get_weight_variable('W_' + domain, (self.state_dim[domain], self.layers_shape[1]))

        for i in range(1, self.n_layers):
            get_weight_variable('W' + str(i), (self.layers_shape[i], self.layers_shape[i + 1]))


    def encoder(self, X, reuse=False):
        """
        Encode part of the autoencoder.
        :param x: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            # W = get_weight_variable('W_' + self.domainString)
            # Y = tf.nn.tanh(tf.matmul(X, W))
            #
            # Y = tf.nn.dropout(Y, keep_prob=self.prob)

            if self.n_layers == 3:
                input_layer = tf.nn.tanh(dense(X, self.input_dim, self.layers_shape[1], 'e_input_layer'))
                input_layer = tf.nn.dropout(input_layer, keep_prob=self.prob)
                e_dense1 = tf.nn.tanh(dense(input_layer, self.layers_shape[1], self.layers_shape[2], 'e_dense1'))
                e_dense1 = tf.nn.dropout(e_dense1, keep_prob=self.prob)
                e_dense2 = tf.nn.tanh(dense(e_dense1, self.layers_shape[2], self.layers_shape[3], 'e_dense2'))
                e_dense2 = tf.nn.dropout(e_dense2, keep_prob=self.prob)
                latent_variable = tf.nn.tanh(dense(e_dense2, self.layers_shape[3], self.layers_shape[4], 'e_latent_variable'))
                latent_variable = tf.nn.dropout(latent_variable, keep_prob=self.prob)

            return latent_variable


    def decoder(self, Y, reuse=False):
        """
        Decoder part of the autoencoder.
        :param x: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """

        with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
            X = Y
            if self.n_layers == 3:
                d_dense1 = tf.nn.tanh(dense(X, self.layers_shape[4], self.layers_shape[3], 'd_dense1'))
                d_dense2 = tf.nn.tanh(dense(d_dense1, self.layers_shape[3], self.layers_shape[2], 'd_dense2'))
                d_dense3 = tf.nn.tanh(dense(d_dense2, self.layers_shape[2], self.layers_shape[1], 'd_dense3'))
                output = tf.nn.tanh(dense(d_dense3, self.layers_shape[1], self.input_dim, 'd_output'))

            return output


    def discriminator(self, X, reuse=False):
        """
        Discriminator that is used to match the posterior distribution with a given prior distribution.
        :param x: tensor of shape [batch_size, z_dim]
        :param reuse: True -> Reuse the discriminator variables,
                      False -> Create or search of variables before creating
        :return: tensor of shape [batch_size, 1]
        """

        with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
            dc_den1 = tf.nn.relu(dense(X, self.layers_shape[-1], self.discr_layers_shape[1], name='dc_den1'))
            dc_den2 = tf.nn.relu(dense(dc_den1, self.discr_layers_shape[1], self.discr_layers_shape[2], name='dc_den2'))
            dc_den3 = tf.nn.relu(dense(dc_den2, self.discr_layers_shape[2], self.discr_layers_shape[3], name='dc_den3'))
            output = dense(dc_den3, self.discr_layers_shape[3], 1, name='dc_output')

            return output

    def define_operations(self):
        self.X_data_placeholder = tf.placeholder(dtype=_FLOATX, shape=[None, self.input_dim])
        self.prob = tf.placeholder_with_default(1.0, shape=())

        self.X_encoded = self.encoder(self.X_data_placeholder)
        X_reconstructed = self.decoder(self.X_encoded)  # Network prediction

        # Create a scalar summary object for the encoder and decoder so it can be displayed
        self.tf_encode_summary = tf.summary.scalar('loss', self.X_encoded)
        self.tf_decode_summary = tf.summary.scalar('loss', X_reconstructed)

        # Autoencoder loss
        self.train_loss = tf.reduce_mean(tf.pow(self.X_data_placeholder - X_reconstructed, 2))



    def restore_variables(self):
        pass



