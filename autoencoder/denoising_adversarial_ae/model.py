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
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())  # check the initializer
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out



"""
Denoising Adversarial Autoencoder
"""

class Autoencoder(object):
    def __init__(self, domainString, policyType, variable_scope_name):
        self.cfg, self.learning_rate_params, self.optim_params = get_configuration(domainString, policyType)
        cfg = self.cfg
        # self.l2 = float(cfg['l2'])
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
        # discriminator model
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
            # self.create_variables()
            self.define_operations()

            if cfg['restore_model']:
                try:
                    self.restore_variables()
                    print("Variables restored for " + domainString + ' DAAE ' + str(self.layers_shape))
                except:
                    print("Variables restore Failed for " + domainString + ' DAAE!')
                    pass


    def encoder(self, X, reuse=False):
        """
        Encode part of the autoencoder.
        :param x: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            if self.n_layers == 4:
                input_layer = tf.nn.tanh(dense(X, self.input_dim, self.layers_shape[1], 'e_input_layer'))
                input_layer = tf.nn.dropout(input_layer, keep_prob=self.prob)
                e_dense1 = tf.nn.tanh(dense(input_layer, self.layers_shape[1], self.layers_shape[2], 'e_dense1'))
                e_dense1 = tf.nn.dropout(e_dense1, keep_prob=self.prob)
                e_dense2 = tf.nn.tanh(dense(e_dense1, self.layers_shape[2], self.layers_shape[3], 'e_dense2'))
                e_dense2 = tf.nn.dropout(e_dense2, keep_prob=self.prob)
                latent_variable = tf.nn.tanh(
                    dense(e_dense2, self.layers_shape[3], self.layers_shape[4], 'e_latent_variable'))
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
            if self.n_layers == 4:
                d_dense1 = tf.nn.tanh(dense(Y, self.layers_shape[4], self.layers_shape[3], 'd_dense1'))
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
        self.X_data_clean_placeholder = tf.placeholder(dtype=_FLOATX, shape=[None, self.input_dim])
        self.real_distribution = tf.placeholder(dtype=tf.float32, shape=[None, self.encode_dim],
                                                name='Real_distribution')

        self.prob = tf.placeholder_with_default(1.0, shape=())

        Y = self.X_data_placeholder

        with tf.variable_scope(tf.get_variable_scope()):
            self.encoder_output = self.encoder(Y)
            self.decoder_output = self.decoder(self.encoder_output)

        with tf.variable_scope(tf.get_variable_scope()):
            self.d_real = self.discriminator(self.real_distribution)
            self.d_fake = self.discriminator(self.encoder_output, reuse=True)  ### We need to check reuse = True??

        # Autoencoder loss
        self.autoencoder_loss = tf.reduce_mean(tf.square(self.X_data_clean_placeholder - self.decoder_output))

        # Discriminator Loss
        self.dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_real), logits=self.d_real))
        self.dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_fake), logits=self.d_fake))
        self.dc_loss = self.dc_loss_fake + self.dc_loss_real

        # Generator loss
        self.generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_fake), logits=self.d_fake))

        # collect only Discriminator's and Generator's variables as required
        all_variables = tf.trainable_variables()
        self.dc_var = [var for var in all_variables if 'dc_' in var.name]
        self.en_var = [var for var in all_variables if 'e_' in var.name]

        # define learning rate decay method
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = get_learning_rate(self.cfg['learning_rate_method'],
                                               self.global_step, self.learning_rate_params)


        # define the optimization algorithm
        opt_name = self.cfg['optimization_algorithm'].lower()
        optimizer = get_optimizer(opt_name, self.learning_rate, self.optim_params)

        # Optimizers
        self.autoencoder_optimizer = optimizer.minimize(self.autoencoder_loss, global_step=self.global_step)
        self.discriminator_optimizer = optimizer.minimize(self.dc_loss, var_list=self.dc_var,
                                                          global_step=self.global_step)
        self.generator_optimizer = optimizer.minimize(self.generator_loss, var_list=self.en_var,
                                                      global_step=self.global_step)

        # Saving the model
        self.saver = tf.train.Saver()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)