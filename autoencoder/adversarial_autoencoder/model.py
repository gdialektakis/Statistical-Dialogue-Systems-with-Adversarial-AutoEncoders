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

# def get_weight_variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d()):
#     if shape is None:
#         return tf.get_variable(name)
#     else:
#         return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)
#
#
# def get_bias_variable(name, shape=None, initializer=tf.constant_initializer(value=0.0, dtype=_FLOATX)):
#     if shape is None:
#         return tf.get_variable(name)
#     else:
#         return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)


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
        weights = tf.get_variable("weights", shape=[n1, n2], initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        out = tf.matmul(x, weights)
        return out


"""
Adversarial Autoencoder
"""


class Autoencoder(object):
    def __init__(self, domainString, policyType, variable_scope_name):
        self.cfg, self.learning_rate_params, self.optim_params = get_configuration(domainString, policyType)
        cfg = self.cfg
        #self.l2 = float(cfg['l2'])
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
            #self.create_variables()
            self.define_operations()

            if cfg['restore_model']:
                try:
                    self.restore_variables()
                    print("Variables restored for " + domainString + ' AAE ' + str(self.layers_shape))
                except:
                    print("Variables restore Failed for " + domainString + ' AAE!')
                    pass

    # def create_variables(self):
    #     for domain in self.domains:
    #         get_weight_variable('W_' + domain, (self.state_dim[domain], self.layers_shape[1]))
    #
    #     for i in range(1, self.n_layers):
    #         get_weight_variable('W' + str(i), (self.layers_shape[i], self.layers_shape[i + 1]))

    def encoder(self, X, reuse=False):
        """
        Encode part of the autoencoder.
        :param x: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):

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
        self.real_distribution = tf.placeholder(dtype=tf.float32, shape=[None, self.encode_dim], name='Real_distribution')

        self.prob = tf.placeholder_with_default(1.0, shape=())

        Y = self.X_data_placeholder

        with tf.variable_scope(tf.get_variable_scope()):
            self.encoder_output = self.encoder(Y)
            self.decoder_output = self.decoder(self.encoder_output)

        with tf.variable_scope(tf.get_variable_scope()):
            self.d_real = self.discriminator(self.real_distribution)
            self.d_fake = self.discriminator(self.encoder_output, reuse=True)  ### We need to check reuse = True??

        # Autoencoder loss
        self.autoencoder_loss = tf.reduce_mean(tf.square(self.X_data_placeholder - self.decoder_output))

        # Discrimminator Loss
        self.dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_real), logits=self.d_real))
        self.dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_fake), logits=self.d_fake))
        self.dc_loss = self.dc_loss_fake + self.dc_loss_real

        # Generator loss
        self.generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_fake), logits=self.d_fake))

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
        self.discriminator_optimizer = optimizer.minimize(self.dc_loss, var_list=self.dc_var, global_step=self.global_step)
        self.generator_optimizer = optimizer.minimize(self.generator_loss, var_list=self.en_var, global_step=self.global_step)

        # Saving the model
        self.saver = tf.train.Saver()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

    def train(self, input_x):
        if self.trainable:
            a_mean_loss = 0
            d_mean_loss = 0
            g_mean_loss = 0
            for _ in range(self.epochs):
                z_real_dist = np.random.randn(self.encode_dim) * 5.  ## check how to generate the desired prior distribution
                tmp_a_mean_loss, _ = self.sess.run(self.autoencoder_optimizer, feed_dict={self.X_data_placeholder: input_x})
                tmp_d_mean_loss, _ = self.sess.run(self.discriminator_optimizer,
                                        feed_dict={self.X_data_placeholder: input_x, self.real_distribution: z_real_dist})
                tmp_g_mean_loss, _ = self.sess.run(self.generator_optimizer, feed_dict={self.X_data_placeholder: input_x})
                # Computing losses
                a_mean_loss += tmp_a_mean_loss
                d_mean_loss += tmp_d_mean_loss
                g_mean_loss += tmp_g_mean_loss

            a_mean_loss = float(a_mean_loss) / float(self.epochs)
            d_mean_loss = float(d_mean_loss) / float(self.epochs)
            g_mean_loss = float(g_mean_loss) / float(self.epochs)

            # toDo: save mean loss for graph plot
            print('Autoencoder ' + self.domainString + ' mean loss:', a_mean_loss)
            print('Disciminator ' + self.domainString + ' mean loss:', d_mean_loss)
            print('Generator ' + self.domainString + ' mean loss:', g_mean_loss)
            if math.isnan(a_mean_loss):
                print('Autoencoder returned cost Nan')
            lr = (self.sess.run(self.learning_rate))
            print("Learning rate: %.10f" % lr)
            self.save_learning_rate(dstring=self.domainString, learning_rate=lr)

    def encode(self, X_data):
        X_encoded_np = self.sess.run(self.encoder_output, feed_dict={self.X_data_placeholder: X_data, self.prob: 1.0})

        return X_encoded_np

    def save_variables(self):
        print('save autoencoder')
        shape_str = ''
        for i in self.cfg['layers_shape']:
            shape_str += str(i) + '_'

        if self.cfg['multi_domain']:
            domainString = 'multi_domain'
        else:
            domainString = self.cfg['domainString']

        model_path = os.path.join(self.cfg['base_dir'], self.cfg['saved_models_dir'], domainString,
                                  self.cfg['policyType'], str(self.cfg['model_id']),
                                  shape_str[:-1])
        '''model_path = os.path.join(self.cfg['base_dir'], self.cfg['saved_models_dir'], self.environment,
                                  domainString, self.cfg['policyType'], str(self.cfg['model_id']),
                                  shape_str[:-1])'''
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        checkpoint_path = os.path.join(model_path, 'autoencoder')

        path = self.saver.save(self.sess, checkpoint_path)

        return path

    def restore_variables(self):
        print('restore_variables')
        saver = tf.train.Saver(tf.trainable_variables())

        shape_str = ''
        for i in self.cfg['layers_shape']:
            shape_str += str(i) + '_'

        model_path = os.path.join(self.cfg['base_dir'], self.cfg['saved_models_dir'], self.cfg['domainString'],
                                  self.cfg['policyType'], str(self.cfg['model_id']),
                                  shape_str[:-1])
        '''model_path = os.path.join(self.cfg['base_dir'], self.cfg['saved_models_dir'], self.environment,
                                          domainString, self.cfg['policyType'], str(self.cfg['model_id']),
                                          shape_str[:-1])'''
        if not os.path.exists(model_path):
            print(model_path, "Does not exist")
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    """
    save states in the buffer from every domain
    """

    def saveToStateBuffer(self, vector):
        if self.trainable:
            self.state_buffer.append(vector.reshape((1, -1)))

    def checkReadyToTrain(self):
        if len(self.state_buffer) >= self.batch_size:
            return True
        else:
            return False

    def getTrainBatch(self):
        state_batch = np.concatenate(self.state_buffer[:self.batch_size], axis=0)
        return state_batch

    # todo: use more sample on the go
    def resetStateBuffer(self):
        self.state_buffer = self.state_buffer[self.batch_size:]

    def loadEpisodes(self):
        data = None
        try:
            print("Loading past experiences.")
            data = np.load("episodes_log.csv")
            print(data)
        except:
            print("FAILED: Loading past experiences.")
            pass

        if data is not None:
            for state in data:
                self.saveToStateBuffer(state)

    def saveEpisodesToFile(self, saveEpisodes):
        if saveEpisodes:
            if os.path.isfile("episodes_log.csv"):
                saved = np.loadtxt("episodes_log.csv", delimiter=',')
                for state in self.state_buffer:
                    np.append(saved, state)
                np.savetxt("episodes_log.csv", saved, delimiter=",")
            else:
                np.savetxt("episodes_log.csv", self.state_buffer, delimiter=",")

        print(np.loadtxt("episodes_log.csv", delimiter=','))

    # not working properly yet
    # todo:fix
    def get_tensorflow_device(self):
        if self.cfg['training_device'] == 'cpu':
            return '/cpu:0'
        else:
            return '/gpu:0'

    def save_learning_rate(self, dstring, learning_rate):
        with open(dstring + '_ae_learning_rate_logs.csv', 'a+') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([learning_rate])

    def load_learning_rate(self, dstring, learning_rate_params):
        with open(dstring + '_ae_learning_rate_logs.csv', 'r+') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            lr_list = [entry for entry in reader]
            lr = float(lr_list[-1][0])
            if lr != 0.0:
                learning_rate_params['learning_rate'] = lr

                # erase old values from learning rate log files
                # writer = csv.writer(csvfile, delimiter=',')
                # writer.writerow([0.0])

        return learning_rate_params



