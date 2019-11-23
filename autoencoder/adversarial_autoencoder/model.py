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
        # variational
        self.latent_size = cfg["latent_size"]

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
        pass

    def define_operations(self):
        pass

    def restore_variables(self):
        pass