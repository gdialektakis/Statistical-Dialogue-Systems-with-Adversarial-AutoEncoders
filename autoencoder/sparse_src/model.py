from __future__ import division, print_function

__author__ = """experiments Lygerakis"""

import sys
import os
import logging
import math
import tensorflow as tf
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
Sparce AutoEncoder Implementation
based on Mr.Tsiaras code for Autoencoders
"""

class Autoencoder(object):

    def __init__(self, domainString, policyType, variable_scope_name):
        self.cfg, self.learning_rate_params, self.optim_params = get_configuration(domainString, policyType)
        cfg = self.cfg

        # sparse parameters
        self.rho = cfg['rho']
        self.beta = cfg['beta']

        self.l2 = cfg['l2']
        self.layers_shape = cfg['layers_shape']
        self.n_layers = len(self.layers_shape) - 1
        self.input_dim = cfg['input_dim'] #self.layers_shape[0]
        self.encode_dim = self.layers_shape[-1]
        self.train_model = cfg['train_model']
        self.domainString = cfg['domainString'] 
        self.domains = cfg['domains']
        self.state_dim = cfg['state_dim']

        # fotis
        self.input_dim = cfg['input_dim']
        self.isMulti = cfg["multi_domain"]
        #self.project_dim = cfg['project_dim']
        self.batch_size = cfg['batch_size']
        self.state_buffer = []
        self.dropout_rate = cfg['dropout_rate']
        self.epochs = cfg['epochs']
        self.trainable = cfg['trainable']

        setup_logger('msg_logger', cfg['msg_logging_dir'], level=logging.INFO)

        """
        Find where to close the Session
        """
        self.sess = tf.Session()
        with tf.variable_scope("autoencoder", reuse=tf.AUTO_REUSE):
            self.create_variables()
            self.define_operations()

            if cfg['restore_model']:
                try:
                    self.restore_variables()
                    print("Variables restored")
                except:
                    print("Variables Restore Failed!")
                    pass
      
  

    def create_variables(self):
        for domain in self.domains:
            get_weight_variable('W_'+domain, (self.state_dim[domain], self.layers_shape[1])) 

        for i in range(1, self.n_layers):
            get_weight_variable('W'+str(i), (self.layers_shape[i], self.layers_shape[i+1]))
   

    def encoder(self, X):
        W = get_weight_variable('W_'+self.domainString)
        Y = tf.nn.sigmoid(tf.matmul(X, W))

        for i in range(1, self.n_layers):
            W = get_weight_variable('W' + str(i))
            Y = tf.nn.sigmoid(tf.matmul(Y, W))

        return Y

    def decoder(self, Y):
        X = Y
        for i in range(self.n_layers-1, 0, -1):
            W = get_weight_variable('W' + str(i))
            X = tf.nn.sigmoid(tf.matmul(X, tf.transpose(W)))

        W = get_weight_variable('W_'+self.domainString)
        X = tf.nn.sigmoid(tf.matmul(X, tf.transpose(W)))

        return X 
            
    #sparsity penalty
    def KL_Divergence(self):
        return self.rho * tf.log( self.rho / self.rho_hat ) + ( 1 - self.rho ) * tf.log( ( 1 - self.rho ) / ( 1 - self.rho_hat ) )

    def define_operations(self):
        self.X_data_placeholder = tf.placeholder(dtype=_FLOATX, shape=[None, self.input_dim]) 
        self.X_encoded = self.encoder(self.X_data_placeholder)
        # Average hidden layer over all data points in X
        self.rho_hat = tf.reduce_mean(self.X_encoded, 0)

        """
        Sometimes I have zeros in the X_encoded tensor 
        and therefore rho_hat contains INF so I discard them
        """
        #changed activation func: ReLU gave me zero activations which explode the log
        self.kl = tf.reduce_sum(self.KL_Divergence())
        X_reconstructed = self.decoder(self.X_encoded)

        self.train_loss = tf.reduce_mean(tf.pow(self.X_data_placeholder - X_reconstructed, 2))

        # Regularization loss | alpha = l2
        if self.l2 is not None and self.l2 > .0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not('_b' in v.name)])
            #self.train_loss += self.l2*l2_loss
            self.train_loss += self.alpha * l2_loss

        self.train_loss = self.train_loss + self.beta * self.kl

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
        kl = self.sess.run(self.kl, feed_dict={self.X_data_placeholder: X})
        mean_loss, _ = self.sess.run([self.train_loss, self.update_ops], feed_dict={self.X_data_placeholder: X})
        print('Autoencoder mean loss:', mean_loss)
        if math.isnan(mean_loss):
            print('Autoencoder returned cost Nan')

        print("KL value: ", kl)



    def encode(self, X_data):
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

    def saveToStateBuffer(self, vector, vector_clean):
        self.state_buffer.append(vector.reshape((1, -1)))
        self.state_clean_buffer.append(vector_clean.reshape((1, -1)))

    def checkReadyToTrain(self):
        if len(self.state_buffer) >= self.batch_size:
            return True
        else:
            return False

    def getTrainBatch(self):
        state_batch = np.concatenate(self.state_buffer[:self.batch_size], axis=0)
        state_clean_batch = np.concatenate(self.state_clean_buffer[:self.batch_size], axis=0)
        return state_batch, state_clean_batch

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
    #not working properly yet
    #todo:fix
    def get_tensorflow_device(self):
        if self.cfg['training_device'] == 'cpu':
            return '/cpu:0'
        else:
            return '/gpu:0'

    def save_learning_rate(self, dstring, learning_rate):
        with open(dstring + '_dae_learning_rate_logs.csv', 'a+') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([learning_rate])

    def load_learning_rate(self, dstring, learning_rate_params):
        with open(dstring + '_dae_learning_rate_logs.csv', 'r+') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            lr_list = [entry for entry in reader]
            lr = float(lr_list[-1][0])
            if lr != 0.0:
                learning_rate_params['learning_rate'] = lr

                # erase old values from learning rate log files
                #writer = csv.writer(csvfile, delimiter=',')
                #writer.writerow([0.0])

        return learning_rate_params
        


