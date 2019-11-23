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