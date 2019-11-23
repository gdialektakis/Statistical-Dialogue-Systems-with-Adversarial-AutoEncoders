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