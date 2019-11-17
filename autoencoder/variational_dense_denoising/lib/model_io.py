from __future__ import division, print_function

import os
import logging
import json
import numpy as np
import tensorflow as tf

__author__ = """Vassilis Tsiaras (tsiaras@csd.uoc.gr)"""
#    Vassilis Tsiaras <tsiaras@csd.uoc.gr>
#    Computer Science Department, University of Crete.

def get_configuration(domainString, policyType):

    config_filename = os.path.join('.',  'autoencoder', 'config', 'config_params_vdae.json')
    with open(config_filename, 'r') as f:
        cfg = json.load(f)  

    cfg['domainString'] = domainString 

    cfg['policyType'] = policyType

    #cfg['layers_shape'][0] = int(cfg['state_dim'][domainString])
    cfg['input_dim'] = int(cfg['state_dim'][domainString])
    if not cfg['multi_domain']:
        cfg['layers_shape'][0] = int(cfg['state_dim'][domainString])

    cfg['msg_logging_dir'] = os.path.join('.', cfg['base_dir'], cfg['logging_dir'], 'log_'+str(cfg['model_id'])+'.txt') 

    optim_filename = os.path.join('.', 'autoencoder', 'config', cfg['optimization_params'])
    with open(optim_filename, 'r') as f:
        optim_params = json.load(f)

    return cfg, optim_params['learning_rate_method'][cfg['learning_rate_method']], optim_params['optimization_algorithm'][cfg['optimization_algorithm']]

def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)


def read_model_id(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as fid:
            model_id = int(fid.read())
        fid.close()
    else:
        write_model_id(filename, 1)
        model_id = 0
        
    return model_id      


def write_model_id(filename, model_id):
    model_id_txt = str(model_id) 
    with open(filename, 'w') as fid:
        fid.write(model_id_txt)
    fid.close() 

def get_modle_id(cfg):
    model_id_filename = os.path.join(cfg['base_dir'], cfg['saved_models_dir'], cfg['domainString'], cfg['model_ids']) 
    model_id = read_model_id(model_id_filename) + 1 # Reserve the next model_id. If file does not exists then create it 
    write_model_id(model_id_filename, model_id) 

    return model_id 






