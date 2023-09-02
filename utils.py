import random
import sys
import logging
import numpy as np

import torch


from Pre_data import Z_Data

def get_logger():
    logging.basicConfig(level=logging.INFO,filename='result/log/log.log',
                        format='%(asctime)s  - %(message)s')
    logger = logging.getLogger("COVID_FNN")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s  - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_data_train(image_type):
    d=Z_Data('train',image_type)
    return d

def get_data_val(image_type):
    d=Z_Data('val',image_type)
    return d

def get_data_test(image_type):
    d=Z_Data('test',image_type)
    return d

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
