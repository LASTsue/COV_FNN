import random
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch


from Pre_data import Z_Data

def get_logger(name='log'):
    logging.basicConfig(level=logging.INFO,filename=f'result/log/{name}.log',
                        format='%(levelname)s - %(asctime)s  - %(message)s')
    logger = logging.getLogger("COVID_FNN")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(asctime)s  - %(message)s")
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

def draw_loss(loss):
    plt.clf()
    plt.plot(loss)
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #保存图片
    plt.savefig('result/loss/loss.png')

def draw_acc(acc):
    #清空画布
    plt.clf()
    plt.plot(acc)
    plt.title('acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    #保存图片
    plt.savefig('result/acc/acc.png')
