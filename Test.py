import numpy as np
import torch
from Cla_net import Cla_net
import utils as ut
from torch.utils import data
import torch.nn as nn
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score

bt=100
model_path='result/model/cla_net_9.pth'
device=torch.device('cuda:0')
test_data=ut.get_data_test(0)
test_loader=data.DataLoader(test_data,batch_size=bt)

print(f'test data num:{len(test_data)}')
counts=test_data.get_num()
print(f'covid:{counts[0]} normal:{counts[1]} lung:{counts[2]} pne:{counts[3]} ')

test_net=Cla_net()
test_net.load_state_dict(torch.load(model_path))
test_net=test_net.to(device)
test_net.eval()

with torch.no_grad():
    for img,label in test_loader:
        img=img.to(device)
        label=label.to(device)
        output=test_net(img)
        #通过sklearn计算acc，precision，recall，f1
        output=torch.argmax(output,dim=1)
        output=output.cpu().numpy()
        label=label.cpu().numpy()
        acc=accuracy_score(label,output)
        pre=precision_score(label,output,average='weighted',zero_division=1)
        rec=recall_score(label,output,average='weighted',zero_division=1)
        f1=f1_score(label,output,average='weighted',zero_division=1)
        #在logger中保留四位小数
        print("test====acc:{:.4f},pre:{:.4f},rec:{:.4f},f1:{:.4f}".format(acc,pre,rec,f1))
