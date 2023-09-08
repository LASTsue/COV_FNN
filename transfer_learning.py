import numpy as np
import torch
from Qfnn import Qfnn
import utils as ut
from torch.utils import data
import torch.nn as nn
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
import pennylane as qml
from Cla_net import Cla_net

tr_batch_size=40
va_batch_size=100
lr=0.001
epochs=200
image_type=0
seed=777
device=torch.device('cuda:0')

load_cla_model_path='result/model/cla_net_10epochs.pth'
load_qfnn_model_path='result/model/trans_net_20.pth'


logger=ut.get_logger('Transfer_Train')
ut.setup_seed(seed)

#确认随机seed是否生效
# for i in range(10):
#     print(torch.randint(0,1000,(1,)))

logger.critical("============================Start print log============================")


qu_net=Qfnn(load_cla_model_path).to(device)
# qu_net.load_state_dict(torch.load(load_qfnn_model_path))

# for param in cla_net.parameters():
#     param.requires_grad = False

logger.critical("transfer_net have {} paramerters in total".format(
    sum(x.numel() for x in qu_net.parameters())
    ))

train_data=ut.get_data_train(image_type)
train_loader=data.DataLoader(
    train_data,batch_size=tr_batch_size,shuffle=True,num_workers=2
    )
train_loader_len=len(train_loader)

val_data=ut.get_data_val(image_type)
val_loader=data.DataLoader(
    val_data,batch_size=va_batch_size,shuffle=True,num_workers=2
    )

conunts=train_data.get_num()
logger.critical("train data length:{}".format(len(train_data)))
logger.critical("train data have {} covid,{} normal,{} lung,{} pne".format(
    conunts[0],conunts[1],conunts[2],conunts[3]
    ))

conunts=val_data.get_num()
logger.critical("val data length:{}".format(len(val_data)))
logger.critical("val data have {} covid,{} normal,{} lung,{} pne".format(
    conunts[0],conunts[1],conunts[2],conunts[3]
    ))

#加载优化器，损失函数
optimizer=torch.optim.Adam(qu_net.parameters(),lr=lr)
#SGD优化器
# optimizer=torch.optim.SGD(qu_net.parameters(),lr=lr,momentum=0.9)
loss_func=nn.CrossEntropyLoss()
loss_val=[]
acc_val=[]

#训练
for i in range(epochs):
    #训练
    qu_net.train()
    
    for j,(img,label) in enumerate(train_loader):

        img=img.to(device)
        label=label.to(device)
        optimizer.zero_grad()
        output=qu_net(img)
        loss=loss_func(output,label)
        loss.backward()
        optimizer.step()

        if(j%37==0):

            logger.info("epoch:{}[{}\{}],train_loss:{:.4f}".format(i,j,train_loader_len,loss.item()))
            #通过sklearn计算acc，precision，recall，f1
            output=torch.argmax(output,dim=1)
            output=output.cpu().numpy()
            label=label.cpu().numpy()
            acc=accuracy_score(label,output)
            pre=precision_score(label,output,average='weighted',zero_division=1)
            rec=recall_score(label,output,average='weighted',zero_division=1)
            f1=f1_score(label,output,average='weighted',zero_division=1)
            logger.info("train====acc:{:.4f},pre:{:.4f},rec:{:.4f},f1:{:.4f}".format(acc,pre,rec,f1))

            loss_val.append(loss.item())
            acc_val.append(acc)
            np.save('result/loss/loss_val.npy',np.array(loss_val))
            np.save('result/acc/acc_val.npy',np.array(acc_val))
            ut.draw_loss(loss_val)
            ut.draw_acc(acc_val)

    
    
    torch.save(qu_net.state_dict(),'result/model/trans_net_{}.pth'.format(i))
    #验证
    # qu_net.eval()
    # with torch.no_grad():
    #     for k,(img,label) in enumerate(val_loader):
    #         img=img.to(device)
    #         label=label.to(device)

    #         pre_data=cla_net(img)
    #         output=qu_net(pre_data)

    #         loss=loss_func(output,label)
    #         logger.critical("epoch:{},val_loss:{:.4f}".format(i,loss.item()))
    #         #通过sklearn计算acc，precision，recall，f1
    #         output=torch.argmax(output,dim=1)
    #         output=output.cpu().numpy()
    #         label=label.cpu().numpy()
    #         acc=accuracy_score(label,output)
    #         pre=precision_score(label,output,average='weighted',zero_division=1)
    #         rec=recall_score(label,output,average='weighted',zero_division=1)
    #         f1=f1_score(label,output,average='weighted',zero_division=1)
    #         #在logger中保留四位小数
    #         logger.critical("val====acc:{:.4f},pre:{:.4f},rec:{:.4f},f1:{:.4f}".format(acc,pre,rec,f1))
        

