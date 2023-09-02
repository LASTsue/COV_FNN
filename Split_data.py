import pandas as pd

cov_data = pd.read_csv('datasets/COVID-19/COVID.csv')
nor_data = pd.read_csv('datasets/COVID-19/Normal.csv')
op_data=pd.read_csv('datasets/COVID-19/Lung_Opacity.csv')
pne_data=pd.read_csv('datasets/COVID-19/Viral Pneumonia.csv')


cov_data['LABEL']=0
nor_data['LABEL']=1
op_data['LABEL']=2
pne_data['LABEL']=3
cov_data=cov_data.loc[:,['FILE NAME','LABEL']]
nor_data=nor_data.loc[:,['FILE NAME','LABEL']]
op_data=op_data.loc[:,['FILE NAME','LABEL']]
pne_data=pne_data.loc[:,['FILE NAME','LABEL']]

#遍历nor_data,将文件名改为首字母大写
for i in range(len(nor_data)):
    nor_data.loc[i,'FILE NAME']=nor_data.loc[i,'FILE NAME'].capitalize()

#将数据拆分成训练集，验证集，测试集
#训练集
train_cov=cov_data.iloc[:int(0.7*len(cov_data)),:]
train_nor=nor_data.iloc[:int(0.7*len(nor_data)),:]
train_op=op_data.iloc[:int(0.7*len(op_data)),:]
train_pne=pne_data.iloc[:int(0.7*len(pne_data)),:]
train=pd.concat([train_cov,train_nor,train_op,train_pne],axis=0)
train=train.sample(frac=1).reset_index(drop=True)
train.to_csv('datasets/COVID-19/train.csv',index=False)

#验证集
val_cov=cov_data.iloc[int(0.7*len(cov_data)):int(0.85*len(cov_data)),:]
val_nor=nor_data.iloc[int(0.7*len(nor_data)):int(0.85*len(nor_data)),:]
val_op=op_data.iloc[int(0.7*len(op_data)):int(0.85*len(op_data)),:]
val_pne=pne_data.iloc[int(0.7*len(pne_data)):int(0.85*len(pne_data)),:]
val=pd.concat([val_cov,val_nor,val_op,val_pne],axis=0)
val=val.sample(frac=1).reset_index(drop=True)
val.to_csv('datasets/COVID-19/val.csv',index=False)

#测试集
test_cov=cov_data.iloc[int(0.85*len(cov_data)):,:]
test_nor=nor_data.iloc[int(0.85*len(nor_data)):,:]
test_op=op_data.iloc[int(0.85*len(op_data)):,:]
test_pne=pne_data.iloc[int(0.85*len(pne_data)):,:]
test=pd.concat([test_cov,test_nor,test_op,test_pne],axis=0)
test=test.sample(frac=1).reset_index(drop=True)
test.to_csv('datasets/COVID-19/test.csv',index=False)
