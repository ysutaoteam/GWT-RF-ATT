#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
#忽略警告
warnings.filterwarnings("ignore")
# 使用自带的样式进行美化
plt.style.use('fivethirtyeight')

from sklearn.model_selection import train_test_split
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


# parkinson_x = pd.read_csv(r"距离矩阵.csv",engine='python')
parkinson_x = pd.read_csv(r"距离矩阵_max.csv",engine='python')
columns = ['Unnamed: 0']
parkinson_x = parkinson_x.drop(columns, axis=1)
dataset = parkinson_x.values
corr = dataset.astype('float32')
print(corr.shape)

cluster=r"b00.csv"
r2=[]
init=0

# ！！！特征之间的最小值连在一起，比较平滑，小波系数比较稀疏
# set=[]
# set.append(init)
#
# idx = np.argmin(corr[init,:], axis=0)  # 最小值的索引
# pro = np.amin(corr[init,:], axis=0)    # 最小值是多少
#
# set.append(idx)
#
# for i in range(16):
#     idx = np.argmin(corr[idx, :], axis=0)
# while idx in set:
#     id = set[-1]
#     corr[id, idx]=100
#     a=corr[id, :]
#     idx = np.argmin(a, axis=0)
# set.append(idx)
# print(set)

# ！！！特征之间的最大值连在一起，比较波折
set = []
set.append(init)

idx = np.argmax(corr[init, :], axis=0)  # 最大值的索引
pro = np.amax(corr[init, :], axis=0)  # 最大值是多少

set.append(idx)

for i in range(16):
    idx = np.argmax(corr[idx, :], axis=0)
    while idx in set:
        id = set[-1]
        corr[id, idx] = -5
        a = corr[id, :]
        idx = np.argmax(a, axis=0)
    set.append(idx)
print(set)


parkinson = pd.read_csv(cluster)
columns = ['Unnamed: 0']
parkinson = parkinson.drop(columns, axis=1)
set1=[]
for i in set:
    set1.append(''+str(i)+'')

parkinson=parkinson[set1]
# print(parkinson)
parkinson.to_csv("最长路径排序后的parks.csv")


a00 = r"最长路径排序后的parks.csv"
y00 = r'y00.csv'

parkinson_x = pd.read_csv(a00)
parkinson_dataset_y = pd.read_csv(y00)


# parkinson_dataset = parkinson_dataset.dropna()
columns = ['Unnamed: 0']
parkinson_x = parkinson_x.drop(columns, axis=1)

columns = ['0']
parkinson_y = pd.DataFrame(parkinson_dataset_y, columns=columns)

dataset = parkinson_x.values
parkinson_x = dataset.astype('float32')

dataset = parkinson_y.values
parkinson_y = dataset.astype('float32')

x, test_x, y, test_y =\
    train_test_split(parkinson_x, parkinson_y,test_size=0.1, random_state=67,shuffle=True) #, random_state=5


from sklearn.preprocessing import StandardScaler,MinMaxScaler
import keras
scale=MinMaxScaler()
trainX = scale.fit_transform(x)
testX = scale.transform(test_x)

scale1 = MinMaxScaler()
trainY = scale1.fit_transform(y)
test_y = scale1.transform(test_y)


import pywt

# 2层分解
def wavelet(trainX):
    train=[]
    for i in range(trainX.shape[0]):
        train1 = trainX[i,:]

        coeffs = pywt.wavedec(train1, 'db2', level=2)
        length = len(coeffs)

        coeffs = np.array(coeffs)

        fenliang=[]
        for i in range(length):
            fenliang.extend(coeffs[i])

        fenliang=np.array(fenliang).reshape((1,-1))

        train.append(fenliang)
    train=np.array(train)
    return train

trainX = wavelet(trainX).reshape((trainX.shape[0], -1))
# print(trainX.shape)

testX =wavelet(testX).reshape((testX.shape[0],-1))

num_tree=30

from sklearn import datasets,ensemble
rf_model = ensemble.RandomForestRegressor(n_estimators=num_tree,max_depth=50,criterion='mse')
rf_model.fit(trainX,trainY)

pre = rf_model.predict(testX)
pre =pre.reshape((-1,1))

pre_train = np.array([tree.predict(trainX) for tree in rf_model.estimators_])
print("pre_train:",pre_train.shape)
pre_test = np.array([tree.predict(testX) for tree in rf_model.estimators_])
print("pre_test:",pre_test.shape)

pre_train = pre_train.T
print("pre_train:",pre_train.shape)
pre_test = pre_test.T
print("pre_test:",pre_test.shape)

trainYY=pd.DataFrame(pre_train)
trainYY.to_csv("最长train_tree_pre.csv")
trainYY=pd.DataFrame(pre_test)
trainYY.to_csv("最长test_tree_pre.csv")






