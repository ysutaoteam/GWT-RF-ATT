#!/usr/bin/python3
# -*- coding:utf-8 -*-
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#忽略警告
warnings.filterwarnings("ignore")
# 使用自带的样式进行美化
plt.style.use('fivethirtyeight')
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

a00 = r"b00.csv"

parkinson_x = pd.read_csv(a00)
columns = ['Unnamed: 0']
parkinson_x = parkinson_x.drop(columns, axis=1)
dataset = parkinson_x.values
parkinson_x = dataset.astype('float32')

from sklearn.preprocessing import StandardScaler,MinMaxScaler
import keras
scale=MinMaxScaler()
trainX = scale.fit_transform(parkinson_x)
print(trainX.shape)

def embedding_distance(feature_1, feature_2):
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist

m=[]
for i in range(18):
    for j in range(18):
        if i==j:
            juli=0
            m.extend([juli])
        else:
            juli=embedding_distance(trainX[:,i],trainX[:,j])
            print(juli)
            m.extend([juli])
m=np.array(m).reshape((18,18))
print(m.shape)

m=pd.DataFrame(m)
m.to_csv("距离矩阵max.csv")
