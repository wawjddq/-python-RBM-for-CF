# -*- coding: utf-8 -*-
"""
Created on Thu May 12 21:53:16 2016

@author:ddq

"""

import numpy as np
import os

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def load_data(path):
    data_dir, data_file = os.path.split(path)
    if data_dir == "" and not os.path.isfile(path):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            path
        )
        if os.path.isfile(new_path) or data_file == 'e:\u_data1.txt':
            path = new_path
    if (not os.path.isfile(path)) and data_file == 'e:\u_data1.txt':
        print("The file %s is not exist" % data_file)
    print('... loading data') 
    dataset=np.zeros([943,1682])
    dataset=np.zeros([10,200])
    for readline in open(path):
        userId, ItemId, rate, timeStamp = readline.strip().split('\t')
        u_id = int(userId)-1
        i_id = int(ItemId)-1
        dataset[u_id][i_id] = int(rate)
    m, n = dataset.shape
    return m,n,dataset


def splitData(dataSet):
    m, n = dataSet.shape     #获得数据集的行，列
    train_num = int(round(m * 0.8))     #训练集个数占总数据集的80%
    test_num = m - train_num
    train_set = np.zeros((train_num,n))
    test_set = np.zeros((test_num, n))
    #print train_num 
    #print test_num
    for i in range(train_num):
        for j in range(n):
            train_set[i][j] = dataSet[i][j]
    cursor = train_num-1
    for i in range(test_num):
        cursor += 1
        for j in range(n):
            test_set[i][j] = dataSet[cursor][j]
    return train_set, test_set      
        
        
    
    
    
    


    
    