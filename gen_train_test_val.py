#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 将一个文件夹下图片按比例分在三个文件夹下
import os
import random
import shutil
from shutil import copy2

datadir_normal = "../dataset_origin"
folderlist = os.listdir(datadir_normal)
# datadir_normal = "./Annotations"
for folder in folderlist:
    inner_path = os.path.join(datadir_normal, folder)
    all_data = os.listdir(inner_path)
    num_all_data = len(all_data)
    print("num_all_data: " + str(num_all_data))
    index_list = list(range(num_all_data))
    # print(index_list)
    random.seed(1)
    random.shuffle(index_list)
    num = 0


    trainDir = "./train"  # （将训练集放在这个文件夹下）
    if not os.path.exists(trainDir):
        os.mkdir(trainDir)


    validDir = "./val"  # （将验证集放在这个文件夹下）
    if not os.path.exists(validDir):
        os.mkdir(validDir)


    testDir = './test'  # （将测试集放在这个文件夹下）
    if not os.path.exists(testDir):
        os.mkdir(testDir)

    for i in index_list:
        fileName = os.path.join(inner_path, all_data[i])
        if num < num_all_data * 0.8:
            # print(str(fileName))
            copy2(fileName, trainDir)
        elif num > num_all_data * 0.8 and num <= num_all_data * 0.9:
            # print(str(fileName))
            copy2(fileName, validDir)
        else:
            copy2(fileName, testDir)
        num += 1
