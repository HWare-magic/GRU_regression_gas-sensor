import numpy as np
import pandas as pd
import math
import torch
import os
import argparse
import re
import matplotlib.pyplot as plt
from scipy.signal import hilbert,chirp
from torch import nn
from d2l import torch as d2l


################# python input parameters #######################
parser = argparse.ArgumentParser()
parser.add_argument('-cycle_data_length', type=int, default=1000, help='choose cycle point')  ##  1000 means  10s
parser.add_argument('-traget_data_length', type=int, default=70, help='choose data length')  ##
parser.add_argument('-pulse_length', type=int, default=30, help='choose pulse data')  ##
parser.add_argument('-pulse_high_volt',type=int,default=2.5,help='heating_volt')
parser.add_argument('-pulse_low_volt',type=int,default=0.55,help='low_volt')
parser.add_argument('-gas',type=str,default='eth',help='gas type')
parser.add_argument('-start_con',type=int,default=1,help='start concentration div 10')  # 10ppm -100ppm
parser.add_argument('-con_num',type=int,default=10,help='number of concentration datatype')
args = parser.parse_args() # python  args.参数名:可以获取传入的参数
################ data prepare ###########
path_name =  r'C:\Users\86136\PycharmProjects\pythonProject\GRU_regression_gas sensor\data\origin data'
os.chdir(path_name)
if not (os.path.exists('process_result')):
    os.mkdir('process_result')
file_chdir = os.getcwd() ##  获取当前路径
file_name_list=[]
file_list=[]
for root,dirs,files in os.walk(file_chdir):  ## file_chdir :代表需要遍历的根文件夹  root :表示正在遍历的文件夹的名字（根/子）
                                            ## dirs :记录正在遍历的文件夹下的子文件夹集合  files:记录正在遍历的文件夹中的文件集合(list形式)
    for file in files:
        if os.path.splitext(file)[-1] =='.csv': ## os.path.splitext()  分离文件名与扩展名；默认返回(fname,fextension)元组 切片后-1 表示后缀
            file_name_list.append(file)
            file_list.append(pd.read_csv(file))
### 数据预处理 加入最后湿度的列名 #######
for num,file in enumerate(file_list):
      file.rename(columns={'Humidity(0-3:0-100%)': 'humidity'},inplace=True)

###参数继承#####
cycle_data_length = args.cycle_data_length
traget_data_length = args.traget_data_length
pulse_length =args.pulse_length
pulse_high_volt = args.pulse_high_volt
pulse_low_volt = args.pulse_low_volt
gas = args.gas
start_con = args.start_con
con_num = args.con_num
######
pre_point = int((traget_data_length - pulse_length)/2)
count =[]
## 计算出不同表格所包含的脉冲数量
for num,file in enumerate(file_list):
    count.append(math.floor(len(file_list[num])/cycle_data_length))
## 创建 2维矩阵  第一维代表 表格个数  第二维代表脉冲定位的点数 （这里是初始化）  长度代表脉冲宽度
z=[]
for i in range(len(count)):
    z.append([0]*count[i])
## 寻找阶跃的点 并且将位置信息存在  z 列表中
for num ,counts in enumerate(count):
    for j in range(counts):
        for i in range(j*cycle_data_length,(j+1)*cycle_data_length):
            if file_list[num].loc[i," pulse "] > pulse_high_volt and file_list[num].loc[(i-1)," pulse "] < pulse_low_volt:
                z[num][j]=file_list[num].index[i]

## 创建目标的dataFrame 用于输出
df = []
for i in range(len(count)):
    df.append(pd.DataFrame(np.zeros(file_list[i].shape), index=file_list[i].index, columns=file_list[i].columns))

## 将脉冲数据 按格式输出
for num, down_l in enumerate(z):
    for lb, down in enumerate(down_l):
        for j in range(traget_data_length):
            df[num].iloc[lb * traget_data_length + j, :] = file_list[num].iloc[down - pre_point + j, :]

## 去掉全为0的行
for i in range(len(count)):
    df[i]=df[i].loc[(df[i]!=0).any(1)]

##将文件按格式保存
for i in range(len(count)):
    df[i].to_csv(f'./process_result/dataset{i+start_con}0ppm_{gas}.csv')  ## save format  is dataset10ppm

for i in range(start_con, len(count) + start_con):

    for j in range(1, 5):  ## 传感器数量
        exec('ppm{}0sensor{}= df[{}].iloc[:,{}].values.reshape(-1,traget_data_length)'.format(i, j, i - start_con,
                                                                                              j))  ## serial 没有 reshape 功能
        exec('ppm{}0sensor{}= pd.DataFrame(ppm{}0sensor{})'.format(i, j, i, j))
        exec('ppm{}0sensor{}.loc[:,traget_data_length+1] = {}0'.format(i, j, i))
        exec('ppm{}0sensor{}.rename(columns={},inplace=True)'.format(i, j, {70: "humidity(v)"}))
        exec('ppm{}0sensor{}.rename(columns={},inplace=True)'.format(i, j, {71: "label(ppm)"}))
        # exec('print(len(ppm{}0sensor{}))'.format(i,j))
        exec(
            'ppm{}0sensor{}=ppm{}0sensor{}.iloc[math.ceil(len(ppm{}0sensor{})/6):math.ceil(5*len(ppm{}0sensor{})/6),:]'.format(
                i, j, i, j, i, j, i, j))  ## 取1/6到 5/6的 数据
        exec('ppm{}0sensor{}.to_csv(\'./process_result/{}0ppms{}_dset.csv\')'.format(i, j, i, j)) ## save format is 10ppm3_dset  3 means senson3
#####   数据拼接过程 并且存入到对应的文件夹中  #######
##### sensor 1 2 3 4
os.chdir(r'C:\Users\86136\PycharmProjects\pythonProject\GRU_regression_gas sensor\data\origin data\process_result')
Folder_Path =r'C:\Users\86136\PycharmProjects\pythonProject\GRU_regression_gas sensor\data\origin data\process_result'
for i in range(1,5):
    if not (os.path.exists('sensor{}'.format(i))):
        exec('os.mkdir(\'sensor{}\')'.format(i))

file_list = os.listdir()

Folder_Path1 ='C:\\Users\86136\\PycharmProjects\\pythonProject\\GRU_regression_gas sensor\\data\\origin data\\process_result\\'
for j in range(1,5):
    exec('snum{} =[]'.format(j))
    SaveFile_Path =  Folder_Path1 +'\\sensor{}'.format(j) + '\\'
    for i in range(len(file_list)):
        pattern= re.compile(r'\d+(ppms{})'.format(j))
        m = pattern.match(file_list[i])
        if m is not None:
            exec('snum{}.append(file_list[i])'.format(j))
    #exec('print(snum{}[0])'.format(j))
    exec('df = pd.read_csv(Folder_Path1 + snum{}[0])'.format(j))
    exec('df.to_csv(SaveFile_Path + \'sensor{}.csv \', encoding="utf_8_sig", index=False)'.format(j))
    for k in range(1,con_num): ## 每一个文件夹中所包含的文件数  如果对应的测试浓度数数不等于它  需要改掉
            exec('df = pd.read_csv(Folder_Path1 +  snum{}[k])'.format(j))
            exec('df.to_csv(SaveFile_Path + \'sensor{}.csv \',encoding="utf_8_sig",index=False,header=False,mode=\'a+\')'.format(j))  ## final output  is sensor1 from 10ppm -100ppm