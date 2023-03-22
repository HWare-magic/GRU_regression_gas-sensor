import argparse ## 作用是可以在命令行传入参数给程序
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import time
import configparser
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import gc
from torchsummary import summary
import importlib.machinery, importlib.util
import lib.Metrics
import lib.Utils
import random

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='gru_beta', help='choose which model to train and test')  ## 这里默认使用的模型是GRU
parser.add_argument('-version', type=int, default=2, help='train version')
parser.add_argument('-instep', type=int, default=1, help='input step')
parser.add_argument('-outstep', type=int, default=1, help='predict step')
parser.add_argument('-sca', type=int, default=0, help='predict step')
parser.add_argument('-hc', type=int, default=8, help='hidden channel')
parser.add_argument('-batch', type=int, default=32, help='batch size')  ## batch size 32
parser.add_argument('-epoch', type=int, default=200, help='training epochs')
parser.add_argument('-mode', type=str, default='eval', help='train, debug or eval')  ## mode 有三种模式 训练 debug 和 评估
parser.add_argument('-data', type=str, default='4',
                    help='choose which ')
parser.add_argument('-train', type=float, default=0.8, help='train data: 0.8,0.7,0.6,0.5')
parser.add_argument('-test', type=str, default='100', help='choose which label to be test dataset')  ## 60 作为预测的类型
parser.add_argument('-scaler', type=str, default='zscore', help='data scaler process type, zscore or minmax') ## 归一化
parser.add_argument('-snorm', type=int, default=1)  # STNorm Hyper Param
parser.add_argument('-tnorm', type=int, default=1)  # STNorm Hyper Param
parser.add_argument('-cuda', type=int, default=0, help='cuda device number')  ## 默认需要把cuda 改为-cuda defaul改为0
args = parser.parse_args()  # python  args.参数名:可以获取传入的参数
# args = parser.parse_args(args=[])    # jupyter notebook
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu")
TIMESTEP_IN = 71  ## 输入的宽度 70 个
TIMESTEP_OUT = 1 ## 输出的宽度 这里是一个
BATCHSIZE =32
SCA = bool(args.sca)  ## 这个参数未知
MODELNAME = args.model
TEST=args.test
if args.data == '4': ## 读了sensor1的数据
    DATAPATH = '../data/with_rate/sensor' + args.data + '_rate'+'.csv'  #
    data = pd.read_csv(DATAPATH,index_col=0)
    DATANAME = 'sensor' + args.data
def get_inputdata(data, test, if_stats=False, if_scaler =False):  ## 这里读了 sensor1的数据 if_stats = True and if_scaler = True 是数据集归一化
    test = test.split(',')
    trainval_data = data[~data['label(ppm)'].isin(list(map(float, test)))].values[:,:,np.newaxis,np.newaxis] # [samples, 70, 1, 1]
    test_data = data[data['label(ppm)'].isin(list(map(float, test)))].values[:,:,np.newaxis,np.newaxis]   # [samples, 70, 1, 1]
    # print(test_data.shape)
    data_index=[i for i in range(trainval_data.shape[0])]
    random.shuffle(data_index)
    train_xy = trainval_data[data_index[:int(0.8*trainval_data.shape[0])]]   ## 取前百分80最为训练集
    val_xy = trainval_data[data_index[int(0.8*trainval_data.shape[0]):]]     ## 后百分20作为验证集
    test_xy = test_data
    if if_stats == True: ## 这里只取了训练集和测试集的均值和方差（有点不够严谨）
        train_mean, train_std = np.mean(trainval_data, axis=(0,1)), np.std(trainval_data, axis=(0,1))
        x_stats = {'mean': train_mean, 'std': train_std}
    if if_stats == False:
        x_stats = None
    seq_data = {'train': train_xy, 'val': val_xy, 'test': test_xy}  ## 这里就输出了seq data 中的三个数据集
    if if_scaler == True:
        for key in seq_data.keys():
            seq_data[key][:, 0:TIMESTEP_IN, :, :] = lib.Utils.z_score(seq_data[key][:, 0:TIMESTEP_IN, :, :],
                                                                      x_stats['mean'], x_stats['std'])
    return seq_data, x_stats ## numpy 数据类型

seq_data, x_stats = get_inputdata(data, TEST, if_stats=SCA, if_scaler =SCA)

def torch_data_loader(device, data, data_type, shuffle=True):
    x = torch.Tensor(data[data_type][:, 0:TIMESTEP_IN, :, :]).to(device)  # [B,T=TIMESTEP_IN,N=sensor_number,C=in_dim = 这里默认为1]
    # x = torch.Tensor(data[data_type][:, TIMESTEP_IN:TIMESTEP_IN+1, :, :1]).to(device)
    if args.model == 'LSTNet':
        y = torch.Tensor(data[data_type][:, TIMESTEP_IN + TIMESTEP_OUT - 1:TIMESTEP_IN + TIMESTEP_OUT, :, :]).to(
            device)  # [B,T=TIMESTEP_OUT,N,C]
    else:
        y = torch.Tensor(data[data_type][:, TIMESTEP_IN:TIMESTEP_IN + TIMESTEP_OUT, :, :]).to(
            device)  # [B,T=TIMESTEP_OUT,N,C]
    data = torch.utils.data.TensorDataset(x, y)
    data_iter = torch.utils.data.DataLoader(data, BATCHSIZE, shuffle=shuffle)
    return data_iter

test_iter = torch_data_loader(device, seq_data, data_type='test', shuffle=False)


def predictModel(model, data_iter):
    YS_truth = []
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            output = model(x)
            YS_pred_batch = output.cpu().numpy()
            YS_truth_batch = y.cpu().numpy()
            YS_pred.append(YS_pred_batch)
            YS_truth.append(YS_truth_batch)
        YS_pred = np.vstack(YS_pred)
        YS_truth = np.vstack(YS_truth)
    return YS_truth, YS_pred  # [B,T,N,C]
def getModel(name, device):
    ### load different baseline model.py  ###
    model_path = '../model/' + args.model + '.py'  # AGCRN.py 的路径
    loader = importlib.machinery.SourceFileLoader('baseline_py_file', model_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    baseline_py_file = importlib.util.module_from_spec(spec)
    loader.exec_module(baseline_py_file)
    ########## select the baseline model ##########
    if args.model == 'gru_beta': #gru_beta
        model = baseline_py_file.GRU(in_dim=1, out_dim=1, hidden_layer=args.hc, timestep_in=70,timestep_out=1,num_layers=4,device=device).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model
model = getModel(MODELNAME, device)
path = r'C:\Users\86136\PycharmProjects\pythonProject\GRU_regression_gas sensor\save\sensor4_gru_in1_out1_lr0.001_lossMSE_hc8_train0.8_test40_version2'
model.load_state_dict(torch.load(path + '/' + 'gru' + '.pt',map_location=device))
YS_truth,YS_pred =predictModel(model,test_iter)

print(YS_pred)
