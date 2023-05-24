import sys
import pynvml
import os
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

################# python input parameters #######################
parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='gru', help='choose which model to train and test')  ## 这里默认使用的模型是GRU
parser.add_argument('-version', type=int, default=2, help='train version')
parser.add_argument('-instep', type=int, default=1, help='input step')
parser.add_argument('-outstep', type=int, default=1, help='predict step')
parser.add_argument('-sca', type=int, default=0, help='predict step')
parser.add_argument('-hc', type=int, default=8, help='hidden channel')
parser.add_argument('-batch', type=int, default=32, help='batch size')  ## batch size 32
parser.add_argument('-epoch', type=int, default=2000, help='training epochs')
parser.add_argument('-mode', type=str, default='eval', help='train, debug or eval')  ## mode 有三种模式 训练 debug 和 评估
parser.add_argument('-data', type=str, default='2',
                    help='choose which ')
parser.add_argument('-train', type=float, default=0.8, help='train data: 0.8,0.7,0.6,0.5')
parser.add_argument('-test', type=str, default='40', help='choose which label to be test dataset')  ## 60 作为预测的类型
parser.add_argument('-scaler', type=str, default='zscore', help='data scaler process type, zscore or minmax') ## 归一化
parser.add_argument('-snorm', type=int, default=1)  # STNorm Hyper Param
parser.add_argument('-tnorm', type=int, default=1)  # STNorm Hyper Param
parser.add_argument('-cuda', type=int, default=0, help='cuda device number')  ## 默认需要把cuda 改为-cuda defaul改为0
args = parser.parse_args()  # python  args.参数名:可以获取传入的参数
# args = parser.parse_args(args=[])    # jupyter notebook
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu")



def getModel(name, device):
    ### load different baseline model.py  ###
    model_path = '../model/' + args.model + '.py'  # AGCRN.py 的路径
    loader = importlib.machinery.SourceFileLoader('baseline_py_file', model_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    baseline_py_file = importlib.util.module_from_spec(spec)
    loader.exec_module(baseline_py_file)
    ########## select the baseline model ##########
    if args.model == 'gru_beta':
        model = baseline_py_file.GRU(in_dim=1, out_dim=1, hidden_layer=args.hc, timestep_in=70,timestep_out=1,num_layers=4,device=device).to(device)
    if args.model == 'gru': #gru_beta
        model = baseline_py_file.GRU(in_dim=1, out_dim=1, hidden_layer=args.hc, timestep_in=70,timestep_out=1,num_layers=4,device=device).to(device)
    if args.model == 'lstnet':
        model = baseline_py_file.LSTNet(data_m=N_NODE * CHANNEL, window=TIMESTEP_IN, hidRNN=64, hidCNN=64, CNN_kernel=3,
                                        skip=3, highway_window=TIMESTEP_IN).to(device)
    if args.model == 'agcrn':
        model = baseline_py_file.AGCRN(num_nodes=N_NODE, input_dim=CHANNEL, output_dim=CHANNEL,
                                       horizon=TIMESTEP_OUT).to(device)
    if args.model == 'wavenet':
        model = baseline_py_file.wavenet(device, num_nodes=1, in_dim=1, out_dim=1,supports=None).to(device)

    if args.model == 'gwn':
        ADJPATH = './data/METRLA/adj_mx.pkl'
        ADJTYPE = 'doubletransition'
        adj_mx = baseline_py_file.load_adj(ADJPATH, ADJTYPE)
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        model = baseline_py_file.GWN(device, num_nodes=N_NODE, in_dim=CHANNEL, out_dim=TIMESTEP_OUT, supports=None,
                                     kernel_size=2, blocks=1, layers=4).to(
            device)  # if support is None, random initial adj to train
    if args.model == 'mtgnn':
        subgraph_size_data = 4
        model = baseline_py_file.gtnet(gcn_true=True, buildA_true=True, gcn_depth=2, num_nodes=N_NODE, device=device,
                                       predefined_A=None, static_feat=None, dropout=0, subgraph_size=subgraph_size_data,
                                       node_dim=2, dilation_exponential=1, conv_channels=32, residual_channels=32,
                                       skip_channels=32, end_channels=32, seq_length=TIMESTEP_IN, in_dim=CHANNEL,
                                       out_dim=TIMESTEP_OUT, layers=1, propalpha=0.05, tanhalpha=3,
                                       layer_norm_affline=True).to(device)
    if args.model == 'STNorm':
        model = baseline_py_file.Wavenet(device, N_NODE, tnorm_bool=tnorm_bool, snorm_bool=snorm_bool, in_dim=1,
                                         out_dim=TIMESTEP_OUT, channels=args.hiddenchannel, kernel_size=2, blocks=1,
                                         layers=4).to(device)
    ###############################################
    ### initial the model parameters ###
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model

MODELNAME = args.model  ## 传入GRU
TIMESTEP_IN = args.instep  ## 输入的宽度 70 个
TIMESTEP_OUT = args.outstep ## 输出的宽度 这里是一个

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

path = '../save/sensor4_gru_in1_out1_lr0.001_lossMSE_hc8_train0.8_test10_version10'
model = getModel(MODELNAME, device)
# model = baseline_py_file.GRU(in_dim=TIMESTEP_IN, out_dim=TIMESTEP_OUT, hidden_layer=args.hc, device=device).to(device)
model.eval()
model.load_state_dict(torch.load(path + '/' + 'gru' + '.pt',map_location=device))  ## 模型的cuda 要和本地的显卡匹配起来
# map_location=torch.device('cpu')
print(type(model))
result = []
data_path = '../data/test data/s4test40.csv'
data_test = pd.read_csv(data_path,header=None,index_col=None)
data = np.array(data_test)
tor_data = torch.Tensor(data).unsqueeze(-1)
tt_data = tor_data.unsqueeze(-1).to(device)
print(tt_data)
y = model(tt_data)
print(y)
np.savetxt('sensor4_reg.csv',y.cpu().detach().numpy().reshape(-1,1),fmt='%.2f',delimiter=',')
# np.save(y.cpu.detach.numpy())
# for i in range(len(data_test)):
#     data = np.array(data_test)[i]
#     print(data)
#     # tor_data = torch.Tensor(data).unsqueeze(-1)
#     # tt_data = tor_data.unsqueeze(-1).to(device)
#     # print(tt_data)
#     # y = model(tt_data)
#     # result.append(y.tolist)
# print(result)
# summary(mode, (TIMESTEP_IN, 1, 1), device=device)

# lz = torch.load(path + '/' + 'wavenet' + '.pt',map_location=torch.device('cpu'))
# print(lz.keys())
# for key,value in lz["state_dict"].items():
#     print(key,value.size(),sep='')
# for parameter in lz.parameters():
#     print(parameter)