###  train model is fix random seed seed_torch(213984798)
###  two train model change  use "parser.add_argument('-model', type=str, default='gru', help='choose which model to train and test')"
###  gru  and  gru_beta
###  dataset change by  parser.add_argument('-data', type=str, default='4',
####                    help='choose which ')   1 to 4  means sensor1 to 4



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

def seed_torch(seed=77):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
################# python input parameters #######################
parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='gru', help='choose which model to train and test')  ## 这里默认使用的模型是GRU
parser.add_argument('-version', type=int, default=12, help='train version')
parser.add_argument('-instep', type=int, default=1, help='input step')
parser.add_argument('-outstep', type=int, default=1, help='predict step')
parser.add_argument('-sca', type=int, default=0, help='predict step')
parser.add_argument('-hc', type=int, default=8, help='hidden channel')
parser.add_argument('-batch', type=int, default=32, help='batch size')  ## batch size 32
parser.add_argument('-epoch', type=int, default=200, help='training epochs')
parser.add_argument('-mode', type=str, default='train', help='train, debug or eval')  ## mode 有三种模式 训练 debug 和 评估
parser.add_argument('-data', type=str, default='4',
                    help='choose which ')
parser.add_argument('-train', type=float, default=0.8, help='train data: 0.8,0.7,0.6,0.5')
parser.add_argument('-test', type=str, default='10', help='choose which label to be test dataset')  ## 60 作为预测的类型
# parser.add_argument('-test', type=list, default=[40], help='choose which label to be test dataset')
parser.add_argument('-scaler', type=str, default='zscore', help='data scaler process type, zscore or minmax') ## 归一化
parser.add_argument('-snorm', type=int, default=1)  # STNorm Hyper Param 
parser.add_argument('-tnorm', type=int, default=1)  # STNorm Hyper Param 
parser.add_argument('-cuda', type=int, default=0, help='cuda device number')  ## 默认需要把cuda 改为-cuda defaul改为0
args = parser.parse_args()  # python  args.参数名:可以获取传入的参数
# args = parser.parse_args(args=[])    # jupyter notebook
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu")
################# data selection #######################
if args.data == '4': ## 读了sensor1的数据
    DATAPATH = './data/train_dataset_with_rate/sensor' + args.data + '_rate'+'.csv'  #
    data = pd.read_csv(DATAPATH,index_col=0,header=0)
    DATANAME = 'sensor' + args.data

################# Global Parameters setting #######################   
MODELNAME = args.model  ## 传入GRU
BATCHSIZE = args.batch  ##  32
EPOCH = args.epoch  ## 默认2000
if args.mode=='debug':   ## 如果是debug 模式 那么 epoch = 1
    EPOCH=1
TIMESTEP_IN = 71  ## 输入的宽度 70 个
TIMESTEP_OUT = 1 ## 输出的宽度 这里是一个
SCA = bool(args.sca)  ## 这个参数未知
snorm_bool = bool(args.snorm)  # STNorm Hyper Param
tnorm_bool = bool(args.tnorm)  # STNorm Hyper Param
################# Statistic Parameters from init_config.ini #######################   
ini_config = configparser.ConfigParser()
ini_config.read('./init_config.ini', encoding='UTF-8')  ## 读取一些静态 参数 主要是超参数
common_config = ini_config['common']
# data_config = ini_config[DATANAME]
N_NODE = 1
CHANNEL = int(common_config['CHANNEL'])  # 1
# LEARNING_RATE = float(common_config['LEARNING_RATE'])   # 0.001
LEARNING_RATE = 0.001
# PATIENCE = int(common_config['PATIENCE'])   # 10
PRINT_EPOCH = 1
PATIENCE = 200  ## 耐心值 是啥
OPTIMIZER = 'Adam'#str(common_config['OPTIMIZER'])  # Adam
LOSS = 'MSE'# str(common_config['LOSS'])  # MAE
# TRAIN = float(common_config['TRAIN']) # 0.8
TRAIN = args.train ## 训练数据的比重  这里是0.8
VAL = 1-TRAIN
TEST=args.test
################# random seed setting #######################
# torch.manual_seed(100)
# torch.cuda.manual_seed(100)
# np.random.seed(77)  # for reproducibility
seed_torch(213984798)
torch.backends.cudnn.benchmark = False
################# System Parameter Setting #######################
PATH = "./save/{}_{}_in{}_out{}_lr{}_loss{}_hc{}_train{}_test{}_version".format(DATANAME, args.model, args.instep,
                                                                           args.outstep, LEARNING_RATE, LOSS,args.hc, TRAIN,
                                                                           TEST) ## HC 表示hidden channel 这里默认是8
single_version_PATH = PATH + str(args.version)  ## 单版本的路径
#multi_version_PATH = PATH + '0to4'
import os

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)    ## 限制pytorch 运行的线程数 假如我有4个cpu ,但是只想让Pytorch在1个cpu上运行

# from multiprocessing import cpu_count
#
# cpu_num = cpu_count() # 自动获取最大核心数目
# os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
# os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
# os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
# os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
# os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
# torch.set_num_threads(cpu_num)

##################  data preparation   #############################
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


def getModel(name, device):
    ### load different baseline model.py  ###
    model_path = './model/' + args.model + '.py'  # AGCRN.py 的路径
    loader = importlib.machinery.SourceFileLoader('baseline_py_file', model_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    baseline_py_file = importlib.util.module_from_spec(spec)
    print(baseline_py_file)
    loader.exec_module(baseline_py_file)
    ########## select the baseline model ##########
    if args.model == 'biGRU':
        model = baseline_py_file.GRUModel(input_dim=1, hidden_dim=args.hc, output_dim=1,num_layers=4,
                          dropout=0.1,device=device).to(device)
    if args.model == 'gru': #gru_beta
        model = baseline_py_file.GRU(in_dim=1, out_dim=1, hidden_layer=args.hc, timestep_in=70,timestep_out=1,num_layers=4,device=device).to(device) ## hidden_layer = dim
    if args.model == 'gru_beta': #gru_beta
        model = baseline_py_file.GRU(in_dim=1, out_dim=1, hidden_layer=args.hc, timestep_in=70,timestep_out=1,num_layers=4,device=device).to(device)
    if args.model == 'lstnet':
        model = baseline_py_file.LSTNet(data_m=N_NODE * CHANNEL, window=TIMESTEP_IN, hidRNN=64, hidCNN=64, CNN_kernel=3,
                                        skip=3, highway_window=TIMESTEP_IN).to(device)
    if args.model == 'agcrn':
        model = baseline_py_file.AGCRN(num_nodes=N_NODE, input_dim=CHANNEL, output_dim=CHANNEL,
                                       horizon=TIMESTEP_OUT).to(device)
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


def model_inference(model, val_iter, test_iter, x_stats, save=False, if_scaler=False):
    val_y_truth, val_y_pred = predictModel(model, val_iter)
    if if_scaler == True: 
        val_y_pred = lib.Utils.z_inverse(val_y_pred, x_stats['mean'], x_stats['std'])
    val_mse, val_rmse, val_mae, val_mape = lib.Metrics.evaluate(val_y_pred, val_y_truth)  # [T]
    test_y_truth, test_y_pred = predictModel(model, test_iter)
    if if_scaler == True: 
        test_y_pred = lib.Utils.z_inverse(test_y_pred, x_stats['mean'], x_stats['std'])
    test_mse, test_rmse, test_mae, test_mape = lib.Metrics.evaluate(test_y_pred, test_y_truth)  # [T]
    if save == True:
        return test_y_truth, test_y_pred
    else:
        return val_mse, val_rmse, val_mae, val_mape, test_mse, test_rmse, test_mae, test_mape


def trainModel(name, device, data, x_stats, if_stats=False, if_scaler =False):   ## data = seq_data
    mode = 'Train'
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name, device) 
    summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)     # 第一个初始化是的b= 2
    print('parameters_count:', count_parameters(model))
    train_iter = torch_data_loader(device, data, data_type='train', shuffle=True)
    val_iter = torch_data_loader(device, data, data_type='val', shuffle=True)
    test_iter = torch_data_loader(device, data, data_type='test', shuffle=False)
    if if_scaler:
        torch_mean = torch.Tensor(x_stats['mean'].reshape((1, 1, -1, 1))).to(device)
        torch_std = torch.Tensor(x_stats['std'].reshape((1, 1, -1, 1))).to(device)
    min_test_loss = np.inf
    wait = 0
    #     LOSS = "MSE"
    print('LOSS is :', LOSS)
    if LOSS == "masked_mae":
        criterion = lib.Utils.masked_mae
    if LOSS == "masked_mse":
        criterion = lib.Utils.masked_mse
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCH):  # EPOCH
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            if if_scaler:
                y_pred = y_pred * torch_std + torch_mean
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        if epoch % PRINT_EPOCH == 0:
            train_loss = loss_sum / n
            val_mse, val_rmse, val_mae, val_mape, test_mse, test_rmse, test_mae, test_mape = model_inference(model,
                                                                                                             val_iter,
                                                                                                             test_iter,
                                                                                                             x_stats,
                                                                                                             save=False,
                                                                                                             if_scaler=False)
            print(f'Epoch {epoch}: ')
            print("|%01d    Horizon | MAPE: %.6f, %.6f; MAE: %.6f, %.6f; RMSE: %.6f, %.6f;" % (
            TIMESTEP_OUT, val_mape, test_mape, val_mae, test_mae, val_rmse, test_rmse))

            if np.mean(val_rmse) < min_test_loss:
                best_epoch = epoch
                wait = 0
                min_test_loss = np.mean(val_rmse)
                torch.save(model.state_dict(), single_version_PATH + '/' + name + '.pt')  ## 这里是保存 模型 看下怎么改个格式  或者 怎么直接用起来
            else:
                wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time, " seconds ", "train loss:", np.around(train_loss, 3),
                  "val rmse:", np.around(np.mean(val_rmse), 3), "test rmse:", np.around(np.mean(test_rmse), 3))
        with open(single_version_PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.3f, %s, %.3f, %s, %.3f\n" % (
            "epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation rmse:",
            np.mean(val_rmse), "test rmse:", np.mean(test_rmse)))

    model.load_state_dict(torch.load(single_version_PATH + '/' + name + '.pt'))
    val_mse, val_rmse, val_mae, val_mape, test_mse, test_rmse, test_mae, test_mape = model_inference(model, val_iter,
                                                                                                     test_iter, x_stats,
                                                                                                     save=False,if_scaler=False)
    print('Model ', name, ' Best Results:')
    print(f'Best Epoch {best_epoch}: ')
    print("|%01d    Horizon | MAPE: %.6f, %.6f; MAE: %.6f, %.6f; RMSE: %.6f, %.6f;" % (
    TIMESTEP_OUT, val_mape, test_mape, val_mae, test_mae, val_rmse, test_rmse))
    print('Model Training Ended ...', time.ctime())


def multi_version_test(name, device, train, versions,if_stats=False,if_scaler=False):
    mode = 'multi version test'
    print('Model Testing Started ...', time.ctime())
    print('INPUT_STEP, PRED_STEP', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name, device)
    mse_all, rmse_all, mae_all, mape_all = np.zeros((len(train), len(versions), TIMESTEP_OUT)), np.zeros(
        (len(train), len(versions), TIMESTEP_OUT)), np.zeros((len(train), len(versions), TIMESTEP_OUT)), np.zeros(
        (len(train), len(versions), TIMESTEP_OUT))  # [V,T]
    #     val_iter = torch_data_loader(device, data, data_type='val', shuffle=True)
    #     test_iter = torch_data_loader(device, data, data_type='test', shuffle=False)
    for train_ind, tr in enumerate(train):
        which_train = tr
        data, x_stats = get_inputdata(data, TEST, if_stats=if_stats, if_scaler =if_scaler)
        val_iter = torch_data_loader(device, data, data_type='val', shuffle=True)
        test_iter = torch_data_loader(device, data, data_type='test', shuffle=False)
        print('*' * 40)
        print('*' * 40)
        print('Under Train Strategy --- ', tr, ' ---:')
        for v_ in versions:
            print('--- version ', v_, ' evaluation start ---')
            multi_test_PATH = "./save/{}_new{}_in{}_out{}_lr{}_hc{}_train{}_test{}_version{}/{}.pt".format(DATANAME,
                                                                                                          args.model,
                                                                                                          args.instep,
                                                                                                          args.outstep,
                                                                                                          LEARNING_RATE,
                                                                                                          args.hc,
                                                                                                          which_train,
                                                                                                          TEST, v_,
                                                                                                          args.model)
            if os.path.isfile(multi_test_PATH):
                model.load_state_dict(torch.load(multi_test_PATH, map_location=device))
                print("file exists")
            else:
                print("file not exist")
                break
            print('*' * 20)
            print(f'Version: {v_} Start Testing :')
            val_mse, val_rmse, val_mae, val_mape, test_mse, test_rmse, test_mae, test_mape = model_inference(model,
                                                                                                             val_iter,
                                                                                                             test_iter,
                                                                                                             x_stats,
                                                                                                             save=False,
                                                                                                             if_scaler=False) # [T]
            mse_all[train_ind, v_], rmse_all[train_ind, v_], mae_all[train_ind, v_], mape_all[
                train_ind, v_] = test_mse, test_rmse, test_mae, test_mape
            print("|%01d    Horizon | MAPE: %.6f, %.6f; MAE: %.6f, %.6f; RMSE: %.6f, %.6f;" % (
                    TIMESTEP_OUT, val_mape, test_mape, val_mae, test_mae, val_rmse, test_rmse))
            print('--- version ', v_, ' evaluation end ---')
            print('')
    #     np.save(multi_version_PATH + '/' + MODELNAME + '_groundtruth.npy', y_truth)  # [V,samples,T,N]
    #     np.save(multi_version_PATH + '/' + MODELNAME + '_prediction.npy', y_pred)
    mse = np.mean(mse_all, axis=(0, 1))  # [train, V, T]  -> [T]  np.mean(mse_all, axis=(0,1))
    rmse = np.mean(rmse_all, axis=(0, 1))
    mae = np.mean(mae_all, axis=(0, 1))
    mape = np.mean(mape_all, axis=(0, 1))
    print('*' * 40)
    print('*' * 40)
    print('*' * 40)
    print('Results in Test Dataset in Each Horizon with All Version Average:')
    print(args.model, ' :')

    print("|%01d    | MAPE: %.6f; MAE: %.6f; RMSE: %.6f;" % (TIMESTEP_OUT, mape[0], mae[0], rmse[0]))

    print('Model Multi Version Testing Ended ...', time.ctime())
    print("*" * 40)
    print("*" * 40)
#     print("Overleaf Format in table 2 ----  Version Average in test dataset: ")
#     if name == 'LSTNet':
#         print(" &   MAPE   &   MAE  &   RMSE ")
#         #     GWN\_8\_1\_1 & 56.755 & 30.298 & 20.360\% & 82.327 & 40.783 & 27.326\% & 114.689 & 54.537 & 36.511\%
#         if args.data == 'exchangerate':
#             print(" & {:.6f}\% & {:.6f} & {:.6f} "
#                   .format(mape[0], mae[0], rmse[0]))
#         else:
#             print(" & {:.3f}\% & {:.3f} & {:.3f} "
#                   .format(mape[0], mae[0], rmse[0]))
#     else:
#         print(" &   MAPE   &   MAE  &   RMSE &   MAPE   &   MAE  &   RMSE &   MAPE   &   MAE  &  RMSE  &")
#         #     GWN\_8\_1\_1 & 56.755 & 30.298 & 20.360\% & 82.327 & 40.783 & 27.326\% & 114.689 & 54.537 & 36.511\%
#         if args.data == 'exchangerate':
#             print(" & {:.6f}\% & {:.6f} & {:.6f} & {:.6f}\% & {:.6f} & {:.6f} & {:.6f}\% & {:.6f} & {:.6f} &  \\\\"
#                   .format(mape[0], mae[0], rmse[0], mape[1], mae[1], rmse[1], mape[2], mae[2], rmse[2]))
#         else:
#             print(" & {:.3f}\% & {:.3f} & {:.3f} & {:.3f}\% & {:.3f} & {:.3f} & {:.3f}\% & {:.3f} & {:.3f} &  \\\\"
#                   .format(mape[0], mae[0], rmse[0], mape[1], mae[1], rmse[1], mape[2], mae[2], rmse[2]))


def main():
#     if not os.path.exists(multi_version_PATH):
#         os.makedirs(multi_version_PATH)

#     shutil.copy2(model_path, multi_version_PATH)

    if args.mode == 'train' or "debug":  # train and test in single version
        if not os.path.exists(single_version_PATH):
            os.makedirs(single_version_PATH)       
        model_path = './model/' + args.model + '.py'    
        shutil.copy2(model_path, single_version_PATH)      ## 把算法复制到其中模型和结果输出的文件夹下
        print(single_version_PATH, 'training started', time.ctime())
        seq_data, x_stats = get_inputdata(data, TEST, if_stats=SCA, if_scaler =SCA)
        for key in seq_data.keys():
            print(key, ' : ', seq_data[key].shape)  ##打印出每个数据集的含量
        trainModel(MODELNAME, device, seq_data, x_stats, if_stats=SCA, if_scaler =SCA)
    if args.mode == 'eval0':  # eval in multi version
        print(single_version_PATH, 'single version ', args.version, ' testing started', time.ctime())
        seq_data, x_stats = get_inputdata(data, TEST, if_stats=SCA, if_scaler=SCA)
        # predictModel(MODELNAME,data_iter=seq_data)
    #    multi_version_test(MODELNAME, device, train=[args.train], versions=args.version,if_stats=SCA, if_scaler =SCA)
    # if args.mode == 'eval':  # eval in multi version
    #     if not os.path.exists(multi_version_PATH):
    #         os.makedirs(multi_version_PATH)
    #     print(multi_version_PATH, 'multi version testing started', time.ctime())
    #     multi_version_test(MODELNAME, device, train=[args.train], versions=np.arange(0, 10),if_stats=False, if_scaler =False)  #
    # if args.mode == 'all':
    #     if not os.path.exists(multi_version_PATH):
    #         os.makedirs(multi_version_PATH)
    #     multi_version_test(MODELNAME, device, train=[0.5, 0.6, 0.7, 0.8], versions=np.arange(0, 5),if_stats=False, if_scaler =False) #


if __name__ == '__main__':
    main()

