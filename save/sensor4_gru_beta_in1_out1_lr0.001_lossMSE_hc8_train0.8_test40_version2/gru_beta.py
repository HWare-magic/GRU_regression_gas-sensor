import torch
from torch import nn, optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from torchsummary import summary

class GRU(nn.Module):
    def __init__(self, in_dim, out_dim, timestep_in, timestep_out, num_layers, hidden_layer, device):
        # in_dim=1, out_dim=1, hidden_layer=args.hc=8, timestep_in=70,timestep_out=1,num_layers=4
        super().__init__()
        self.device = device
        # self.rand_wight = torch.randn((1), requires_grad=False).to(self.device) * 100
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layer = hidden_layer  ## 这里应该是hidden_dim
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(in_dim, hidden_layer, self.num_layers, batch_first=True)
        self.fc1 = torch.nn.Linear(timestep_in, timestep_out) ## linner 的输入和输出
        self.fc2 = torch.nn.Linear(hidden_layer, out_dim)
        # self.fc3 = torch.nn.Linear(out_dim+1, out_dim)



    def forward(self, x):
        x_r = x[:,-1,:,:] ## 70 为 timestep_in
        # print(x_r.shape)
        # print(x_r)
        x = x[:,0:70,:,:]
        b,t,n,c = x.shape
        x = x.permute(0,2,1,3)  # b,t,n,c ---> b,n,t,c
        # print('x shape1 is ', x.shape)
        x_r = torch.reshape(x_r, (b,1,c))
        x = torch.reshape(x,(b*n,t,c)) # b,n,t,c ---> b*n,t,c
#         x = x[:,:,:,0].permute(0,2,1) # [B,T,N,C] > [B,N,T]
#         print('x shape is ', x.shape)
        batch = x.shape[0]
        h_0 = torch.randn(size = (self.num_layers, batch, self.hidden_layer)).to(self.device)
        # print('x shape2 is ', x.shape)
        output, h1 = self.gru(x, h_0)
        # print('output shape is : ',output.shape)  # torch.Size([32, 70, 8]) 第一个的batchsize =2 是因为summary函数中默认为2
        # print('h1 shape is : ',h1.shape)  # torch.Size([4, 32, 8])
        out = self.fc1(output.permute(0,2,1)) # torch.Size([32, 8, 1])
        # print('fc1 shape is ', out.shape)
        out = self.fc2(out.permute(0,2,1))  #torch.Size([32, 1, 1])
        out = out - (5 * x_r) # new ver 10  old ver 5  sensor1 = 10  sensor2 =7.5  sensor3 = 2.2 sensor4 = 2.5
        # print('out is : ',out.shape)
        # out = torch.cat((out,x_r),1)
        # out = self.fc3(out.permute(0,2,1))
        out = torch.reshape(out, (b,n,1,1)).permute(0,2,1,3) 
        # print('final out shape is: ',out.shape)  #torch.Size([32,1,1,1])
        return out
    
def main():
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    N_NODE,TIMESTEP_IN,TIMESTEP_OUT,CHANNEL = 1,70,1,1
    model = GRU(in_dim=1, out_dim=1, timestep_in =TIMESTEP_IN, timestep_out=TIMESTEP_OUT, num_layers=4, hidden_layer=16, device=device)
    summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)
    
if __name__ == '__main__':
    main()  
