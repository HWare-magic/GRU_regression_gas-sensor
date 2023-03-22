import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout,device):
        super().__init__()
        self.device = device
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x[:, 0:70, :, :]
        b, t, n, c = x.shape
        x = x.permute(0, 2, 1, 3)  # b,t,n,c ---> b,n,t,c
        # print('x shape1 is ', x.shape)
        #x_r = torch.reshape(x_r, (b, 1, c))
        x = torch.reshape(x, (b * n, t, c))  # b,n,t,c ---> b*n,t,c
        #         x = x[:,:,:,0].permute(0,2,1) # [B,T,N,C] > [B,N,T]
        #         print('x shape is ', x.shape)
        batch = x.shape[0]
        h_0 = torch.randn(size=(self.num_layers, batch, self.hidden_layer)).to(self.device)
        # x: (seq_len, batch_size, input_dim)
        output, hidden = self.gru(x,h_0)
        # output: (seq_len, batch_size, hidden_dim*2)
        # hidden: (num_layers*2, batch_size, hidden_dim)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden: (batch_size, hidden_dim*2)
        output = self.fc(hidden)
        # output: (batch_size, output_dim)
        return output