# Ref: Hyland, S.L. and et al. "REAL-VALUED (MEDICAL) TIME SERIES GENERA- TION WITH RECURRENT CONDITIONAL GANS,"
# arXiv:1706.02633v2 [stat.ML] 4 Dec 2017

import numpy as np
import pandas as pd
import re
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
from torch import optim

class RGAN_G(nn.Module):
    # Generator
    def __init__(self, seq_len, noise_sz, hidden_sz, num_layer, out_sz):
        super(RGAN_G, self).__init__()
        self.seq_len = seq_len
        self.noise_sz = noise_sz
        self.hidden_sz = hidden_sz
        self.num_layer = num_layer
        self.out_sz = out_sz
        
        self.lstm = nn.LSTM(self.noise_sz, self.hidden_sz, self.num_layer)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.hidden_sz, self.out_sz)

        # Initialize weights.
        nn.init.xavier_normal_(self.linear.weight)
        
    def forward(self, z, reshape=True):
        if reshape:
            z = z.view(self.seq_len, -1, self.noise_sz)
        
        out, _ = self.lstm(z)
        out = self.dropout(out)
        # pass on the entirety of lstm out to next layer if it is a seq2seq prediction
        out = self.linear(out)
        return out
    
class RGAN_D(nn.Module):
    # Discriminator
    def __init__(self, seq_len, in_sz, hidden_sz, num_layer):
        super(RGAN_D, self).__init__()
        self.seq_len = seq_len
        self.in_sz = in_sz
        self.hidden_sz = hidden_sz
        self.num_layer = num_layer

        self.lstm = nn.LSTM(self.in_sz, self.hidden_sz, self.num_layer)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.hidden_sz*self.seq_len, 1)
        self.sig = nn.Sigmoid()

        # Initialize weights.
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        # normal orientation
        out = out.transpose(0, 1).view(-1, self.seq_len, self.hidden_sz)
        out = self.linear(out.reshape(-1, self.seq_len*self.hidden_sz))
        out = self.sig(out)
        return out

def format_data(D, seq_len):
    '''Return dim: (n_sample, long+lat)'''
    M = []
    for i in range(len(D)):
        gps_lst = re.sub(r"[[|[|]|]|]]", "", str(D["POLYLINE"][i])).split(",")
        if len(gps_lst) == seq_len*2:
            num_lst = [float(i) for i in gps_lst]
            long = num_lst[::2]
            lat = num_lst[1::2]
            M.append(long+lat)
        else:
            continue
    df = pd.DataFrame(M)
    return df

def init_rgan(seq_len, noise_sz, in_sz, out_sz, hidden_sz, num_layer, device):
    # Initialize RGAN
    G = RGAN_G(seq_len, noise_sz, hidden_sz, num_layer, out_sz)
    D = RGAN_D(seq_len, in_sz, hidden_sz, num_layer)
    G.to(device)
    D.to(device)
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(),lr=0.001,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)
    optimizer_D = optim.Adam(D.parameters(),lr=0.001,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)
    return G, D, criterion, optimizer_G, optimizer_D

def prepare_rgan_data(D, seq_len, test_sz, batch_size):
    data = format_data(D, seq_len) # n * (long+lat)
    scaler = StandardScaler() # zero mean, unit variance
    trans_data = scaler.fit_transform(data)
    tr_d, te_d = train_test_split(trans_data, test_size=test_sz, random_state=42)
    train = torch.from_numpy(tr_d).float()
    test = torch.from_numpy(te_d).float()
    train = torch.utils.data.TensorDataset(train)
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=10)
    return train_loader, test, scaler
    
def train_rgan(epochs, train_loader, seq_len, noise_sz, G, D, criterion, optimizer_G, optimizer_D, device):
    G_losses = []
    D_losses = []
    real_label = 1
    fake_label = 0

    for epoch in range(epochs):
        print('epoch: '+str(epoch))
        for i, x in enumerate(train_loader):
            x = autograd.Variable(x[0].view(-1, seq_len, noise_sz))
            
            # train with real on D
            label_r = torch.full((len(x), 1), real_label).to(device)
            out_real = D(x.transpose(0, 1).view(seq_len, -1, noise_sz).to(device)) # transpose for lstm input
            errD_real = criterion(out_real, label_r)
            D.zero_grad()
            errD_real.backward()
            
            # train with fake on D
            noise = torch.randn(len(x), seq_len*noise_sz).to(device)
            fake = G(noise)
            label_f = torch.full((len(x), 1), fake_label).to(device)
            out_fake = D(fake.detach().to(device))
            errD_fake = criterion(out_fake, label_f)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizer_D.step()
            
            # update G
            output = D(fake.to(device))
            errG = criterion(output, label_r)
            G.zero_grad()
            errG.backward()
            optimizer_G.step()
        G_losses.append(errG.item())
        D_losses.append(errD.item())
    return G, D, G_losses, D_losses

def tgan_format_data(D, seq_len):
    # return (seq_len, 2) per sample
    M = []
    for i in range(len(D)):
        gps_lst = re.sub(r"[[|[|]|]|]]", "", str(D["POLYLINE"][i])).split(",")
        if len(gps_lst) == seq_len*2:
            num_lst = [float(i) for i in gps_lst]
            long = num_lst[::2]
            lat = num_lst[1::2]
            M.append(np.column_stack((long, lat)))
        else:
            continue
    return M