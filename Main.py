from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from CTGAN_utils import *
from RGAN import *
from timegan import timegan
from Metrics import *
from Plots import *
import argparse
import ast

def main (args):
    '''Args:
       -model: CTGAN, RGAN, or TimeGAN
       -cate: taxi call types, A, B or C
       -seq_len: sequence length
       -epochs: iteration of training (for CTGAN epochs refer to the number of generated samples)
       -parameter: dictionary with corresponding hyperparameters
       -plot: whether to plot trajectories on the map or not
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # replace with your path to read the file
    D = pd.read_csv('~/work/Taxi_Trajectories/train_taxi_trajectory.csv')
    if args.cate == 'A':
        sub_D = D[(D['CALL_TYPE']=='A')].reset_index(drop=True)
    elif args.cate == 'B':
        sub_D = D[(D['CALL_TYPE']=='B')].reset_index(drop=True)
    else:
        sub_D = D[(D['CALL_TYPE']=='C')].reset_index(drop=True)
    
    if args.model == 'CTGAN':
        # idx_lst: indices of all samples with selected seq_len
        print('Start training.')
        syn, idx_lst = train_ctgan(args.parameter['it'], sub_D, args.cate, args.parameter['repeat'], args.parameter['n_sample'], args.seq_len, args.epochs)
        print('Finish Synthetic Data Generation')
    
    elif args.model == 'RGAN':
        train_loader, test, scaler = prepare_rgan_data(sub_D, args.seq_len, args.parameter['test_sz'], args.batch_size)
        G, D, criterion, optimizer_G, optimizer_D = init_rgan(args.seq_len, args.parameter['noise_sz'], args.parameter['in_sz'], args.parameter['out_sz'], args.parameter['hidden_sz'], args.parameter['num_layer'], device)
        print('Start training.')
        G, D, G_losses, D_losses = train_rgan(args.epochs, train_loader, args.seq_len, args.parameter['noise_sz'], G, D, criterion, optimizer_G, optimizer_D, device)
        syn_r = G(test.view(args.seq_len, -1, args.parameter['noise_sz']).to(device))
        t = syn_r.transpose(0, 1).reshape(-1, args.seq_len, args.parameter['out_sz'])
        syn = scaler.inverse_transform(torch.reshape(t, (len(test),-1)).cpu().detach().numpy())
        print('Finish Synthetic Data Generation')
    
    else:
        args.parameter['iterations'] = args.epochs
        args.parameter['batch_size'] = args.batch_size
        print('Start training.')
        data = tgan_format_data(sub_D, args.seq_len)
        syn = timegan(data, args.parameter)
        print('Finish Synthetic Data Generation')

    # Metrics:
    if args.model == 'CTGAN':
        real_data = sample_real(sub_D, idx_lst, args.epochs)
        dtw_sc = dtw_score(syn.values, real_data.values)
        print('Average Dynamic Time Warping score is: ' + str(np.mean(dtw_sc)))
        mmd_sc = mmd_score(torch.Tensor(syn.values), torch.Tensor(real_data.values), device)
        print('Maximum Mean Descrepancy score is: ' + str(mmd_sc.numpy()))
    
    elif args.model == 'RGAN':
        dtw_sc = dtw_score(syn, test.cpu().detach().numpy())
        print('Average Dynamic Time Warping score is: ' + str(np.mean(dtw_sc)))
        mmd_sc = mmd_score(torch.Tensor(syn), torch.Tensor(scaler.inverse_transform(test)), device)
        print('Maximum Mean Descrepancy score is: ' + str(mmd_sc.numpy()))
    
    else:
        real_data = format_data(sub_D, args.seq_len)
        dtw_sc = dtw_score(syn.transpose(0,2,1).reshape(len(real_data),-1), real_data.values)
        print('Average Dynamic Time Warping score is: ' + str(np.mean(dtw_sc)))
        id1 = np.random.choice(range(len(real_data)), size=1000)
        id2 = np.random.choice(range(len(real_data)), size=1000)
        mmd_sc = mmd_score(torch.Tensor(syn.transpose(0,2,1).reshape(len(real_data),-1)[id1,:]), torch.Tensor(real_data.values[id2,:]), device)
        print('Maximum Mean Descrepancy score is: ' + str(mmd_sc.numpy()))
        
    if args.plot == True:
        plot_routes(np.array(syn), 10)
    if args.model == 'RGAN':
        fig = plt.figure()
        plt.plot(G_losses, c='red', linewidth=3)
        plt.plot(D_losses, c='blue', linewidth=3)
        plt.legend(['Generator', 'Discriminator'])
        plt.xlabel('Iterations')
        plt.ylabel('BCE loss')
        plt.show()
    
    return syn, dtw_sc, mmd_sc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['CTGAN','RGAN','TimeGAN'], type=str)
    parser.add_argument('--cate', help='taxi call type', choices=['A','B','C'], type=str)
    parser.add_argument('--seq_len', help='sequence length', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--parameter', help='parameters for selected model', type=ast.literal_eval)
    parser.add_argument('--plot', choices=[True, False], help='plot sample synthetic routes', type=bool)
    args = parser.parse_args()
    generated_data, dtw_sc, mmd_sc = main(args)
