#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pickle
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import models
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

import load_so_data as so_data
from torch.backends import cudnn

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

def isnan(x):
    return x != x

def main(args):

    # Loading Parameters
    params = {'batch_size': 256, 'shuffle': True, 'num_workers': 6}

    max_epochs = 2500
    PRINT_NUM = 50
    learning_rate = 1e-3 # think of different training rates for beta and for rest???
    # weight_decay = 1e-5
    window_len = 5*7

    common_params = {
        'window_length': window_len,
        'badge_focus': 'CivicDuty',
        'out_dim': ['QuestionVotes', 'AnswerVotes'],
        'data_path': '../data',
        'badge_threshold': 300,
        'badges_to_avoid': ['Electorate']
    }

    dset_train = so_data.StackOverflowDatasetIncCounts(dset_type='train', subsample=15000,
                                                       **common_params, self_initialise=True)
    scalers = dset_train.get_scalers()
    # dset_test = so_data.StackOverflowDatasetIncCounts(dset_type='test', subsample=1000,
                                                      # **common_params, scaler_in=scalers[0], scaler_out=scalers[1])
    dset_valid = so_data.StackOverflowDatasetIncCounts(dset_type='validate', subsample=5000, centered=True,
                                                       **common_params, scaler_in=scalers[0], scaler_out=scalers[1])

    train_loader = DataLoader(dset_train, **params)
    valid_loader = DataLoader(dset_valid, **params)

    model_to_test = {
        # 'cd_baseline_count': models.BaselineVAECount,
        # 'cd_linear_count': models.LinearParametricVAECount,
        # 'personalised_linear_count': models.LinearParametricPlusSteerParamVAECount,
        # 'cd_full_parameterised_count': models.FullParameterisedVAECount,
        # 'full_personalised_parameterised_count': models.FullParameterisedPlusSteerParamVAECount,
        'cd_baseline_flow_count': models.NormalizingFlowBaseline,
        'cd_fp_flow_count': models.NormalizingFlowFP,
        # 'cd_full_personalised_normalizing_flow': models.NormalizingFlowFP_PlusSteer
    }

    dset_shape = dset_train.data_shape

    for name, model_class in model_to_test.items():

        model = model_class(obsdim=dset_shape[0]*dset_shape[1], outdim=dset_shape[0], device=device, proximity_to_badge=True).to(device)

        PATH = '../models/' + name + '.pt'
        model.load_state_dict(torch.load(PATH, map_location=device))

        my_list = ['badge_param', 'badge_param_bias']
        badge_params = [p[1] for p in list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))]
        base_params = [p[1] for p in list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))]

        optimizer = torch.optim.Adam([{'params': base_params}, {'params': badge_params, 'lr': 1e-5}], lr=1e-5, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)

        loss = lambda x1,x2,x3: models.ZeroInflatedPoisson_loss_function(x1,x2,x3, data_shape=dset_shape, act_choice=5)

        print("Training model for: {}".format(name))
        model = train_model(model, train_loader, valid_loader, optimizer, scheduler, loss,
                            NUM_EPOCHS=max_epochs, PRINT_NUM=PRINT_NUM, name=name)

        print("Done")
        print("Saving model into ../models/{}".format(name))
        torch.save(model.state_dict(), "../models/{}.pt".format(name))

def train_model(model, train_loader, valid_loader, optimizer, scheduler, loss_fn, NUM_EPOCHS, PRINT_NUM=25, name=''):

    for i in tqdm(np.arange(NUM_EPOCHS)):
        model.train()
        train_loss = 0

        # if name == 'full_personalised_normalizing_flow':
        # beta = torch.tensor(1. * (i / float(NUM_EPOCHS))).float().to(model.device)
        # else:
        beta = 1

        for train_in, kernel_data, train_out, train_prox, badge_date in train_loader:

            # Transfer to GPU
            train_in, kernel_data, train_out, train_prox, badge_date = train_in.to(device), kernel_data.to(device), train_out.to(device), train_prox.to(device), badge_date.to(device)

            # Model computations
            recon_batch, latent_loss = model(train_in, kernel_data=kernel_data, dob=badge_date, prox_to_badge=train_prox)
            # print(train_out)
            # print(recon_batch)
            loss = loss_fn(recon_batch, train_out, beta*latent_loss)
            # loss = 100*torch.sum(model.badge_param.pow(2))

            optimizer.zero_grad()
            loss.backward()

            # clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        if i%PRINT_NUM==0:


            model.eval()
            validation_loss = 0
            for val_in, kernel_data, val_out, val_prox, badge_date in valid_loader:
                # Transfer to GPU
                val_in, kernel_data, val_out, val_prox, badge_date = val_in.to(device), kernel_data.to(device), val_out.to(device), val_prox.to(device), badge_date.to(device)
                recon_batch, latent_loss = model(val_in, kernel_data=kernel_data, dob=badge_date, prox_to_badge=val_prox)

                loss = loss_fn(recon_batch, val_out, latent_loss)
                validation_loss += loss.item()

            print('====> Epoch: {} Average Valid loss: {:.4f}'.format(i, validation_loss/len(valid_loader.dataset)))
            if not isnan(validation_loss):
                torch.save(model.state_dict(), "../models/{}.pt".format(name))

            model.train()

    return model

if __name__ == "__main__":
    main([])
