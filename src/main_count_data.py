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
    learning_rate = 1e-3
    # weight_decay = 1e-5
    window_len = 5*7

    common_params = {
        'window_length': window_len,
        'badge_focus': 'Electorate',
        'out_dim': 'QuestionVotes',
        'data_path': '../data',
    }

    dset_train = so_data.StackOverflowDatasetIncCounts(dset_type='train', subsample=4000,
                                                       **common_params, self_initialise=True)
    scalers = dset_train.get_scalers()
    # dset_test = so_data.StackOverflowDatasetIncCounts(dset_type='test', subsample=1000,
                                                      # **common_params, scaler_in=scalers[0], scaler_out=scalers[1])
    dset_valid = so_data.StackOverflowDatasetIncCounts(dset_type='validate', subsample=1000, centered=True,
                                                       **common_params, scaler_in=scalers[0], scaler_out=scalers[1])

    train_loader = DataLoader(dset_train, **params)
    valid_loader = DataLoader(dset_valid, **params)

    model_to_test = {
        'baseline_count': models.BaselineVAECount,
        'linear_count': models.LinearParametricVAECount,
        'personalised_linear_count': models.LinearParametricPlusSteerParamVAECount,
        'full_parameterised_count': models.FullParameterisedVAECount,
        'full_personalised_parameterised_count': models.FullParameterisedPlusSteerParamVAECount,
        'full_personalised_normalizing_flow': models.NormalizingFlowFP_PlusSteer
    }

    dset_shape = dset_train.data_shape

    for name, model_class in model_to_test.items():

        model = model_class(obsdim=dset_shape[0]*dset_shape[1], outdim=dset_shape[0], device=device, proximity_to_badge=True).to(device)
        # PATH = '../models/' + name + '.pt'
        # model.load_state_dict(torch.load(PATH, map_location=device))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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

        if name == 'full_personalised_normalizing_flow':
            beta = torch.tensor(1. * (i / float(NUM_EPOCHS))).float().to(model.device)
        else:
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
            if validation_loss>0:
                torch.save(model.state_dict(), "../models/{}.pt".format(name))

            model.train()

    return model




badge_threshold = 500
badge_type = "Electorate"
l1_loss = torch.nn.L1Loss(reduction='sum')

def build_data(data):
    train_x, val_x, test_x = [], [], []
    train_y, val_y, test_y = [], [], []
    test_yT = {}
    day_of_badge = {}

    for label, dset, outcome in tqdm(zip(['train', 'valid', 'test'],
                                                  [train_x, val_x, test_x],
                                                  [train_y, val_y, test_y])):

        seqs = [s.numpy() for s in data[label]['sequences']]
        badges = data[label]['outcomes']
        mask = np.ones(90).astype(bool)

        seq_shape = list(seqs[0].shape)
        seq_shape[0] = np.sum(mask)

        input_vecs = np.zeros(shape=(len(seqs), np.sum(mask), seq_shape[1]))
        proximity = np.zeros(shape=(len(seqs), np.sum(mask)))

        for i, seq in enumerate(seqs):
            if badge_type in badges[i]:
                badge_day = 45
                acts_for_badge = seq[:, 5]

                cs = np.cumsum(acts_for_badge)
                cs[45:] += 1
                cs[46:] += 1

                proximity[i] = cs[mask]
                proximity[i] = (proximity[i] - cs[45]) + badge_threshold
                proximity[i][proximity[i] < 0] = 0
                proximity[i][proximity[i] > badge_threshold] = 0

            input_vecs[i] = np.array(seq)[mask]


        sequence_data = input_vecs

        lookback = 56
        offset = (seq_shape[0] - lookback) // 7

        inputs = np.zeros((len(sequence_data) * offset, lookback, seq_shape[1]))
        outputs = np.zeros((len(sequence_data) * offset, 7))
        prox_to_badge = np.zeros((len(sequence_data) * offset, lookback))
        badge_day = np.zeros(len(sequence_data)* offset)

        for i, seq in enumerate(sequence_data):
            for j in np.arange(lookback, seq_shape[0] - 7, 7):
                #             print(i*offset + (j - lookback)//7)
                inputs[i * offset + (j - lookback) // 7] = seq[j - lookback:j, [0, 1, 2, 3, 4, 5, 6]]
                outputs[i * offset + (j - lookback) // 7] = seq[j:j + 7, 5] > 1
                prox_to_badge[i * offset + (j - lookback) // 7] = proximity[i][j - lookback:j]
                badge_day[i * offset + (j - lookback) // 7] = 45 - (j-lookback)
                #         inputs[(i * offset + (j - lookback))] = seq[j - lookback: j, [0,1,2,3,6]]
                #         labels[(i * offset + (j - lookback))] = (input_vecs[i, j, 5]) > 0
                #         prox_to_badge[(i * offset + (j - lookback))] = proximity[i, j - lookback: j]

        dset.append(inputs)
        outcome.append(outputs)
        test_yT[label] = prox_to_badge
        day_of_badge[label] = badge_day.astype(int)

    return train_x[0], test_yT['train'], day_of_badge['train'], test_x[0], test_yT['test'], day_of_badge['test']


def test(model, device, test_loader, loss, dset_shape):

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (data, prox, dob,) in enumerate(test_loader):
            data = data.to(device)
            prox = prox.to(device)
            recon_batch, mu, logvar = model(data, prox, dob)

            test_loss += loss(recon_batch, data, mu, logvar, dset_shape[1], dset_shape[2]).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def test_on_votes_only(model, loader, dset_shape, device):

    model.eval()
    test_loss, pred_len = 0, 0
    true_num_1 = 0
    true_num_pos = 0
    pred_pos_len = 0
    false_num_pos = 0
    total_pred_pos = 0
    log_likelihood = 0

    with torch.no_grad():
        for i, (data, prox, dob) in enumerate(loader):
            data = data.to(device)
            prox = prox.to(device)
            recon_batch, mu, logvar = model(data, prox, dob)

            pp = recon_batch.cpu().detach().numpy().reshape(-1, dset_shape[1])
            pred = (recon_batch.cpu().detach().numpy().reshape(-1, dset_shape[1]) > 0.5).astype(int)
            true = (data.detach().cpu().numpy().reshape(-1, dset_shape[1], dset_shape[2])).astype(int)
            true[true > 0] = 1

            for t1, p1, pp_ in zip(true, pred, pp):
                log_likelihood += np.sum(np.log(pp_) * t1[:, 5] + np.log(1 - pp_) * (1 - t1[:, 5] ))
                true_num_1 += np.sum(t1[:,5])
                test_loss += np.sum(p1 == t1[:,5])

                pred_len += len(p1)

                true_num_pos += np.sum(t1[:, 5][t1[:, 5] == 1] == p1[t1[:, 5] == 1])
                pred_pos_len += np.sum(t1[:, 5] == 1)

                false_num_pos += np.sum(t1[:, 5][p1 == 1])
                total_pred_pos += np.sum(p1 == 1)


    print('====> Action specific accuracy: {:.4f}'.format(test_loss/pred_len))
    print('====> Recall: {:.4f}'.format(true_num_pos / pred_pos_len))
    print('====> Precision: {:.4f}'.format(false_num_pos / total_pred_pos))
    print('====> Log-likelihood: {:.4f}'.format(log_likelihood/(i*len(pp_))))
    print('====> Baseline count: {:.4f}'.format(true_num_1 / pred_len))
    print()

if __name__ == "__main__":
    main([])
