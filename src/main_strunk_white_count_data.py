#!/usr/bin/env python

import sys
import os
import argparse
import logging

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.backends import cudnn

import models
import load_so_data as so_data

available_models = {
        'baseline_count': models.BaselineVAECount,
        # 'cd_linear_count': models.LinearParametricVAECount,
        # 'personalised_linear_count': models.LinearParametricPlusSteerParamVAECount,
        'full_parameterised_count': models.FullParameterisedVAECount,
        'full_personalised_parameterised_count': models.FullParameterisedPlusSteerParamVAECount,
        'baseline_flow_count': models.NormalizingFlowBaseline,
        'fp_flow_count': models.NormalizingFlowFP,
        'full_personalised_normalizing_flow': models.NormalizingFlowFP_PlusSteer
    }

def isnan(x):
    return x != x

def raise_cuda_error():
    raise ValueError('You wanted to use cuda but it is not available. '
                     'Check nvidia-smi and your configuration. If you do '
                         'not want to use cuda, pass the --no-cuda flag.')

def setup_cuda(seed, device):
    if device.index:
        device_str = f"{device.type}:{device.index}"
    else:
        device_str = f"{device.type}"

    os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # This does make things slower :(
    torch.backends.cudnn.benchmark = False

loss_fn = lambda x1,x2,x3: models.ZeroInflatedPoisson_loss_function(x1,x2,x3)

def get_loader_params(args):
    # Loading Parameters
    loader_params = {
        'batch_size': int(args.batch_size),
        'shuffle': True,
        'num_workers': 8
    }
    dset_params = {
        'window_length': args.window_length,
        'badge_focus': 'strunk_white',
        'out_dim': 0,
        'data_path': args.input,
        'badge_threshold': 80,
        'badges_to_avoid': [],
        'ACTIONS': [0]
    }
    return loader_params, dset_params

def main(args):
    # TODO: add checkpointing
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.no_cuda and not use_cuda:
        raise_cuda_error()

    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        logging.info(f'Using device: {torch.cuda.get_device_name()}')

    # For reproducibility:
    #     c.f. https://pytorch.org/docs/stable/notes/randomness.html
    if args.seed is None:
        args.seed = torch.randint(0, 2 ** 32, (1,)).item()
        logging.info(f'You did not set --seed, {args.seed} was chosen')

    if use_cuda:
        setup_cuda(args.seed, device)

    config_args = [str(vv) for kk, vv in vars(args).items()
                   if kk in ['batch_size', 'lr', 'gamma', 'seed']]
    model_name = '_'.join(config_args)

    if not os.path.exists(args.output):
        logging.info(f'{args.output} does not exist, creating...')
        os.makedirs(args.output)

    loader_params, dset_params = get_loader_params(args=args)

    dset_train = so_data.StackOverflowDatasetIncCounts(
                            dset_type='train',
                            subsample=15000,
                            **dset_params,
                            self_initialise=True
                        )
    scalers = dset_train.get_scalers()
    dset_valid = so_data.StackOverflowDatasetIncCounts(
                            dset_type='validate',
                            subsample=5000,
                            centered=True,
                            **dset_params,
                            scaler_in=scalers[0],
                            scaler_out=scalers[1])

    train_loader = DataLoader(dset_train, **loader_params)
    valid_loader = DataLoader(dset_valid, **loader_params)

    print(args.model_name)
    model_class = available_models[args.model_name]

    dset_shape = dset_train.data_shape
    model = model_class(
                obsdim=dset_shape[0] * dset_shape[1],
                outdim=dset_shape[0],
                device=device,
                proximity_to_badge=True
            ).to(device)

    model_name = 'strunk_white-' + args.model_name + "-" + model_name + '.pt'
    PATH_TO_MODEL = args.output+'/models/'+model_name

    if os.path.exists(PATH_TO_MODEL):
        model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    if not os.path.exists(f'{args.output}/logs/'):
        os.mkdir(f'{args.output}/logs/')

    log_fh = open(f'{args.output}/logs/{model_name}.log', 'w')
    best_loss = sys.float_info.max

    results_file = open(f'{args.output}/results.csv', 'a')
    results_file.write(f'{model_name},{args.batch_size},{args.lr},{args.gamma},'
                       f'{args.seed},')

    count_valid_not_improving = 0

    for epoch in tqdm(range(1, args.epochs + 1)):

        loss = train(args, model, device, train_loader, optimizer, epoch)
        vld_loss = test(args, model, device, valid_loader)

        print(f'{epoch},{loss},{vld_loss}', file=log_fh)
        scheduler.step()

        results_file.write(f'{vld_loss},')

        if vld_loss < best_loss:
            # only save the model if it is performing better on the validation set
            best_loss = vld_loss
            torch.save(model.state_dict(),
                       f"{args.output}/models/{model_name}.best.pt")
            count_valid_not_improving = 0

        # early stopping
        else:
            count_valid_not_improving += 1

        if count_valid_not_improving > args.early_stopping_lim:
            print(f'Early stopping implemented at epoch #: {epoch}')
            break

    results_file.write('\n')
    torch.save(model.state_dict(), f"{args.output}/models/{model_name}.final.pt")

    results_file.close()
    log_fh.close()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    beta = 1

    for batch_idx, (data) in enumerate(train_loader):
        data = [d.to(device) for d in data]
        dat_in, dat_kern, dat_out, dat_prox, dat_badge_date = data

        optimizer.zero_grad()
        # Model computations
        recon_batch, latent_loss = model(dat_in,
                                         kernel_data=dat_kern,
                                         dob=dat_badge_date,
                                         prox_to_badge=dat_prox)
        loss = loss_fn(recon_batch, dat_out, beta * latent_loss)
        loss.backward()
        # TODO: clip grad norm here?
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % args.log_interval == 0 and not args.quiet:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return train_loss/len(train_loader.dataset)

def test(args, model, device, valid_loader):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data) in enumerate(valid_loader):
            data = [d.to(device) for d in data]
            dat_in, dat_kern, dat_out, dat_prox, dat_badge_date = data

            recon_batch, latent_loss = model(dat_in,
                                            kernel_data=dat_kern,
                                            dob=dat_badge_date,
                                            prox_to_badge=dat_prox)

            loss = loss_fn(recon_batch, dat_out, latent_loss)
            test_loss += loss.item()

    test_loss /= len(valid_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))

    return test_loss

def construct_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch main script to run inference detailed here: '
                                                 'https://arxiv.org/abs/2002.06160'
                                                 'on the population of users who achieved Strunk & White')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--early-stopping-lim', type=int, default=10, metavar='N',
                        help='Early stopping implemented after N epochs with no improvement '
                             '(default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N', help='input batch size for testing '
                                          '(default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Limits the about of output to std.out')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: random number)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--window-length', type=int, default=35, metavar='N',
                        help='how long is the window that is considered before / after a badge (default: 35)')
    parser.add_argument('-M', '--model-name', default="full_personalised_normalizing_flow", required=False,
                        help='Choose the model to run')
    parser.add_argument('-i', '--input', required=True, help='Path to the input data for the model to read')
    parser.add_argument('-o', '--output', required=True, help='Path to the directory to write output to')
    return parser

if __name__ == "__main__":
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
