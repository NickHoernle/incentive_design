import os
from collections import namedtuple
from urllib.request import urlopen
import pickle

import torch
import numpy as np
import pandas as pd
import json

from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler

ACTIONS = ['Answers', 'Questions', 'Comments', 'Edits', 'AnswerVotes', 'QuestionVotes', 'ReviewTasks']
BADGES = ['CivicDuty', 'CopyEditor', 'Electorate', 'Reviewer', 'Steward', 'StrunkWhite']
action_ixs = {a:i for i,a in enumerate(ACTIONS)}

csv_path = '/Users/nickhoernle/edinburgh/incentive_design/data/'

'''
Scripts that transform the raw CSV data from stack overflow into a data folder that contains
.pt array.
folder/
|---- src/
|--------- scripts.py
|---- data/
|--------- badge_achievements.json
|--------- data_indexes.json
|--------- user_{id}.pt
'''

from torch.utils import data

class StackOverflowDataset(data.Dataset):

    def __init__(self, data_path='../data', dset_type='train', badge_focus='Electorate', subsample=5000, centered=True, window_length=70, input_length='full', out_dim=None):
        '''
        Initialise the stack overflow data loader
        :param list_IDs: list of ids that belong to this dataset
        :param badge_ids: dictionary of badges achieved
        :param badge_focus: focus on a particular badge
        :param subsample: subsample the data to smaller fraction
        :param centered: to center the time series around the badge achievement
        :param window_length: length of window around badge point
        :param dset_type: can be "train", "test", "validate"
        :param input_length: 'full' or length specifying the length of the original time series to use in the
                             inference network (encoding) step
        :param out_dim: dimension of the output to use, if None return the whole array, if string then return the
                        string must be a specific action type.
        '''

        with open("{}/badge_achievements.json".format(data_path), 'r') as f:
            badge_ids = json.load(f)
        with open("{}/data_indexes.json".format(data_path), 'r') as f:
            list_IDs = json.load(f)
        self.badge_focus = badge_focus
        self.subsample = subsample
        self.centered = centered
        self.window_length = window_length
        self.dset_type = dset_type
        self.data_path = data_path

        if input_length == 'full':
            self.input_length = self.window_length * 2
        else:
            self.input_length = input_length

        if out_dim == None:
            self.out_dim = [0,1,2,3,4,5,6]
        elif type(out_dim) == str:
            self.out_dim = action_ixs[out_dim]
        else:
            self.out_dim = out_dim

        self.list_IDs, self.badge_ids = self._preprocess_user_ids(list_IDs[dset_type], badge_ids)
        self.data_shape = (self.input_length, len(ACTIONS))

    def _preprocess_user_ids(self, list_IDs, badge_ids):
        feasible_users = []
        feasible_badges = {}
        for user in list_IDs:
            user = str(user)
            if self.badge_focus in badge_ids[user]:
                feasible_users.append(user)
                feasible_badges[user] = int(np.random.choice(badge_ids[user][self.badge_focus], size=1))

        assert len(feasible_users) >= self.subsample
        return np.random.choice(feasible_users, size=self.subsample, replace=False), feasible_badges

    def __data_trans_in(self, data):
        data[data > 0] = 1
        return data.float().numpy()

    def __data_trans_out(self, data):
        data[data > 0] = 1
        return data.float().numpy()

    def __len__(self):
        return len(self.list_IDs)

    def __get_prox_to_badge(self, output, badge_index):
        prox_to_badge = torch.cumsum(output[:, self.out_dim], dim=0).float()
        prox_to_badge = (prox_to_badge + (500 - prox_to_badge[badge_index])) / 500

        prox_to_badge[badge_index + 1:] = 0
        prox_to_badge[prox_to_badge < 0] = 0
        prox_to_badge = prox_to_badge - prox_to_badge[0]
        prox_to_badge[badge_index + 1:] = 0
        prox_to_badge[prox_to_badge < 0] = 0

        return prox_to_badge

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]
        badge_index = self.badge_ids[ID]

        # Load data and get label
        X = torch.load(os.path.join(self.data_path, 'user_' + ID + '.pt'))
        output = torch.zeros(size=(2*self.window_length, X.size()[1]))

        if self.centered:
            center = badge_index
            # correct where the badge was achieved
            badge_index = self.window_length

        else:
            start = np.max((0, badge_index-self.window_length+1))
            stop = np.min((badge_index+self.window_length-1, X.size()[0]))

            center = np.random.choice(np.arange(start, stop))
            if badge_index < center:
                badge_index = (self.window_length - (center-badge_index))
            else:
                badge_index = (badge_index - center) + self.window_length

        if center < self.window_length:
            output[self.window_length - center:] = X[0:center + self.window_length]
        elif X.size()[0] - self.window_length < center:
            diff = X.size()[0] - center
            output[:self.window_length + diff] = X[center - self.window_length:]
        else:
            output = X[center - self.window_length:center + self.window_length, :]

        x_in = self.__data_trans_in(output[:self.input_length,:].clone())
        x_out = self.__data_trans_out(output[:,self.out_dim].clone())

        # get the proximity to a badge
        prox_to_badge = self.__get_prox_to_badge(output, badge_index)

        # create the kernel that indexes the badge beung accepted at index=2*window_length
        kernel_data = np.arange(0, 4*self.window_length)

        start = 2*self.window_length-badge_index
        stop = 4*self.window_length-badge_index

        return (torch.tensor(x_in).float(),
                torch.tensor(kernel_data[start: stop]).view(-1).float(),
                torch.tensor(x_out).float(),
                prox_to_badge.view(-1,).float(),
                torch.tensor(badge_index, dtype=torch.float))

class IdentityScaler:
    def transform(self, x):
        return x
    def inverse_transform(self, x):
        return x

class StackOverflowDatasetIncCounts(StackOverflowDataset):
    def __init__(self, data_path='../data', dset_type='train', badge_focus='Electorate', subsample=5000, centered=True,
                 window_length=70, input_length='full', out_dim=None, scaler_in=IdentityScaler(), scaler_out=IdentityScaler(), self_initialise=False):
        super(StackOverflowDatasetIncCounts, self).__init__(data_path=data_path, dset_type=dset_type, badge_focus=badge_focus, subsample=subsample,
                         centered=centered, window_length=window_length, input_length=input_length, out_dim=out_dim)

        self.scaler_in = scaler_in
        self.scaler_out = scaler_out

        if self_initialise:
            self.scaler_in, self.scaler_out = calculate_feature_transformation(self)

    def _StackOverflowDataset__data_trans_in(self, data):
        mid = torch.log1p(data.float().view(-1, self.data_shape[0]*self.data_shape[1])).numpy()
        return self.scaler_in.transform(mid).reshape(self.data_shape[0], self.data_shape[1])

    def _StackOverflowDataset__data_trans_out(self, data):
        # return self.scaler_out.transform(torch.log1p(data.float()))
        return data.float().numpy()

    def get_scalers(self):
        return self.scaler_in, self.scaler_out

    def inverse_transform_in(self, data):
        return torch.exp(self.scaler_in.inverse_transform(data.float))-1

    def inverse_transform_out(self, data):
        return data.float

def calculate_feature_transformation(train_dataset):
    dat_in, dat_out = [], []

    print("Processing training data")
    for i in tqdm(range(len(train_dataset))):
        in_d, kern, out_d, _, _ = train_dataset.__getitem__(i)
        dat_in.append(in_d.numpy())
        dat_out.append(out_d.numpy())

    dat_in = np.array(dat_in)
    dat_out = np.array(dat_out)

    dset_shape = dat_in.shape
    dat_in = dat_in.reshape(-1, dset_shape[1] * dset_shape[2])
    dat_out = dat_out.reshape(-1, dset_shape[1])

    scaler_in = MinMaxScaler(feature_range=(-1, 1))
    scaler_out = MinMaxScaler(feature_range=(0, 1))

    scaler_in.fit(dat_in)
    scaler_out.fit(dat_out)

    return scaler_in, scaler_out


######################################################################################################################
######################################################################################################################
###################################### LEGACY AND FOR INITIAL DATA TRANSFORMATION ####################################
######################################################################################################################
######################################################################################################################

def process_data(base_path):
    # processed_dataset = {}
    # validation == 1000 samples
    # train === 5000 samples
    # test === 1000 samples
    # convert to number of actions per week
    # edit out the badge outcome variables

    print("Processing raw data")

    output_fname = os.path.join(base_path, 'so_data.pkl')

    labels = ['train', 'valid', 'test']

    input_fname = os.path.join(csv_path, 'so_badges.csv')
    data = pd.read_csv(input_fname)
    data.Date = pd.to_datetime(data.Date)
    data['week'] = (data.Date - pd.datetime(year=2017, month=1, day=1)).dt.days

    data = data.groupby(['DummyUserId', 'week']).agg('sum').reset_index()
    badge_ixs = data[data.Electorate > 0]
    max_week = data.week.max()
    badge_ixs = badge_ixs[badge_ixs.week > 45]
    badge_ixs = badge_ixs[badge_ixs.week < max_week - 46]
    badge_ixs = badge_ixs.DummyUserId

    print(len(badge_ixs.unique()))

    indexes = badge_ixs.unique()
    train = np.random.choice(indexes, size=4000, replace=False)
    indexes = indexes[~np.in1d(indexes, train)]
    validate = np.random.choice(indexes, size=1000, replace=False)
    indexes = indexes[~np.in1d(indexes, validate)]
    test = np.random.choice(indexes, size=1000, replace=False)

    # data.set_index('DummyUserId', inplace=True)
    processed_dataset = {}

    for s, dset in enumerate([train, validate, test]):

        split = labels[s]
        processed_dataset[split] = {}

        sub_data = data[data.DummyUserId.isin(dset)]
        n_seqs = len(dset)

        processed_dataset[split]['sequence_lengths'] = torch.zeros(n_seqs, dtype=torch.long)
        processed_dataset[split]['sequences'] = []
        processed_dataset[split]['outcomes'] = []
        idx = 0

        for u_id, seqs in sub_data.groupby('DummyUserId'):
            seqs = seqs.sort_values('week')

            out = {}
            for b in BADGES:
                idxs = np.where(seqs[b] == 1)[0]
                if len(idxs) > 0:
                    out[b] = torch.tensor(idxs, dtype=torch.long)

            civic_duty = out['Electorate']
            days = 90

            action_vec = seqs[ACTIONS].values[civic_duty - days // 2:civic_duty + days // 2, :]
            out['Electorate'] = torch.tensor([days // 2], dtype=torch.long)

            processed_dataset[split]['sequence_lengths'][idx] = days
            processed_sequence = torch.tensor(action_vec, dtype=torch.long)
            processed_dataset[split]['sequences'].append(processed_sequence)

            processed_dataset[split]['outcomes'].append(out)
            idx += 1

    pickle.dump(processed_dataset, open(output_fname, "wb"), pickle.HIGHEST_PROTOCOL)
    print("dumped processed data to %s" % output_fname)


def load_data(base_path):
    output_fname = os.path.join(base_path, 'so_data.pkl')
    if not os.path.exists(output_fname):
        process_data(base_path)

    return pickle.load(open(output_fname, 'rb'))


def transform_data_to_file_folder_structure(path_to_csv, path_to_data_dir):
    data = pd.read_csv(path_to_csv)
    data.Date = pd.to_datetime(data.Date)
    data['day'] = (data.Date - pd.datetime(year=2017, month=1, day=1)).dt.days

    data = data.groupby(['DummyUserId', 'day']).agg('sum').reset_index()
    user_ids = data.DummyUserId.unique()
    size_data = len(user_ids)
    # print(int(np.floor(0.6*size_data)))

    np.random.seed(11)

    train = np.random.choice(user_ids, size=int(np.floor(0.6 * size_data)), replace=False)
    user_ids = user_ids[~np.in1d(user_ids, train)]
    validate = np.random.choice(user_ids, size=int(np.floor(0.2 * size_data)), replace=False)
    user_ids = user_ids[~np.in1d(user_ids, validate)]
    test = np.random.choice(user_ids, size=int(np.floor(0.2 * size_data)), replace=False)
    # print(len(user_ids))

    badge_achievements = {}

    for dset in [train, validate, test]:
        for user, trajectory in data[data.DummyUserId.isin(dset)].groupby('DummyUserId'):
            trajectory = trajectory.sort_values('day')
            badge = {}
            for b in BADGES:
                idxs = np.where(trajectory[b] == 1)[0]
                if len(idxs) > 0:
                    badge[b] = [int(i) for i in idxs]
            badge_achievements[user] = badge

            action_trajectory = torch.tensor(trajectory[ACTIONS].values, dtype=torch.long)
            torch.save(action_trajectory, '{}/user_{}.pt'.format(path_to_data_dir, user))

    with open('{}/badge_achievements.json'.format(path_to_data_dir), 'w') as f:
        json.dump(badge_achievements, f)

    with open('{}/data_indexes.json'.format(path_to_data_dir), 'w') as f:
        obj = {}
        obj['train'] = [int(u) for u in train]
        obj['test'] = [int(u) for u in test]
        obj['validate'] = [int(u) for u in validate]
        json.dump(obj, f)
