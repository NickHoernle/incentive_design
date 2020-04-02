#!/usr/bin/env python

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

    def __init__(self, data_path='../data', dset_type='train',
                 badge_focus='Electorate', badges_to_avoid=[], subsample=5000,
                 centered=True, window_length=70,
                 input_length='full', out_dim=None,
                 return_user_id=False,
                 badge_threshold=600, ACTIONS=ACTIONS):

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
        self.return_user_id = return_user_id
        self.badge_threshold = badge_threshold
        self.badges_to_avoid = badges_to_avoid
        self.ACTIONS = ACTIONS

        if input_length == 'full':
            self.input_length = self.window_length * 2
        else:
            self.input_length = input_length

        if out_dim == None:
            self.out_dim = [0,1,2,3,4,5,6]
        elif type(out_dim) == str:
            self.out_dim = action_ixs[out_dim]
        elif type(out_dim) == list:
            self.out_dim = [action_ixs[elem] for elem in out_dim]
        elif type(out_dim) == int:
            self.out_dim = out_dim

        self.list_IDs, self.badge_ids = self._preprocess_user_ids(list_IDs[dset_type], badge_ids)
        self.data_shape = (self.input_length, len(ACTIONS))

    def _preprocess_user_ids(self, list_IDs, badge_ids):
        feasible_users = []
        feasible_badges = {}

        for user in list_IDs:
            user = str(user)

            if self.badge_focus in badge_ids[user]:
                valid = True

                for badge in self.badges_to_avoid:
                    if badge in badge_ids[user]:
                        valid = False
                        break

                if valid:
                    feasible_users.append(user)
                    feasible_badges[user] = int(np.random.choice(badge_ids[user][self.badge_focus], size=1))

        if len(feasible_users) < self.subsample:
            self.subsample = len(feasible_users)

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
        if type(self.out_dim) == list:
            prox_to_badge = torch.cumsum(output[:, self.out_dim].sum(dim=-1), dim=0).float()
        else:
            prox_to_badge = torch.cumsum(output[:, self.out_dim], dim=0).float()
        prox_to_badge = (prox_to_badge + (self.badge_threshold - prox_to_badge[badge_index])) / self.badge_threshold

        prox_to_badge = 1 - prox_to_badge
        prox_to_badge[badge_index + 1:] = 0
        prox_to_badge[prox_to_badge < 0] = 0
        prox_to_badge[prox_to_badge > 1] = 1
        # prox_to_badge = prox_to_badge - prox_to_badge[0]
        # prox_to_badge[badge_index + 1:] = 0
        # prox_to_badge[prox_to_badge < 0] = 0

        return prox_to_badge

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]
        badge_index = self.badge_ids[ID]

        # Load data and get label
        X = torch.load(os.path.join(self.data_path, 'user_' + ID + '.pt'))
        output = torch.zeros(size=(2*self.window_length, X.size()[1]))

        # if torch.sum(X[badge_index, [4,5]]) == 0:
        #     badge_index = badge_index-1
        #     if torch.sum(X[badge_index, [4, 5]]) == 0:
        #         print(X[badge_index-1:badge_index+3].T)
        #         print()

        if self.centered:
            center = badge_index
            # correct where the badge was achieved
            badge_index = self.window_length

        else:
            start = np.max((0, badge_index-self.window_length+1))
            stop = np.min((badge_index+self.window_length-1, X.size()[0]))
            options = np.arange(start, stop)
            np.random.seed()
            center = np.random.choice(options)

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
            output = X[center - self.window_length : center + self.window_length, :]

        x_in = self.__data_trans_in(output[:self.input_length,:].clone())
        if type(self.out_dim) == list:
            x_out = self.__data_trans_out(output[:, self.out_dim].sum(dim=-1).clone())
        else:
            x_out = self.__data_trans_out(output[:,self.out_dim].clone())

        # get the proximity to a badge
        prox_to_badge = self.__get_prox_to_badge(output, badge_index)[:self.input_length]

        # create the kernel that indexes the badge beung accepted at index=2*window_length
        kernel_data = np.arange(0, 4*self.window_length)

        start = 2*self.window_length-badge_index
        stop = 4*self.window_length-badge_index

        if self.return_user_id:
            return (torch.tensor(x_in).float(),
                    torch.tensor(kernel_data[start: stop]).view(-1).float(),
                    torch.tensor(x_out).float(),
                    prox_to_badge.view(-1, ).float(),
                    torch.tensor(badge_index, dtype=torch.float),
                    ID
                    )

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
    def __init__(self, data_path='../data', dset_type='train',
                 badge_focus='Electorate',
                 subsample=5000,
                 centered=True,
                 window_length=70,
                 input_length='full',
                 out_dim=None,
                 scaler_in=IdentityScaler(),
                 scaler_out=IdentityScaler(),
                 self_initialise=False,
                 return_user_id=False, **kwargs):

        super(StackOverflowDatasetIncCounts, self).__init__(
                        data_path=data_path,
                        dset_type=dset_type,
                        badge_focus=badge_focus,
                        subsample=subsample,
                        centered=centered,
                        window_length=window_length,
                        input_length=input_length,
                        out_dim=out_dim,
                        return_user_id=return_user_id,
                        **kwargs)

        self.scaler_in = scaler_in
        self.scaler_out = scaler_out

        if self_initialise:
            self.scaler_in, self.scaler_out = calculate_feature_transformation(self)

    def _StackOverflowDataset__data_trans_in(self, data):
        mid = (data.float().view(-1, self.data_shape[0]*self.data_shape[1])).numpy()
        return self.scaler_in.transform(mid).reshape(self.data_shape[0], self.data_shape[1])

    def _StackOverflowDataset__data_trans_out(self, data):
        # return self.scaler_out.transform(torch.log1p(data.float()))
        return data.float().numpy()

    def get_scalers(self):
        return self.scaler_in, self.scaler_out

    def inverse_transform_in(self, data):
        return self.scaler_in.inverse_transform(data)

    def inverse_transform_out(self, data):
        return data.float

def calculate_feature_transformation(train_dataset):
    dat_in, dat_out = [], []

    print("Processing training data")
    for i in tqdm(range(len(train_dataset))):
        resp = train_dataset.__getitem__(i)

        in_d = resp[0]
        out_d = resp[2]

        dat_in.append(in_d.numpy())
        dat_out.append(out_d.numpy())

    dat_in = np.array(dat_in)
    maxes_in = {}
    if len(train_dataset.ACTIONS) > 1:
        for i, action in enumerate(ACTIONS):
            maxes_in[action] = np.max(dat_in[:,:,i])
            dat_in[:, :, i] = dat_in[:,:,i]/maxes_in[action]
    else:
        maxes_in[0] = np.max(dat_in[:, :, 0])
        dat_in[:, :, 0] = dat_in[:, :, 0]/maxes_in[0]

    maxes_out = np.max(dat_out)
    dat_out = dat_out / maxes_out

    class ScalerIn(IdentityScaler):
        def __init__(self, maxes_in, actions=ACTIONS):
            self.maxes_in = maxes_in
            self.ACTIONS = actions

        def transform(self, x):
            if len(x.shape) == 2:
                x = x.reshape(1,-1,len(self.ACTIONS))
            for i,a in enumerate(self.ACTIONS):
                x[:, :, i] = x[:, :, i] / self.maxes_in[a]
            return x

        def inverse_transform(self, x):
            if len(x.shape) == 2:
                x = x.reshape(1,-1,len(self.ACTIONS))
            for i,a in enumerate(self.ACTIONS):
                x[:, :, i] = x[:, :, i] * self.maxes_in[a]
            return x

    class ScalerOut(IdentityScaler):
        def __init__(self, maxes_in):
            self.maxes_in = maxes_in
        def transform(self, x):
            return x/self.maxes_in

        def inverse_transform(self, x):
            return x*self.maxes_in

    scaler_in = ScalerIn(maxes_in, train_dataset.ACTIONS)
    scaler_out = ScalerOut(maxes_out)
    # dat_out = np.array(dat_out)

    # dset_shape = dat_in.shape
    # dat_in = dat_in.reshape(-1, dset_shape[1] * dset_shape[2])

    # scaler_in = MinMaxScaler(feature_range=(-1, 1))
    # scaler_out = MinMaxScaler(feature_range=(0, 1))

    # scaler_in.fit(dat_in)
    # scaler_out.fit(dat_out)

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

def compile_smaller_files(input_actions, input_badges):
    output_action_f_name = '../data/editor/actions_over_time.csv'
    output_badges_f_name = '../data/editor/strunk_and_white_achievements.csv'

    df_actions = pd.read_csv(input_actions[0])
    for file in input_actions[1:]:
        df_actions_temp = pd.read_csv(file)
        df_actions = df_actions.merge(df_actions_temp, on='UserId', suffixes=("_x", ""), how='outer')

    df_badges = pd.read_csv(input_badges[0])
    for file in input_badges[1:]:
        df_badges_temp = pd.read_csv(file)
        df_badges = df_badges.merge(df_badges_temp, on='UserId', suffixes=("_x", ""), how='outer')

    cols_to_drop = df_actions.columns[df_actions.columns.str.contains("_x")]
    df_actions.drop(columns=cols_to_drop, inplace=True)

    cols = df_actions.columns[df_actions.columns.str.contains("-")]
    dates = pd.to_datetime(cols).date
    df_actions.rename(columns={d: c for d, c in zip(cols, dates)}, inplace=True)

    df_actions.drop_duplicates(subset="UserId", inplace=True)
    df_badges.drop_duplicates(subset="UserId", inplace=True)

    df_actions.fillna(0, inplace=True)
    df_badges.fillna(0, inplace=True)

    df_badges.set_index('UserId', inplace=True)
    df_badges.loc[df_badges['Date'] == 0, 'Date'] = df_badges.loc[df_badges['Date'] == 0, 'Date_x']
    df_badges.drop(columns=['Date_x'], inplace=True)
    df_badges = df_badges[df_badges['Date'] != 0]
    df_badges.reset_index(inplace=True)

    df_actions.to_csv(output_action_f_name, index=False)
    df_badges.to_csv(output_badges_f_name, index=False)

def transform_editing_data_to_file_folder_structure(path_to_csv_actions, path_to_csv_badges, path_to_data_dir):
    '''
    Expecting data in the PIVOTED format from the Stack Overflow query editor. 
    Here the csv file has an index of userIds, and the columns are the date from 
    start to end. The values are the counts of edits that that user performed on that
    day. There is a separate file for the userId.
    '''
    import tqdm

    data_actions = pd.read_csv(path_to_csv_actions)
    badge_achievements = pd.read_csv(path_to_csv_badges)

    data_actions = data_actions[data_actions.UserId.isin(badge_achievements.UserId)]
    badge_achievements = badge_achievements[badge_achievements.UserId.isin(data_actions.UserId)]

    badge_achievements.Date = pd.to_datetime(badge_achievements.Date)
    badge_achievements['day'] = (badge_achievements.Date - pd.datetime(year=2015, month=1, day=1)).dt.days

    user_ids = badge_achievements.UserId.unique()
    size_data = len(user_ids)

    np.random.seed(11)

    train = np.random.choice(user_ids, size=int(np.floor(0.6 * size_data)), replace=False)
    user_ids = user_ids[~np.in1d(user_ids, train)]
    validate = np.random.choice(user_ids, size=int(np.floor(0.2 * size_data)), replace=False)
    user_ids = user_ids[~np.in1d(user_ids, validate)]
    test = np.random.choice(user_ids, size=int(np.floor(0.2 * size_data)), replace=False)

    data_actions.set_index('UserId', inplace=True)
    badge_achievements.set_index('UserId', inplace=True)

    num_days = (badge_achievements.Date.max() - pd.datetime(year=2015, month=1, day=1)).days

    for dset in [train, validate, test]:
        for user in tqdm.tqdm(dset):

            trajectory = data_actions.loc[user]
            trajectory = trajectory.reset_index()
            trajectory['index'] = pd.to_datetime(trajectory['index'])
            trajectory['day'] = (trajectory['index'] - pd.datetime(year=2015, month=1, day=1)).dt.days
            trajectory.rename(columns={'index': 'date', user: 'num_actions'}, inplace=True)
            trajectory.sort_values('day', inplace=True)
            trajectory.set_index('day', inplace=True)
            trajectory = trajectory.reindex(range(num_days+1), fill_value=0)

            action_trajectory = torch.tensor(trajectory[['num_actions']].values, dtype=torch.long)
            torch.save(action_trajectory, '{}/user_{}.pt'.format(path_to_data_dir, user))

    with open('{}/badge_achievements.json'.format(path_to_data_dir), 'w') as f:
        badge_dict = badge_achievements['day'].to_dict()
        badge_dict = {k: {'strunk_white': [int(v)]} for k,v in badge_dict.items()}
        json.dump(badge_dict, f)

    with open('{}/data_indexes.json'.format(path_to_data_dir), 'w') as f:
        obj = {}
        obj['train'] = [int(u) for u in train]
        obj['test'] = [int(u) for u in test]
        obj['validate'] = [int(u) for u in validate]
        json.dump(obj, f)


if __name__ == '__main__':

    #### BUILD THE INPUT FILE ####
    # input_a_fs = ['../data/editor/actions-2015-06.csv', '../data/editor/actions-2016-12.csv']
    # input_b_fs = ['../data/editor/badges-2015-06.csv', '../data/editor/badges-2016-12.csv']
    # compile_smaller_files(input_a_fs, input_b_fs)

    #### BUILD THE TORCH INPUT FILES ####
    path_to_csv_actions = '../data/editor/actions_over_time.csv'
    path_to_csv_badges = '../data/editor/strunk_and_white_achievements.csv'
    data_dir = '../data/editor/'

    transform_editing_data_to_file_folder_structure(path_to_csv_actions, path_to_csv_badges, data_dir)
