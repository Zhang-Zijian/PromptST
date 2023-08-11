import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
# import utils
from normalization import Standard
import pickle

class traffic_demand_prediction_dataset(Dataset):
    def __init__(self, x, y, key, val_len, test_len):
        self.x = x
        self.y = y
        self.key = key
        self._len = {"train_len": x.shape[0] - val_len - test_len,
                     "val_len": val_len, "test_len": test_len}

    def __getitem__(self, item):
        if self.key == 'train':
            return self.x[item], self.y[item]
        elif self.key == 'val':
            return self.x[self._len["train_len"] + item], self.y[self._len["train_len"] + item]
        elif self.key == 'test':
            return self.x[-self._len["test_len"] + item], self.y[-self._len["test_len"] + item]
        else:
            raise NotImplementedError()

    def __len__(self):
        return self._len[f"{self.key}_len"]

def get_data_loader(dataset_name, output_channel, normal_flag, data_dir='../graph_CCRNN/'):
    X_list = [12,11,10,9,8,7,6,5,4,3,2,1]
    Y_list = [0,1,2,3,4,5,6,7,8,9,10,11]
    Y_list = Y_list[:output_channel]

    basic_dir = '/data/zcxbo/autostl/promptST/'
    data, normal = list(), list()
    normal_method = Standard#getattr(normalization, Normal_Method)
    data_category = [dataset_name]
    if dataset_name == 'complaint19_3h':
        _len = [584,1168]
        data = np.load(f'{basic_dir}data_process/complaint/complaint19_3h.npy')
        normal.append(normal_method())
        data = normal[0].fit_transform(data)
    elif dataset_name.startswith('complaint19_3h'):
        _index = dataset_name.replace('complaint19_3h', '')        
        _len = [584,1168]
        data = np.load(f'{basic_dir}data_process/complaint/complaint19_3h{_index}.npy')
        normal.append(normal_method())
        data = normal[0].fit_transform(data)
        data = np.expand_dims(data, -1)
    elif dataset_name == 'nyctaxi2014':
        _len = [648,1296]
        data = np.load(f'{basic_dir}data_process/NYCTaxi/NYCTaxi_JFM.npy')
        normal.append(normal_method())
        data = normal[0].fit_transform(data)
    elif dataset_name.startswith('nyctaxi2014'):
        _index = dataset_name.replace('nyctaxi2014_', '')        
        _len = [648,1296]
        data = np.load(f'{basic_dir}data_process/NYCTaxi/NYCTaxi_JFM{_index}.npy')
        normal.append(normal_method())
        data = normal[0].fit_transform(data)
        data = np.expand_dims(data, -1)
    elif dataset_name == 'PEMSD4':
        _len = [648,1296]
        data = np.load(f'{basic_dir}data_process/PEMSD4/PEMSD4.npy')
        normal.append(normal_method())
        data = normal[0].fit_transform(data)
    elif dataset_name.startswith('PEMSD4'):
        _index = dataset_name.replace('PEMSD4_', '')        
        _len = [1699,3398]
        data = np.load(f'{basic_dir}data_process/PEMSD4/PEMSD4_{_index}.npy')
        normal.append(normal_method())
        data = normal[0].fit_transform(data)
        data = np.expand_dims(data, -1)
    else:
        assert 1==0, 'wrong dataset name'
    # print(f'data shape: {data.shape}')
    # print(f'data: {data.shape}, {data[:10]}')

    X_, Y_ = list(), list()
    for i in range(max(X_list), data.shape[0] - max(Y_list)):
        X_.append([data[i - j] for j in X_list])
        Y_.append([data[i + j] for j in Y_list])
    X_ = torch.from_numpy(np.asarray(X_)).float()
    Y_ = torch.from_numpy(np.asarray(Y_)).float()
    # print(f'X: {X_.shape}')
    dls = dict()

    for key in ['train', 'val', 'test']:
        dataset = traffic_demand_prediction_dataset(X_, Y_, key, _len[0], _len[1])
        # print(f'dataset {key} length: {dataset._len}')
        # dls[key] = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=16)
        dls[key] = dataset
    return dls, normal[0]

def get_graph(dataset_name):
    pickle_file = '../graph_CCRNN/adj_mx_'+dataset_name+'.pkl'
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data