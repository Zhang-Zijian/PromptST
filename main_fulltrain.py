import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
import numpy as np
import pandas as pd
import sys
sys.path.append('./data_process/')
from data_process import get_data_loader, get_graph
from utils import masked_mae, masked_rmse, masked_mape, asym_adj
from sklearn.metrics import mean_absolute_error, mean_squared_error
from promptST import ttnet, prompt_ttnet
import time, random
import scipy.sparse as sp


def get_dataset(name, path, out_channel, normal_flag):
    print(f'load dataset: {name}, {path}')
    assert path in ['train', 'test', 'val']
    dataset, normal = get_data_loader(name, out_channel, normal_flag)
    return dataset[path], normal

def get_criterion(loss_type, device):
    assert loss_type in ['mse', 'mae', 'rmse', 'mse+mae', 'rmse+mae']
    if loss_type == 'mse':
        criterion = torch.nn.MSELoss().to(device)
    elif loss_type == 'mae':
        criterion = torch.nn.L1Loss().to(device)
    elif loss_type == 'rmse':
        mse_ = torch.nn.MSELoss().to(device)
        criterion = lambda x1, x2:torch.sqrt(mse_(x1, x2))
    elif loss_type == 'mse+mae':
        mse_ = torch.nn.MSELoss().to(device)
        mae_ = torch.nn.L1Loss().to(device)
        criterion = lambda x1, x2:mse_(x1, x2)+mae_(x1, x2)
    elif loss_type == 'rmse+mae':
        mse_ = torch.nn.MSELoss().to(device)
        mae_ = torch.nn.L1Loss().to(device)
        criterion = lambda x1, x2:torch.sqrt(mse_(x1, x2))+mae_(x1, x2)

    return criterion


def train(model, optimizer, train_iterator1, criterion, normal, args, log_interval=400):
    model.train()

    num_example = 0
    MAE_LOSS = torch.nn.L1Loss()
    MSE_LOSS = torch.nn.MSELoss()
    for i, (x, y) in enumerate(train_iterator1):
        x = x.to(args.device)
        y = y.to(args.device)
        optimizer.zero_grad()
        logits = model(x)
        predict = logits

        loss = get_criterion('rmse+mae', args.device)(predict, y)
        # loss = get_criterion('mae', args.device)(predict, y)

        loss.backward()
        optimizer.step()

        # predict = normal.rmse_transform(predict)
        # y = normal.rmse_transform(y)
        predict = normal.inverse_transform(predict)
        y = normal.inverse_transform(y)

        if i == 0:
            all_pred = predict
            all_y = y
        else:
            all_pred = torch.cat((all_pred, predict), 0)
            all_y = torch.cat((all_y, y), 0)
        num_example += x.shape[0]
        assert num_example == len(all_pred), f'num_example: {num_example}, all_pred1: {all_pred.shape}'

        if i % log_interval == 0:
            pass
        i+=1
    return  

def test(model, val_iterator1, criterion, normal, args):
    with torch.no_grad():
        model.eval()

        num_example = 0
        MAE_LOSS = torch.nn.L1Loss()
        MSE_LOSS = torch.nn.MSELoss()
        for i, (x, y) in enumerate(val_iterator1):
            x = x.to(args.device)
            y = y.to(args.device)
            logits = model(x)
            # predict = normal.rmse_transform(logits)
            # y = normal.rmse_transform(y)
            predict = normal.inverse_transform(logits)
            y = normal.inverse_transform(y)

            if i == 0:
                all_pred = predict
                all_y = y
                if args.dataset_name in ['complaint19_3h', 'complaint10_3h', 'complaint9_3h', 'nyctaxi2014']:
                    all_pred0 = predict[:,:,:,0]
                    all_pred1 = predict[:,:,:,1]
                    all_y0 = y[:,:,:,0]
                    all_y1 = y[:,:,:,1]
            else:
                all_pred = torch.cat((all_pred, predict), 0)
                all_y = torch.cat((all_y, y), 0)
                if args.dataset_name in ['complaint19_3h', 'complaint10_3h', 'complaint9_3h', 'nyctaxi2014']:
                    all_pred0 = torch.cat((all_pred0, predict[:,:,:,0]), 0)
                    all_pred1 = torch.cat((all_pred1, predict[:,:,:,1]), 0)
                    all_y0 = torch.cat((all_y0, y[:,:,:,0]), 0)
                    all_y1 = torch.cat((all_y1, y[:,:,:,1]), 0)
            num_example += x.shape[0]
            assert num_example == len(all_pred), f'num_example: {num_example}, all_pred1: {all_pred.shape}'

        if args.dataset_name in ['complaint19_3h', 'complaint10_3h', 'complaint9_3h', 'nyctaxi2014']:
            rmse0 = torch.sqrt(MSE_LOSS(all_pred0, all_y0))
            rmse1 = torch.sqrt(MSE_LOSS(all_pred1, all_y1))
            mae0 = MAE_LOSS(all_pred0, all_y0)
            mae1 = MAE_LOSS(all_pred1, all_y1)
            print(f'test rmse0: {rmse0}, mae0: {mae0}, rmse1: {rmse1}, mae1: {mae1}')
        rmse = torch.sqrt(MSE_LOSS(all_pred, all_y))
        mae = MAE_LOSS(all_pred, all_y)
        print(f'val_loss_Masked_RMSE: {masked_rmse(all_pred, all_y, 0.0)}, val_Masked_MAE: {masked_mae(all_pred, all_y, 0.0)}')
        print(f'val_loss_Masked_MAPE: {masked_mape(all_pred, all_y, 0.0)}')
        print(f'val_loss_RMSE: {rmse}, val_mae: {mae}')
        rmse_list, mae_list = [], []
        _pred = all_pred.cpu().numpy()
        _y = all_y.cpu().numpy()
        for i in range(all_pred.shape[-1]):
            rmse_list.append(torch.sqrt(MSE_LOSS(all_pred[...,i], all_y[...,i])))
            mae_list.append(MAE_LOSS(all_pred[...,i], all_y[...,i]))
        _rmse_list = [i.item() for i in rmse_list]
        _mae_list = [i.item() for i in mae_list]
        print(f'mean val_loss_RMSE: {np.mean(_rmse_list)}, mean val_mae: {np.mean(_mae_list)}')
    return rmse, _rmse_list, _mae_list


def main(args):
    if args.seed != 0:
        print(f'fix seed as: {args.seed}')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda:
            torch.cuda.manual_seed(args.seed)
    # device = '1'
    device = torch.device(args.device)
    train_dataset, normal = get_dataset(args.dataset_name, 'train', args.out_channel, args.normal_flag)
    val_dataset, _ = get_dataset(args.dataset_name, 'val', args.out_channel, args.normal_flag)
    test_dataset, _ = get_dataset(args.dataset_name, 'test', args.out_channel, args.normal_flag)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)

    if args.dataset_name.startswith('complaint'):
        in_num_nodes = 64
        in_dim = 19
    elif args.dataset_name.startswith('nyctaxi'):
        in_num_nodes = 200
        in_dim = 4
    else:
        assert 1==0, 'wrong dataset name.'

    model = ttnet(args.pmt_dropout, in_dim=in_dim, out_dim=args.out_channel, hid_dim=args.embedding_size, \
        ts_depth_spa=args.ts_depth_spa, ts_depth_tem=args.ts_depth_tem)

    model = model.to(args.device)
 
    if os.path.exists(args.resume_dir):
        save_path = args.resume_dir
        model_data = torch.load(args.resume_dir, map_location=args.device)
        model.load_state_dict(model_data['model'])
        epoch = model_data['epoch']
        lowest_val_loss_count = model_data['lowest_val_loss_count']
        lowest_val_loss = model_data['lowest_val_loss']
        if args.ft_flag:
            lowest_val_loss_count = 0
            lowest_val_loss = np.inf
        print(f'load model from {args.resume_dir}')
    else:
        lowest_val_loss = np.inf
        lowest_val_loss_count = 0
        resume_epoch = 0
        time_stamp = int(time.time())+random.randint(1, 100)
        save_path=f'{args.save_dir}/{args.dataset_name}_outcnl{args.out_channel}_{time_stamp}.pt'
        print(f'no checkpoint available, train from scratch')
 
    print(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters', params)
    criterion = get_criterion(args.loss_type, args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    test_loss = test(model, test_data_loader, criterion, normal, args)[0]
    print(f'test loss: {test_loss}')
    time_start=time.time()
    for epoch_i in range(args.epoch):
        train(model, optimizer, train_data_loader, criterion, normal, args)
        val_loss = test(model, val_data_loader, criterion, normal, args)[0]
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            print(f'lowest validation loss: {lowest_val_loss}')
            torch.save({
                'epoch':epoch_i+1,
                'lowest_val_loss_count':lowest_val_loss_count,
                'lowest_val_loss':lowest_val_loss,
                'optimizer':optimizer.state_dict(),
                'model':model.state_dict()}, save_path)
            lowest_val_loss_count = 0
            print(f'save model at: {save_path}')
        else:
            lowest_val_loss_count += 1
        print('epoch:', epoch_i)
        if lowest_val_loss_count > args.early_stop_patience:
            print(f'there are already {lowest_val_loss_count} epochs without performance improvement, stop here.')
            print(f'epoch: {epoch_i}, val_loss: {lowest_val_loss}')
            break
    model_data = torch.load(save_path, map_location=args.device)
    model.load_state_dict(model_data['model'])
    test_loss, test_rmse, test_mae = test(model, test_data_loader, criterion, normal, args)
    print(f'test loss: {test_loss}')
    time_end=time.time()
    print('time cost %.4f s' %float(time_end-time_start))
    best_epoch = model_data['epoch']
    print(f'epoch with best val loss: {best_epoch}, save path: {save_path}')
    if not args.out_dir == '':
        df = pd.DataFrame({'epoch': best_epoch, 'time cost' : round(float(time_end-time_start), 1), \
            'test RMSE': [round(i, 5) for i in test_rmse], 'test MAE': [round(i, 5) for i in test_mae]}, index=[args.dataset_name for _ in test_rmse])
        df.to_csv('output/'+args.out_dir, mode='a', header=False)
        print(f'save to {args.out_dir}.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='complaint19_3h', choices=[
        'complaint19_3h', 'complaint10_3h', 'complaint9_3h', 'nyctaxi2014'])
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embedding_size', type=int, default=32)
    parser.add_argument('--out_channel', type=int, default=12)
    parser.add_argument('--ts_depth_spa', type=int, default=2)
    parser.add_argument('--ts_depth_tem', type=int, default=2)
    parser.add_argument('--early_stop_patience', type=int, default=40)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--pmt_dropout', type=float, default=0)
    parser.add_argument('--pmt_init_type', type=str, default='xnor', choices=['xuni','xnor', 'kuni', 'knor', 'nor', 'uni', 'nor', 'none'])
    parser.add_argument('--normal_flag', type=int, default=1)
    parser.add_argument('--device', default='cuda')#cuda:0
    parser.add_argument('--save_dir', default='model_para')
    parser.add_argument('--loss_type', type=str, default='rmse+mae', choices=['mae', 'mse', 'rmse', 'mse+mae', 'rmse+mae'])
    parser.add_argument('--resume_dir', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='out_fulltrain.csv')
    parser.add_argument('--basic_state_dict', type=str, default='')
    parser.add_argument('--data_shuffle', action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)