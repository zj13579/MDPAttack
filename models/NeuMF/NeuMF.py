import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from MDPAttack.models.NeuMF.model import Neural_MF, GMF, MLP
from MDPAttack.models.NeuMF.utils import date, predict_mse, NCFDataset

def train(train_dataloader, valid_dataloader, model, device, learning_rate, l2_regularization,
          learning_rate_decay, train_epochs, model_path):
    print(f'{date()}## Start the training!')
    train_mse = predict_mse(model, train_dataloader, device)
    valid_mse = predict_mse(model, valid_dataloader, device)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, learning_rate_decay)

    best_loss = 100
    for epoch in range(train_epochs):
        model.train()
        total_loss, total_samples = 0, 0
        for batch in train_dataloader:
            user_id, item_id, ratings = [i.to(device) for i in batch]
            predict = model(user_id, item_id)
            predict = predict.view_as(ratings)
            loss = F.mse_loss(predict, ratings, reduction='mean')
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(predict)
            total_samples += len(predict)

        lr_sch.step()
        model.eval()
        valid_mse = predict_mse(model, valid_dataloader, device)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


def test(dataloader, model):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_loss = predict_mse(model, dataloader, next(model.parameters()).device)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_loss:.6f}, time used {end_time - start_time:.0f} seconds.")


class NeuMF(nn.Module):
    def __init__(self, train_data_origin, device, num_users, num_items, args_dict, batch_size=128, mf_dim=64, mlp_layers=[32, 16, 8], learning_rate=0.01,
                 l2_regularization=0.001, learning_rate_decay=0.99, seed=0):
        super(NeuMF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.args_dict = args_dict
        self.result_dirs = args_dict['result_dirs']
        self.mf_dim = mf_dim
        self.mlp_layers = mlp_layers
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.learning_rate_decay = learning_rate_decay
        self.seed = seed
        self.device = device

        self.nonzero_indices = torch.nonzero(train_data_origin, as_tuple=False)
        self.user_ids = self.nonzero_indices[:, 0]
        self.item_ids = self.nonzero_indices[:, 1]
        self.ratings = train_data_origin[self.user_ids, self.item_ids]

        self.train_data = pd.DataFrame({
            0: self.user_ids.cpu().numpy(),
            1: self.item_ids.cpu().numpy(),
            2: self.ratings.cpu().numpy()
        })

        self.train_data.columns = ['userID', 'itemID', 'rating']

        self.train_dataset = NCFDataset(self.train_data)
        self.batch_size = batch_size
        self.train_dlr = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        np.random.seed(seed=self.seed)
        torch.manual_seed(seed=self.seed)

        self.valid_data = pd.read_csv(self.args_dict['valid_data'], sep='\t', header=None)
        self.valid_data.columns = ['userID', 'itemID', 'rating']
        self.test_data = pd.read_csv(self.args_dict['test_data'], sep='\t', header=None)
        self.test_data.columns = ['userID', 'itemID', 'rating']
        self.valid_dataset = NCFDataset(self.valid_data)
        self.test_dataset = NCFDataset(self.test_data)
        self.valid_dlr = DataLoader(self.valid_dataset)
        self.test_dlr = DataLoader(self.test_dataset)

    def forward(self):

        os.makedirs(self.result_dirs, exist_ok=True)
        print(f'############ Pretrain GMF ###########################')
        train_epochs = 3
        model_GMF = GMF(self.num_users, self.num_items, self.result_dirs, self.mf_dim).to(self.device)
        train(self.train_dlr, self.valid_dlr, model_GMF, self.device, self.learning_rate, self.l2_regularization,
              self.learning_rate_decay, train_epochs, self.result_dirs+'/GMF.pt')
        test(self.test_dlr, torch.load(self.result_dirs+'/GMF.pt', weights_only=False))

        print(f'############ Pretrain MLP ###########################')
        train_epochs = 5
        model_MLP = MLP(self.num_users, self.num_items, self.result_dirs, self.mlp_layers).to(self.device)
        train(self.train_dlr, self.valid_dlr, model_MLP, self.device, self.learning_rate, self.l2_regularization,
              self.learning_rate_decay, train_epochs, self.result_dirs+'/MLP.pt')
        test(self.test_dlr, torch.load(self.result_dirs+'/MLP.pt', weights_only=False))

        print(f'############ Train NeuMF ###########################')
        train_epochs = 3
        model_NeuralMF = Neural_MF(self.num_users, self.num_items, self.result_dirs, self.mf_dim, self.mlp_layers, use_pretrain=True).to(self.device)
        train(self.train_dlr, self.valid_dlr, model_NeuralMF, self.device, self.learning_rate, self.l2_regularization,
              self.learning_rate_decay, train_epochs, self.result_dirs+'/NeuMF.pt')
        test(self.test_dlr, torch.load(self.result_dirs+'/NeuMF.pt', weights_only=False))

    def pred(self):
        model_NeuMF = self()
        user_ids = torch.arange(self.num_users).repeat_interleave(self.num_items).to(self.device)
        item_ids = torch.arange(self.num_items).repeat(self.num_users).to(self.device)
        model_NeuMF = torch.load(self.args_dict['result_dirs'] + '/NeuMF.pt', weights_only=False).to(self.device)
        A = model_NeuMF(user_ids, item_ids)
        ratings_pred = torch.full((self.num_users, self.num_items), 0.0, device=self.device)
        ratings_pred[user_ids, item_ids] = A

        return model_NeuMF, ratings_pred