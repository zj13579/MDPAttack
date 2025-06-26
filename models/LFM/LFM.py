import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from MDPAttack.utils.functions import read_data

class LFM(nn.Module):
    def __init__(self, num_users, num_items, latent_factors, device):
        super(LFM, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_factors = latent_factors
        self.device = device
        self.user_indices = torch.arange(num_users).to(device)
        self.item_indices = torch.arange(num_items).to(device)

        self.user_embeddings = nn.Embedding(num_users, latent_factors)
        self.item_embeddings = nn.Embedding(num_items, latent_factors)

    def forward(self, lambda_value):
        user_latent = self.user_embeddings(self.user_indices)
        item_latent = self.item_embeddings(self.item_indices)
        ratings_pred = torch.matmul(user_latent, item_latent.t())
        user_latent_l2_norms = (torch.norm(user_latent, dim=1)) ** 2
        sum_of_user_l2_norms = torch.sum(user_latent_l2_norms)
        item_latent_l2_norms = (torch.norm(item_latent, dim=1)) ** 2
        sum_of_item_l2_norms = torch.sum(item_latent_l2_norms)
        loss_regularization = lambda_value * (sum_of_user_l2_norms + sum_of_item_l2_norms)

        return user_latent, item_latent, ratings_pred, loss_regularization


    def train_model(self, args_dict, lr, num_epochs, lambda_value):
        user_train_ids, item_train_ids, train_data = read_data(self.num_users, self.num_items, path=args_dict['train_data'])
        user_test_ids, item_test_ids, test_data = read_data(self.num_users, self.num_items, path=args_dict['test_data'])

        self.to(self.device)
        train_data = train_data.to(self.device)
        test_data = test_data.to(self.device)
        criterion = nn.MSELoss().to(self.device)
        optimizer = optim.Adam(self.parameters(), lr)

        train_ratings = train_data[user_train_ids, item_train_ids]
        test_ratings = test_data[user_test_ids, item_test_ids]

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            _, _, ratings_pred, loss_regularization = self(lambda_value)
            train_ratings_pred = ratings_pred[user_train_ids, item_train_ids].to(torch.float64)
            train_loss = criterion(train_ratings_pred, train_ratings) + loss_regularization
            train_loss.backward()
            optimizer.step()

            if (epoch + 1) % 200 == 0:
                test_ratings_pred = ratings_pred[user_test_ids, item_test_ids].to(torch.float64)
                test_loss = criterion(test_ratings_pred, test_ratings)
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], train_Loss: {train_loss.item()}, test_loss: {test_loss.item()}")


    def retrain_model(self, poisoned_train_data, args_dict, lr, num_epochs, lambda_value):

        nonzero_indices = torch.nonzero(poisoned_train_data)
        user_train_ids = nonzero_indices[:, 0]
        item_train_ids = nonzero_indices[:, 1]

        user_test_ids, item_test_ids, test_data = read_data(self.num_users, self.num_items, path=args_dict['test_data'])

        self.to(self.device)
        poisoned_train_data = poisoned_train_data.to(self.device)
        test_data = test_data.to(self.device)
        criterion = nn.MSELoss().to(self.device)
        optimizer = optim.Adam(self.parameters(), lr)

        train_ratings = poisoned_train_data[user_train_ids, item_train_ids]
        test_ratings = test_data[user_test_ids, item_test_ids]

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            _, _, ratings_pred, loss_regularization = self(lambda_value)
            train_ratings_pred = ratings_pred[user_train_ids, item_train_ids].to(torch.float64)
            train_loss = criterion(train_ratings_pred, train_ratings) + loss_regularization
            train_loss.backward()
            optimizer.step()

            if (epoch + 1) % 200 == 0:
                test_ratings_pred = ratings_pred[user_test_ids, item_test_ids].to(torch.float64)
                test_loss = criterion(test_ratings_pred, test_ratings)
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], train_Loss: {train_loss.item()}, test_loss: {test_loss.item()}")