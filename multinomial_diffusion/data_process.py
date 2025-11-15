import math
import torch
import numpy as np
from torch.utils.data import DataLoader

dataset_choices = {'filmtrust', 'ml-100k', 'ml-1m'}


def add_data_args(parser):

    # Data params
    parser.add_argument('--dataset', type=str, default='ml-100k',
                        choices=dataset_choices)

    # Train params
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=eval, default=False)
    parser.add_argument('--augmentation', type=str, default=None)


def get_data_id(args):
    return '{}'.format(args.dataset)


def get_data(args):
    assert args.dataset in dataset_choices

    if args.dataset == 'filmtrust':
        data_shape = (1, 33, 61)
        num_classes = 9

        num_users = 796
        num_items = 2011

        rating_matrix = np.zeros((num_users, num_items))

        with open('MDPAttack/datasets/filmtrust/train_data.txt', "r") as file:
            for line in file:
                user, item, rating = map(str, line.strip().split())
                user = int(user)
                item = int(item)
                rating = float(rating)
                rating_matrix[user, item] = rating

        last_two_columns = rating_matrix[:, -2:]
        rating_matrix = np.hstack((rating_matrix, last_two_columns))

        data = torch.from_numpy(rating_matrix).int()
        data = data.view(796, 1, 33, 61)

        dataset_train = data
        train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)

    if args.dataset == 'ml-100k':
        data_shape = (1, 33, 51)
        num_classes = 6

        num_users = 943
        num_items = 1682

        rating_matrix = np.zeros((num_users, num_items))

        with open('MDPAttack/datasets/ml-100k/train_data.txt', "r") as file:
            for line in file:
                user, item, rating = map(str, line.strip().split())
                user = int(user)
                item = int(item)
                rating = float(rating)
                rating_matrix[user, item] = rating

        last_one_columns = rating_matrix[:, -1:]
        rating_matrix = np.hstack((rating_matrix, last_one_columns))

        data = torch.from_numpy(rating_matrix).int()
        data = data.view(943, 1, 33, 51)

        dataset_train = data
        train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)

    if args.dataset == 'ml-1m':
        data_shape = (1, 34, 109)
        num_classes = 6

        num_users = 6040
        num_items = 3706

        rating_matrix = np.zeros((num_users, num_items))

        with open('MDPAttack/datasets/ml-1m/train_data.txt', "r") as file:
            for line in file:
                user, item, rating = map(str, line.strip().split())
                user = int(user)
                item = int(item)
                rating = float(rating)
                rating_matrix[user, item] = rating

        data = torch.from_numpy(rating_matrix).int()
        data = data.view(6040, 1, 34, 109)

        dataset_train = data
        train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)

    if args.dataset == 'ml-10m':
        data_shape = (1, 33, 985)
        num_classes = 6
    
        num_users = 35785
        num_items = 32525
    
        rating_matrix = np.zeros((num_users, num_items))
    
        with open('MDPAttack/datasets/ml-10m/train_data.txt', "r") as file:
            for line in file:
                user, item, rating = map(str, line.strip().split())
                user = int(user)
                item = int(item)
                rating = float(rating)
                rating_matrix[user, item] = rating
    
        last_one_columns = rating_matrix[:, -20:]
        rating_matrix = np.hstack((rating_matrix, last_one_columns))
    
        data = torch.from_numpy(rating_matrix).int()
        data = data.view(35785, 1, 33, 985)
    
        dataset_train = data
        train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)

    return train_loader, data_shape, num_classes
