import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import random
from sklearn.decomposition import PCA
from collections import defaultdict
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import argparse
from MDPAttack.multinomial_diffusion.diffusion_utils.diffusion_multinomial import *

from MDPAttack.models.LFM.LFM import LFM
from MDPAttack.models.NeuMF.NeuMF import NeuMF
from MDPAttack.utils.functions import *
from MDPAttack.train_MD import MDPModelTrainer

np.random.seed(seed=0)
torch.manual_seed(seed=0)

class MDPAttack(object):
    def __init__(self):
        self.args = self.parse_args()
        self.device = torch.device(f"cuda:{self.args.cuda_id}" if torch.cuda.is_available() else "cpu")

    def get_args_dict(self):
        args = self.args
        return {
            'train_data': f'./MDPAttack/datasets/{args.dataset_name}/train_data.txt',
            'valid_data': f'./MDPAttack/datasets/{args.dataset_name}/valid_data.txt',
            'test_data': f'./MDPAttack/datasets/{args.dataset_name}/test_data.txt',
            'result_dirs': './MDPAttack/models/NeuMF/weights',
            'influences_result': f'./MDPAttack/results/mid_results/{args.dataset_name}_influences_result.pth',
            'saved_md': f'./MDPAttack/results/{args.dataset_name}_MD_model.pth',
            'partial_diffusion_list': f'./MDPAttack/results/mid_results/{args.dataset_name}_par_dif_list.pth',
            'first_FGSM_list': f'./MDPAttack/results/mid_results/{args.dataset_name}_first_FGSM_list.pth',
            'integrated_list': f'./MDPAttack/results/mid_results/{args.dataset_name}_integrated_list.pth',
            'second_FGSM_list': f'./MDPAttack/results/mid_results/{args.dataset_name}_second_FGSM_list.pth',
            'reverse_diffusion_list': f'./MDPAttack/results/mid_results/{args.dataset_name}_rev_dif_list.pth',
        }

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_name', type=str, default='filmtrust')
        parser.add_argument('--num_users', type=int, default=796)
        parser.add_argument('--num_items', type=int, default=2011)
        parser.add_argument('--num_classes', type=int, default=9)

        parser.add_argument('--recommender', type=str, default='LFM')
        parser.add_argument('--latent_factors', type=int, default=64)
        parser.add_argument('--learning_rate', type=float, default=0.05)
        parser.add_argument('--num_epochs', type=int, default=2000)
        parser.add_argument('--lambda_value', type=float, default=0.0003)

        parser.add_argument('--num_template', type=int, default=8)
        parser.add_argument('--num_iterations_FGSM', type=int, default=2)
        parser.add_argument('--target_item_index_list', type=list, default=[1371, 1521, 1066, 124, 329])
        parser.add_argument('--reshape_value', type=int, default=2013)
        parser.add_argument('--data_shape', nargs=3, type=int, default=[1, 33, 61])
        parser.add_argument('--poisoning_rate', type=float, default=0.05)
        parser.add_argument('--cuda_id', type=int, default=1)

        parser.add_argument("--mode", default='client')
        parser.add_argument("--host", type=bool, default="localhost")
        parser.add_argument("--port", default=35991)
        parser.add_argument('--num_workers', type=int, default=4)

        return parser.parse_args()

    def select_template_profiles(self):
        args = self.args
        rec = args.recommender
        args_dict = self.get_args_dict()

        user_train_ids, item_train_ids, train_data = read_data(args.num_users, args.num_items, path=args_dict['train_data'])
        train_data = train_data.to(self.device)
        nonzero_train_indices = torch.nonzero(train_data)
        lr = args.learning_rate

        if rec == 'LFM':
            model_LFM = LFM(args.num_users, args.num_items, args.latent_factors, device=self.device)
            model_LFM.train_model(args_dict, lr, args.num_epochs, args.lambda_value)
            P_matrix, Q_matrix, ratings_pred, loss_regularization = model_LFM(args.lambda_value)

            atk_loss = calc_multi_atk_loss(train_data, args.target_item_index_list, ratings_pred,
                                           args.num_users)
            params = [p for p in model_LFM.parameters() if p.requires_grad]
            v = list(grad(atk_loss, params, create_graph=True))
            h_estimate = v.copy()
        elif rec == 'NeuMF':
            model = NeuMF(train_data, self.device, args.num_users, args.num_items, args_dict)
            model_NeuMF, ratings_pred = model.pred()

            atk_loss = calc_multi_atk_loss(train_data, args.target_item_index_list, ratings_pred,
                                           args.num_users)
            params = [p for p in model_NeuMF.parameters() if p.requires_grad]
            v = list(grad(atk_loss, params, create_graph=True))
            h_estimate = v.copy()

        train_ratings = train_data[user_train_ids, item_train_ids]
        criterion = nn.MSELoss().to(self.device)
        if rec == 'LFM':
            inverse_hvp = calc_IHVP_LFM(nonzero_train_indices, train_ratings, ratings_pred, P_matrix, Q_matrix,
                                        args.lambda_value, criterion, params, v, h_estimate)

            influences_list = calc_influence_single_LFM(args.num_users, nonzero_train_indices, train_data, ratings_pred,
                                                        criterion, P_matrix, Q_matrix, params, inverse_hvp, display_progress)

        elif rec == 'NeuMF':
            _, inverse_hvp = calc_IHVP_NeuMF(nonzero_train_indices, train_ratings, ratings_pred, criterion, params, v,
                                             h_estimate)
            influences_list = calc_influence_single_NeuMF(args.num_users, nonzero_train_indices, train_data, ratings_pred,
                                                          criterion, params, inverse_hvp, display_progress)

        influences_result = [(idx, value) for idx, value in influences_list if value > 0]
        torch.save(influences_result, args_dict['influences_result'])
        helpful_list = [t[0] for t in influences_result]
        template_ids = helpful_list[:args.num_template]
        print(f'template ids: {template_ids}\n')

        self.template_ids = template_ids

    def train_MD(self):
        args = self.args
        model = MDPModelTrainer(dataset_name=args.dataset_name, device_id=args.cuda_id)
        model_MD = model.train_and_save_model().cpu()

    def add_noise(self):
        args = self.args
        args_dict = self.get_args_dict()
        model = MDPModelTrainer(dataset_name=args.dataset_name, device_id=args.cuda_id)
        model_MD = model.load_trained_model()
        partial_diffusion_list = []
        _, _, train_data = read_data(args.num_users, args.num_items, path=args_dict['train_data'])

        for influential_instance_index in self.template_ids:
            influential_instance_perturbed = partial_diffusion(args.dataset_name, train_data, influential_instance_index,
                                                               index_to_log_onehot, args.num_classes, model_MD)
            partial_diffusion_list.append(influential_instance_perturbed)

        torch.save(partial_diffusion_list, args_dict['partial_diffusion_list'])
        return partial_diffusion_list

    def calc_influential_instance_FGSM(self, influential_instance_index, influential_instance, train_data):
        args = self.args
        rec = args.recommender
        args_dict = self.get_args_dict()
        influential_instance = influential_instance.to(self.device)

        t = 0.0001
        alpha = 1
        atk_loss_list = []
        for i in range(args.num_iterations_FGSM):

            train_data[influential_instance_index, :] = influential_instance

            if rec == 'LFM':
                model_LFM = LFM(args.num_users, args.num_items, args.latent_factors, device=self.device)
                model_LFM.train_model(args_dict, args.learning_rate, args.num_epochs, args.lambda_value)
                P_matrix, Q_matrix, ratings_pred, loss_regularization = model_LFM(args.lambda_value)
            elif rec == 'NeuMF':
                model = NeuMF(train_data, self.device, args.num_users, args.num_items, args_dict)
                model_NeuMF, ratings_pred = model.pred()

            train_data = train_data.to(self.device)
            atk_loss = calc_multi_atk_loss(train_data, args.target_item_index_list, ratings_pred,
                                           args.num_users)
            atk_loss_list.append(atk_loss)

            if atk_loss > t:
                t = atk_loss
                influential_instance_FGSM = influential_instance.detach_()

            criterion = nn.MSELoss().to(self.device)
            nonzero_train_indices = torch.nonzero(train_data)
            user_train_ids = nonzero_train_indices[:, 0]
            item_train_ids = nonzero_train_indices[:, 1]
            train_ratings = train_data[user_train_ids, item_train_ids]

            influential_instance.requires_grad_()
            influential_instance_pred = ratings_pred[influential_instance_index]
            per_loss1 = criterion(influential_instance_pred, influential_instance)

            if rec == 'LFM':
                per_loss = per_loss1 + loss_regularization
                params = [p for p in model_LFM.parameters() if p.requires_grad]
                attack_grad = list(grad(atk_loss, params, create_graph=True))
                h_estimate = attack_grad.copy()
                inverse_hvp = calc_IHVP_LFM(nonzero_train_indices, train_ratings, ratings_pred, P_matrix, Q_matrix,
                                            args.lambda_value, criterion, params, attack_grad,
                                            h_estimate)

            elif rec == 'NeuMF':
                per_loss = per_loss1
                params = [p for p in model_NeuMF.parameters() if p.requires_grad]
                attack_grad = list(grad(atk_loss, params, create_graph=True))
                h_estimate = attack_grad.copy()
                _, inverse_hvp = calc_IHVP_NeuMF(nonzero_train_indices, train_ratings, ratings_pred, criterion, params,
                                                 attack_grad, h_estimate)

            I_pert_atk = calc_I_pert_atk(per_loss, params, influential_instance, inverse_hvp, args.num_users,
                                         do_not_sum_up=True)

            influential_instance = torch.clamp(influential_instance + alpha * torch.sign(I_pert_atk), 0,
                                               torch.max(train_data)).detach()
            print(f'----The {i + 1}th iteration is over----')

        return influential_instance_FGSM, atk_loss_list

    def _first_fgsm_worker(self, influential_instance_index):
        args = self.args
        args_dict = self.get_args_dict()
        _, _, train_data = read_data(args.num_users, args.num_items, path=args_dict['train_data'])
        influential_instance = train_data[influential_instance_index].to(torch.float32)
        influential_instance_FGSM, atk_loss_list = self.calc_influential_instance_FGSM(
            influential_instance_index, influential_instance, train_data
        )
        influential_instance_FGSM = influential_instance_FGSM.cpu().to(torch.int64)
        atk_list = [B.item() for B in atk_loss_list]
        return influential_instance_FGSM, atk_list

    def first_FGSM(self):
        args = self.args
        args_dict = self.get_args_dict()
        first_FGSM_list = []
        atk_list = []
        print(f'\n---------------The first FGSM iteration---------------')

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(self._first_fgsm_worker, tid): idx
                for idx, tid in enumerate(self.template_ids)
            }
            results = [None] * len(self.template_ids)
            for future in as_completed(futures):
                idx = futures[future]
                influential_instance_FGSM, atk_loss = future.result()
                results[idx] = (influential_instance_FGSM, atk_loss)
                print(f'----------The {idx + 1}th template is over----------')

        for influential_instance_FGSM, atk_loss in results:
            first_FGSM_list.append(influential_instance_FGSM)
            atk_list.append(atk_loss)

        torch.save(first_FGSM_list, args_dict['first_FGSM_list'])
        return first_FGSM_list

    def sample_integrate(self):
        args = self.args
        args_dict = self.get_args_dict()

        partial_diffusion_list = self.add_noise()
        A_list = []
        for influential_instance_perturbed in partial_diffusion_list:
            influential_instance_perturbed_index = log_onehot_to_index(influential_instance_perturbed)
            if args.dataset_name == 'filmtrust':
                A = influential_instance_perturbed_index.view(-1)[:-2]
            elif args.dataset_name == 'ml-100k':
                A = influential_instance_perturbed_index.view(-1)[:-1]
            elif args.dataset_name == 'ml-1m':
                A = influential_instance_perturbed_index.view(-1)
            elif args.dataset_name == 'ml-10m':
                A = influential_instance_perturbed_index.view(-1)[:-20]
            A_list.append(A)
        B_list = self.first_FGSM()

        integrated_list = []
        for i in range(len(A_list)):
            influential_instance_integrated = integrate_instances(A_list[i], B_list[i], PCA, args.dataset_name)
            integrated_list.append(influential_instance_integrated)

        torch.save(integrated_list, args_dict['integrated_list'])
        return integrated_list

    def _second_fgsm_worker(self, idx, influential_instance_tensor):
        args = self.args
        args_dict = self.get_args_dict()
        _, _, train_data = read_data(args.num_users, args.num_items, path=args_dict['train_data'])
        influential_instance = influential_instance_tensor.to(torch.float32)
        influential_instance_index = self.template_ids[idx]
        influential_instance_FGSM, atk_loss_list = self.calc_influential_instance_FGSM(
            influential_instance_index, influential_instance, train_data
        )
        influential_instance_FGSM = influential_instance_FGSM.cpu().to(torch.int64)
        atk_list = [B.item() for B in atk_loss_list]
        return idx, influential_instance_FGSM, atk_list

    def second_FGSM(self):
        args = self.args
        args_dict = self.get_args_dict()
        integrated_list = self.sample_integrate()
        influential_integrated_list = []
        for instance in integrated_list:
            A = instance.reshape((args.reshape_value,))
            if args.dataset_name == 'filmtrust':
                A1 = A[:-2, ]
            elif args.dataset_name == 'ml-100k':
                A1 = A[:-1, ]
            elif args.dataset_name == 'ml-1m':
                A1 = A
            elif args.dataset_name == 'ml-10m':
                A1 = A[:-20, ]
            influential_integrated_list.append(A1)

        second_FGSM_list = []
        atk_list = []
        print(f'\n---------------The second FGSM iteration---------------')

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(self._second_fgsm_worker, idx, influential_integrated_list[idx]): idx
                for idx in range(len(self.template_ids))
            }
            results = [None] * len(self.template_ids)
            for future in as_completed(futures):
                idx, influential_instance_FGSM, atk_loss = future.result()
                results[idx] = (influential_instance_FGSM, atk_loss)
                print(f'----------The {idx + 1}th template is over----------')

        for influential_instance_FGSM, atk_loss in results:
            second_FGSM_list.append(influential_instance_FGSM)
            atk_list.append(atk_loss)

        torch.save(second_FGSM_list, args_dict['second_FGSM_list'])
        return second_FGSM_list

    def reverse_diffusion(self):
        args = self.args
        args_dict = self.get_args_dict()
        second_FGSM_list = self.second_FGSM()

        integrated_list_1 = []
        for i in range(len(second_FGSM_list)):
            A = second_FGSM_list[i]
            if args.dataset_name == 'filmtrust':
                D = A[-2:].clone()
                D_1 = torch.cat([A, D])
                D_2 = D_1.view(1, 1, 33, 61).to(torch.int64)
            elif args.dataset_name == 'ml-100k':
                D = A[-1:].clone()
                D_1 = torch.cat([A, D])
                D_2 = D_1.view(1, 1, 33, 51).to(torch.int64)
            elif args.dataset_name == 'ml-1m':
                D_2 = A.view(1, 1, 34, 109).to(torch.int64)
            elif args.dataset_name == 'ml-10m':
                D = A[-20:].clone()
                D_1 = torch.cat([A, D])
                D_2 = D_1.view(1, 1, 33, 985).to(torch.int64)
            D_3 = D_2.to(torch.int64)
            integrated_list_1.append(D_3)

        shape = tuple(args.data_shape)
        num_timesteps = 100  # 20

        print(f'\n---------Begin reverse diffusion---------')
        reverse_diffusion_list = []

        model = MDPModelTrainer(dataset_name=args.dataset_name, device_id=args.cuda_id)
        model_MD = model.load_trained_model()

        i = 1
        for influential_instance_integrated in integrated_list_1:
            print(f'-----The {i}th instance-----')
            i += 1
            reverse_diffusion_instances = inverse_sampling(index_to_log_onehot, influential_instance_integrated,
                                                           args.num_classes, shape, model_MD, num_timesteps,
                                                           log_onehot_to_index, args.dataset_name)
            reverse_diffusion_list.append(reverse_diffusion_instances)

        torch.save(reverse_diffusion_list, args_dict['reverse_diffusion_list'])
        return reverse_diffusion_list

    def select_profiles(self):
        args = self.args
        args_dict = self.get_args_dict()
        reverse_diffusion_list = self.reverse_diffusion()

        A = []
        for i in range(len(reverse_diffusion_list)):
            tensor = reverse_diffusion_list[i].float()
            result_tensor = tensor[-1]
            result_tensor = result_tensor.view(1, -1).float()
            poisoning_instances = obtain_poisoning_instances(tensor, result_tensor)
            A.append(poisoning_instances)

        poisoning_num = -int(math.ceil((args.num_users * args.poisoning_rate) / len(self.template_ids)))
        selected_profiles = []
        for sublist in A:
            selected_profiles.append(sublist[poisoning_num:])

        return selected_profiles

    def poisoning_attack(self):
        args = self.args
        rec = args.recommender
        args_dict = self.get_args_dict()
        selected_profiles = self.select_profiles()
        _, _, train_data = read_data(args.num_users, args.num_items, path=args_dict['train_data'])

        all_poisoning_instances = torch.tensor(np.vstack(selected_profiles))
        all_poisoning_instances[:, target_item_index_list] = torch.max(train_data)
        poisoned_train_data = torch.cat((train_data, all_poisoning_instances))
        nonzero_indices = torch.nonzero(poisoned_train_data)
        user_train_ids = nonzero_indices[:, 0]
        item_train_ids = nonzero_indices[:, 1]
        num_users_updated = poisoned_train_data.size(0)

        print('\n----------Model training after attack----------')
        if rec == 'LFM':
            model_LFM = LFM(num_users_updated, args.num_items, args.latent_factors, device=self.device)
            model_LFM.retrain_model(poisoned_train_data, args_dict, args.learning_rate, args.num_epochs, args.lambda_value)
            _, _, ratings_pred, _ = model_LFM(args.lambda_value)
        elif rec == 'NeuMF':
            model = NeuMF(poisoned_train_data, self.device, num_users_updated, args.num_items, args_dict)
            _, ratings_pred = model.pred()

        ratings_indices_dict, filtered_indices = obtain_filtered_indices(user_train_ids, item_train_ids,
                                                                         args.target_item_index_list, defaultdict, args.num_items)
        Top_K_list, rec_list = obtain_rec_list(ratings_pred, filtered_indices, defaultdict, args.num_users,
                                               ratings_indices_dict, K=10)

        print('----------Model training before attack----------')
        if rec == 'LFM':
            model_LFM = LFM(args.num_users, args.num_items, args.latent_factors, device=self.device)
            model_LFM.train_model(args_dict, args.learning_rate, args.num_epochs, args.lambda_value)
            _, _, ratings_pred, _ = model_LFM(args.lambda_value)
        elif rec == 'NeuMF':
            model = NeuMF(train_data, self.device, args.num_users, args.num_items, args_dict)
            _, ratings_pred = model.pred()
        user_train_ids, item_train_ids, _ = read_data(args.num_users, args.num_items, path=args_dict['train_data'])
        ratings_indices_dict, filtered_indices = obtain_filtered_indices(user_train_ids, item_train_ids,
                                                                         args.target_item_index_list, defaultdict, args.num_items)
        origin_Top_K_list, origin_rec_list = obtain_rec_list(ratings_pred, filtered_indices, defaultdict, args.num_users,
                                                             ratings_indices_dict, K=10)

        ER_value, RS_value, HR_value, NDCG_value, Recall_value = eval_metric(args.target_item_index_list, Top_K_list,
                                                                             origin_Top_K_list, rec_list,
                                                                             origin_rec_list, math, args.num_users)
        print('----------Evaluation metrics----------')
        print(f'ER_value:{ER_value}')
        print(f'RS_value:{RS_value}')
        print(f'HR_value:{HR_value}')
        print(f'NDCG_value:{NDCG_value}')
        print(f'Recall_value:{Recall_value}')
