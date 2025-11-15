'''Some important functions needed during the experiments'''

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

# 1. read data
##############
def read_data(num_users, num_items, path):
    user_ids = []
    item_ids = []
    rating_matrix = np.zeros((num_users, num_items))
    with open(path, "r") as file:
        for line in file:
            user, item, rating = map(str, line.strip().split())
            user = int(user)
            item = int(item)
            rating = float(rating)

            user_ids.append(user)
            item_ids.append(item)

            rating_matrix[user, item] = rating

    user_ids = torch.tensor(user_ids)
    item_ids = torch.tensor(item_ids)

    rating_matrix = torch.from_numpy(rating_matrix)

    return user_ids, item_ids, rating_matrix



# 2.top-K recommendation
########################
def obtain_filtered_indices(user_ids, item_ids, target_item_index_list, defaultdict, num_items):

    ratings_indices = torch.stack((user_ids, item_ids), dim=1)

    A = torch.tensor(target_item_index_list)
    mask = torch.isin(ratings_indices[:, 1], A)
    ratings_indices = ratings_indices[~mask]
    ratings_indices_list = ratings_indices.tolist()

    ratings_indices_dict = defaultdict(list)
    for user_ids, item_ids in ratings_indices_list:
        ratings_indices_dict[user_ids].append(item_ids)

    filtered_list = []
    full_set = set(range(num_items))
    for i in range(len(ratings_indices_dict)):
        sub_list = ratings_indices_dict[i]
        complement_set = full_set - set(sub_list)
        complement_list = list(complement_set)
        filtered_list.append(complement_list)

    filtered_indices = []
    for i in range(len(filtered_list)):
        result = [(i, j) for j in filtered_list[i]]
        filtered_indices.append(result)
    filtered_indices = [item for sublist in filtered_indices for item in sublist]

    return ratings_indices_dict, filtered_indices


def obtain_rec_list(ratings_pred, filtered_indices, defaultdict, num_users, ratings_indices_dict, K=10):
    filtered_ratings_list = [(item[0], item[1], (ratings_pred[item[0], item[1]]).item()) for item in filtered_indices]

    user_ratings_dict = defaultdict(list)
    for user_id, item_id, rating in filtered_ratings_list:
        user_ratings_dict[user_id].append((item_id, rating))

    Top_K_list = []
    for i in range(num_users):
        sorted_result = sorted(user_ratings_dict[i], key=lambda x: x[1], reverse=True)
        sorted_top_K = sorted_result[:K]
        Top_K_list.append(sorted_top_K)

    Top_K_item_list = []
    for i in range(num_users):
        sublist = []
        for j in range(K):
            sublist.append(Top_K_list[i][j][0])

        assert all(item not in ratings_indices_dict[i] for item in
                   sublist), "Every item should be no raings"
        Top_K_item_list.append(sublist)

    rec_list = []
    for i in range(num_users):
        sorted_result = sorted(user_ratings_dict[i], key=lambda x: x[1], reverse=True)
        rec_list.append(sorted_result)

    rec_item_list = []
    for i in range(num_users):
        sublist = []
        length = len(rec_list[i])
        for j in range(length):
            sublist.append(rec_list[i][j][0])

        assert all(item not in ratings_indices_dict[i] for item in
                   sublist), "Every item should be no raings"
        rec_item_list.append(sublist)

    return Top_K_item_list, rec_item_list



# 3.FGSM
########

def atk_func(x):
    log_sigma_x = -torch.log(1 / (1 + torch.exp(-x)))
    return log_sigma_x


def calc_single_atk_loss(train_data, target_item_index, ratings_pred, num_users):

    rate_mask = (train_data.to_dense() == 0).float()
    rate_mask[:, target_item_index] = 1.0

    sorted_rate, _ = torch.sort(ratings_pred * rate_mask + (1 - rate_mask) * -9999, dim=1, descending=True)
    A = ratings_pred[:num_users, target_item_index][:, None]
    A_squeezed = torch.squeeze(A, dim=1)
    B = sorted_rate[:num_users, :10]

    result_list = []
    for num in range(num_users):
        x = atk_func(B[num, :] - A_squeezed[num])
        result_list.append(x)
    one_target_item_loss = torch.stack(result_list, dim=0)

    atk_loss = torch.sum(one_target_item_loss) / num_users
    return atk_loss


def calc_multi_atk_loss(train_data, target_item_index_list, ratings_pred, num_users):

    rate_mask = (train_data.to_dense() == 0).float()
    rate_mask[:, target_item_index_list] = 1.0

    sorted_rate, _ = torch.sort(ratings_pred * rate_mask + (1 - rate_mask) * 0, dim=1, descending=True)
    A = ratings_pred[:num_users, target_item_index_list][:, None]
    A_squeezed = torch.squeeze(A, dim=1)
    B = sorted_rate[:num_users, :10]

    result_lists = []
    for i in range(len(target_item_index_list)):

        A_1 = A_squeezed[:, i]
        result_list = []
        for num in range(num_users):
            x = atk_func(B[num, :] - A_1[num])
            result_list.append(x)
        one_target_item_loss = torch.stack(result_list, dim=0)

        result_lists.append(one_target_item_loss)

    target_item_loss = torch.sum(torch.stack(result_lists, dim=0), dim=0)
    atk_loss = torch.sum(target_item_loss)/num_users

    return atk_loss



# 4. Influence functions
########################
def hessian_vector_product(ys, xs, v, scale=10., do_not_sum_up=True):
    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")

    # First backprop
    grads = torch.autograd.grad(ys, xs, create_graph=True)

    # grads = xs
    assert len(grads) == length
    elemwise_products = [
        grad_elem * v_elem.detach()
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]

    # Second backprop
    if do_not_sum_up:
        grads_with_none = [
            torch.autograd.grad(elemwise_product / scale, x, retain_graph=True,
                                grad_outputs=torch.ones_like(elemwise_product))[0]
            for elemwise_product, x in zip(elemwise_products, xs)
        ]
    else:
        grads_with_none = torch.autograd.grad(elemwise_products / scale, xs)

    return_grads = [
        grad_elem if grad_elem is not None and not torch.isnan(grad_elem).any() else torch.zeros_like(x)
        for x, grad_elem in zip(xs, grads_with_none)
    ]

    # return_grads = [
    #     grad_elem if grad_elem is not None else torch.zeros_like(x)
    #     for x, grad_elem in zip(xs, grads_with_none)
    # ]
    return return_grads


def calc_IHVP_LFM(nonzero_train_indices, train_ratings, ratings_pred, P_matrix, Q_matrix, lambda_value, criterion, params, v, h_estimate, i_epochs=20001, batch_size=2048, pre_norm=-11111, damp=0.001, scale=10):
    np.random.seed(seed=0)
    for j in range(i_epochs):
        r = np.random.choice(len(nonzero_train_indices), size=batch_size, replace=False)
        users = nonzero_train_indices[:, 0][r]
        items = nonzero_train_indices[:, 1][r]
        t = train_ratings[r]
        t = t.to(torch.float32)
        x = ratings_pred[users, items]

        unique_users = torch.unique(users)
        unique_items = torch.unique(items)

        selected_users_rows = P_matrix[unique_users]
        selected_items_rows = Q_matrix[unique_items]

        loss_regularization = lambda_value * (torch.sum(torch.norm(selected_users_rows, dim=1) ** 2) + torch.sum(
            torch.norm(selected_items_rows, dim=1) ** 2))

        loss = criterion(x, t) + loss_regularization
        hv = hessian_vector_product(loss, params, v, scale=10., do_not_sum_up=True)
        h_estimate = [_v + (1 - damp) * _h_e - _hv for _v, _h_e, _hv in zip(v, h_estimate, hv)]

        if j % 500 == 0 and j > 0:
            # cur_estimate = [tensor.detach().numpy() for tensor in h_estimate]
            cur_estimate = [tensor.cpu().detach().numpy() for tensor in h_estimate]
            cur_norm = np.linalg.norm(cur_estimate[0])
            if (j % 2500 == 0):
                print("Inverse HVP epoch:", j, cur_norm)
            if (
                    abs(cur_norm - pre_norm) < 0.005):
                print("stop early!!!")
                break
            pre_norm = cur_norm

    inverse_hvp = [b / scale for b in h_estimate]
    return inverse_hvp


def calc_IHVP_NeuMF(nonzero_train_indices, train_ratings, ratings_pred, criterion, params, v, h_estimate, i_epochs=50001, batch_size=2048, pre_norm=-11111, damp=1, scale=10):
    np.random.seed(seed=0)
    cur_norm_list = []
    for j in range(i_epochs):
        r = np.random.choice(len(nonzero_train_indices), size=batch_size, replace=False)
        users = nonzero_train_indices[:, 0][r]
        items = nonzero_train_indices[:, 1][r]
        t = train_ratings[r]
        t = t.to(torch.float32)
        x = ratings_pred[users, items]

        loss = criterion(x, t)
        hv = hessian_vector_product(loss, params, v, scale=10., do_not_sum_up=True)
        h_estimate = [_v + (1 - damp) * _h_e - _hv for _v, _h_e, _hv in zip(v, h_estimate, hv)]

        if j % 500 == 0 and j > 0:
            # cur_estimate = [tensor.detach().numpy() for tensor in h_estimate]
            cur_estimate = [tensor.cpu().detach().numpy() for tensor in h_estimate]
            cur_norm = np.linalg.norm(cur_estimate[0])
            if (j % 2500 == 0):
                print("Inverse HVP epoch:", j, cur_norm)
                cur_norm_list.append(cur_norm)
            if (
                    abs(cur_norm - pre_norm) < 0.05):
                print("stop early!!!")
                break
            pre_norm = cur_norm

    inverse_hvp = [b / scale for b in h_estimate]
    return cur_norm_list, inverse_hvp


def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    if not enabled:
        return

    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()

# calculate I_up,loss(x)
def calc_influence_single_LFM(num_users, nonzero_train_indices, train_data, ratings_pred, criterion, P_matrix, Q_matrix, params, inverse_hvp, display_progress, lambda_value=0.0003):

    influences = []
    for i in range(num_users):

        one_user_ratings_indices = nonzero_train_indices[nonzero_train_indices[:, 0] == i]
        one_user_ratings = train_data[one_user_ratings_indices[:, 0], one_user_ratings_indices[:, 1]]
        one_user_ratings = one_user_ratings.to(torch.float32)
        one_user_ratings_pred = ratings_pred[one_user_ratings_indices[:, 0], one_user_ratings_indices[:, 1]]
        loss = criterion(one_user_ratings_pred, one_user_ratings)

        selected_users_rows = P_matrix[i]
        selected_items_rows = Q_matrix[one_user_ratings_indices[:, 1]]
        loss_regularization = lambda_value * (torch.sum(torch.norm(selected_users_rows) ** 2) + torch.sum(
            torch.norm(selected_items_rows, dim=1) ** 2))

        loss = loss + loss_regularization
        grad_z_vec = list(grad(loss, params, create_graph=True))

        tmp_influence = -sum(
            [
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, inverse_hvp)
            ]) / num_users
        influences.append(tmp_influence)
        display_progress("Calc. influence function: ", i, num_users)

    influences_np = np.array([tensor.cpu().numpy() for tensor in influences])
    sorted_influences = sorted(enumerate(influences_np), key=lambda x: x[1], reverse=True)

    return sorted_influences


def calc_influence_single_NeuMF(num_users, nonzero_train_indices, train_data, ratings_pred, criterion, params, inverse_hvp, display_progress):

    influences = []
    for i in range(num_users):
        one_user_ratings_indices = nonzero_train_indices[nonzero_train_indices[:, 0] == i]
        one_user_ratings = train_data[one_user_ratings_indices[:, 0], one_user_ratings_indices[:, 1]]
        one_user_ratings = one_user_ratings.to(torch.float32)
        one_user_ratings_pred = ratings_pred[one_user_ratings_indices[:, 0], one_user_ratings_indices[:, 1]]
        loss = criterion(one_user_ratings_pred, one_user_ratings)

        grad_z_vec = list(grad(loss, params, create_graph=True))

        tmp_influence = -sum(
            [
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, inverse_hvp)
            ]) / num_users
        influences.append(tmp_influence)
        display_progress("Calc. influence function: ", i, num_users)

    influences_np = np.array([tensor.cpu().numpy() for tensor in influences])
    sorted_influences = sorted(enumerate(influences_np), key=lambda x: x[1], reverse=True)

    return sorted_influences




# 5. poisoning attack
#####################
# (1)Partial diffusion
def partial_diffusion(data_id, train_data, influential_instance_index, index_to_log_onehot, num_classes, model_MD, steps=99):


    if data_id == 'filmtrust':
        influential_instance = train_data[influential_instance_index]
        last_two_columns = influential_instance[-2:].clone()
        influential_instance_1 = torch.cat([influential_instance, last_two_columns])
        influential_instance_2 = influential_instance_1.view(1, 1, 33, 61)
        influential_instance_3 = influential_instance_2.to(torch.int64)
        log_x_start = index_to_log_onehot(influential_instance_3, num_classes)
    elif data_id == 'ml-100k':
        influential_instance = train_data[influential_instance_index]
        last_two_columns = influential_instance[-1:].clone()
        influential_instance_1 = torch.cat([influential_instance, last_two_columns])
        influential_instance_2 = influential_instance_1.view(1, 1, 33, 61)
        influential_instance_3 = influential_instance_2.to(torch.int64)
        log_x_start = index_to_log_onehot(influential_instance_3, num_classes)
    elif data_id == 'ml-1m':
        influential_instance = train_data[influential_instance_index]
        influential_instance_1 = influential_instance.view(1, 1, 33, 61)
        influential_instance_2 = influential_instance_1.to(torch.int64)
        log_x_start = index_to_log_onehot(influential_instance_2, num_classes)
    elif data_id == 'ml-10m':
        influential_instance = train_data[influential_instance_index]
        influential_instance_1 = influential_instance.view(1, 1, 33, 985)
        influential_instance_2 = influential_instance_1.to(torch.int64)
        log_x_start = index_to_log_onehot(influential_instance_2, num_classes)

    t1 = torch.tensor([steps], dtype=torch.int64)
    influential_instance_perturbed = model_MD.q_sample(log_x_start=log_x_start, t=t1)

    return influential_instance_perturbed

# (2)Integrate instances
def integrate_instances(A, B, PCA, dataset_name):

    if dataset_name == 'filmtrust':
        C = torch.matmul(A.view(-1, 1), B.view(1, -1))
        C1 = C.view(-1)
        C2 = torch.sign(C1) * torch.sqrt(torch.abs(C1))
        C3 = C2 / torch.norm(C2, p=2)
        C4 = C3.view(2011, -1)

        numpy_tensor = C4.numpy()
        data_standardized = (numpy_tensor - np.mean(numpy_tensor)) / np.std(numpy_tensor)
        pca = PCA(n_components=1)
        data_pca = pca.fit_transform(data_standardized)
        data_pca = data_pca.reshape(len(A))

        data_pca[data_pca < -1] = 0
        data_pca = ((data_pca - min(data_pca)) / (max(data_pca) - min(data_pca))) * 8
        data_pca = torch.round(torch.tensor(data_pca))

        D = data_pca[-2:].clone()
        D_1 = torch.cat([data_pca, D])
        D_2 = D_1.view(1, 1, 33, 61)
        influential_instance_integrated = D_2.to(torch.int64)
    elif dataset_name == 'ml-100k':
        C = torch.matmul(A.view(-1, 1), B.view(1, -1))
        C1 = C.view(-1)
        C2 = torch.sign(C1) * torch.sqrt(torch.abs(C1))
        C3 = C2 / torch.norm(C2, p=2)
        C4 = C3.view(1682, -1)

        numpy_tensor = C4.numpy()
        data_standardized = (numpy_tensor - np.mean(numpy_tensor)) / np.std(numpy_tensor)
        pca = PCA(n_components=1)
        data_pca = pca.fit_transform(data_standardized)
        data_pca = data_pca.reshape(len(A))

        data_pca[data_pca < -1] = 0
        data_pca = ((data_pca - min(data_pca)) / (max(data_pca) - min(data_pca))) * 8
        data_pca = torch.round(torch.tensor(data_pca))

        D = data_pca[-1:].clone()
        D_1 = torch.cat([data_pca, D])
        D_2 = D_1.view(1, 1, 33, 51)
        influential_instance_integrated = D_2.to(torch.int64)
    elif dataset_name == 'ml-1m':
        C = torch.matmul(A.view(-1, 1), B.view(1, -1))
        C1 = C.view(-1)
        C2 = torch.sign(C1) * torch.sqrt(torch.abs(C1))
        C3 = C2 / torch.norm(C2, p=2)
        C4 = C3.view(3706, -1)
    
        numpy_tensor = C4.numpy()
        data_standardized = (numpy_tensor - np.mean(numpy_tensor)) / np.std(numpy_tensor)
        pca = PCA(n_components=1)
        data_pca = pca.fit_transform(data_standardized)
        data_pca = data_pca.reshape(len(A))
    
        data_pca[data_pca < -1] = 0
        data_pca = ((data_pca - min(data_pca)) / (max(data_pca) - min(data_pca))) * 8
        data_pca = torch.round(torch.tensor(data_pca))
    
        D = data_pca.view(1, 1, 34, 109)
        influential_instance_integrated = D.to(torch.int64)
    elif dataset_name == 'ml-10m':
        C = torch.matmul(A.view(-1, 1), B.view(1, -1))
        C1 = C.view(-1)
        C2 = torch.sign(C1) * torch.sqrt(torch.abs(C1))
        C3 = C2 / torch.norm(C2, p=2)
        C4 = C3.view(32525, -1)
    
        numpy_tensor = C4.numpy()
        data_standardized = (numpy_tensor - np.mean(numpy_tensor)) / np.std(numpy_tensor)
        pca = PCA(n_components=1)
        data_pca = pca.fit_transform(data_standardized)
        data_pca = data_pca.reshape(len(A))
    
        data_pca[data_pca < -1] = 0
        data_pca = ((data_pca - min(data_pca)) / (max(data_pca) - min(data_pca))) * 32
        data_pca = torch.round(torch.tensor(data_pca))
    
        D = data_pca.view(1, 1, 33, 985)
        influential_instance_integrated = D.to(torch.int64)


    return influential_instance_integrated

# (3)Inference
def sample_chain(shape, model_MD, log_z, num_timesteps, log_onehot_to_index, b=1):
    zs = torch.zeros((num_timesteps, b) + shape).long()
    for i in reversed(range(num_timesteps)):
        print(f'Chain timestep {i:4d}', end='\r')
        t = torch.full((b,), i, dtype=torch.long)
        log_z = model_MD.p_sample(log_z, t)

        zs[i] = log_onehot_to_index(log_z)
    return zs

def inverse_sampling(index_to_log_onehot, influential_instance_integrated, num_classes, shape, model_MD, num_timesteps, log_onehot_to_index, dataset_name):
    log_z = index_to_log_onehot(influential_instance_integrated, num_classes)
    reserve_diffusion_instances_1 = sample_chain(shape, model_MD, log_z, num_timesteps, log_onehot_to_index)
    reserve_diffusion_instances_2 = reserve_diffusion_instances_1.view(num_timesteps, -1)
    reserve_diffusion_instances = reserve_diffusion_instances_2[:, :-2]

    return reserve_diffusion_instances



# 6.Greedy algorithm
####################
def cosine_similarity(vec1, vec2):
    dot_product = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1, 2)
    norm2 = torch.norm(vec2, 2)
    similarity = dot_product / (norm1 * norm2)
    return similarity.item()

def average_cosine_similarity(tensor, result_tensor):
    similarities = [cosine_similarity(tensor, v) for v in result_tensor]
    avg_similarity = sum(similarities) / len(similarities)
    return avg_similarity

def obtain_poisoning_instances(tensor, result_tensor):

    for t in range(99):
        min_similarity = float('10000')
        min_index = -1

        for i, row in enumerate(tensor):
            avg_cosine_sims = average_cosine_similarity(row, result_tensor)
            if avg_cosine_sims < min_similarity:
                min_similarity = avg_cosine_sims
                min_index = i

        if len(tensor) > 0 and min_index >= 0 and min_index < len(tensor):
            result_tensor = torch.cat((result_tensor, tensor[min_index].view(1, -1)))
        else:
            print(f'The{t+1}th iteration is wrong')
            print("Error: Unable to access tensor[min_index] due to invalid index or empty tensor.")
            break

        tensor = torch.cat((tensor[:min_index], tensor[min_index+1:]))
        result_tensor, indices = torch.unique(result_tensor, dim=0, return_inverse=True)

    return result_tensor



# 7. Evaluation metrics
#######################
def eval_metric(target_item_set, Top_K_list, origin_Top_K_list, rec_list, origin_rec_list, math, num_users):
    count = 0
    ndcg_count = 0
    er_count = 0
    recall_calc = 0
    rs_calc = 0

    for i in range(len(Top_K_list)):
        idx = Top_K_list[i]
        origin_idx = origin_Top_K_list[i]
        idx_1 = rec_list[i]
        origin_idx_1 = origin_rec_list[i]

        # HR and NDCG
        for j in target_item_set:
            count += (j in idx)
            ndcg_count += math.log(2) / math.log(
                idx.index(j) + 2) if j in idx else 0

        # ER
        if set(idx).intersection(set(target_item_set)):
            er_count += 1

        # Recall
        I1 = set(idx)
        I2 = set(origin_idx)
        recall_calc += (len(list(I1.intersection(I2))))/len(idx)

        # Rank shift
        changes = []
        for element in target_item_set:
            index = origin_idx_1.index(element)
            change = index - idx_1.index(element)
            changes.append(change)
        ave_changes = sum(changes)/len(target_item_set)
        rs_calc += ave_changes

    HR_value = count/num_users/len(target_item_set); HR_value = round(HR_value, 4)
    NDCG_value = ndcg_count/num_users/len(target_item_set); NDCG_value = round(NDCG_value, 4)
    ER_value = er_count/num_users; ER_value = round(ER_value, 4)
    Recall_value = recall_calc/num_users; Recall_value = round(Recall_value, 4)
    RS_value = int(rs_calc/num_users)

    return ER_value, RS_value, HR_value, NDCG_value, Recall_value
