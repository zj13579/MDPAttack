from MDPAttack.MDPAttack import MDPAttack
model = MDPAttack()
args = model.args
args.dataset_name = 'filmtrust'  # 'filmtrust'/'ml-100k'/'ml-1m'/'ml-10m'

if args.dataset_name == 'filmtrust':
    args.num_users = 796
    args.num_items = 2011
    args.num_classes = 9

    args.recommender = 'LFM'  # 'NeuMF'
    args.learning_rate = 0.05
    args.num_epochs = 2000  # iterations of LFM
    args.lambda_value = 0.0003  # lambda of LFM

    args.num_template = 8
    args.num_iterations_FGSM = 10
    args.target_item_index_list = [1371, 1521, 1066, 124, 329]  # [336, 1155, 328, 1164, 1387]
    args.reshape_value = 2013
    args.data_shape = [1, 33, 61]

elif args.dataset_name == 'ml-100k':
    args.num_users = 943
    args.num_items = 1682
    args.num_classes = 6

    args.recommender = 'LFM'
    args.learning_rate = 0.1
    args.num_epochs = 1000
    args.lambda_value = 0.0001

    args.num_template = 10
    args.num_iterations_FGSM = 10
    args.target_item_index_list = [1324, 1372, 1579, 1485, 1307]  # [1343, 1501, 1318, 113, 512]
    args.reshape_value = 1683
    args.data_shape = [1, 33, 51]

elif args.dataset_name == 'ml-1m':
    args.num_users = 6040
    args.num_items = 3706
    args.num_classes = 6

    args.recommender = 'LFM'
    args.learning_rate = 0.09
    args.num_epochs = 1000
    args.lambda_value = 0.001

    args.num_template = 120
    args.num_iterations_FGSM = 5
    args.target_item_index_list = [1920, 2089, 140, 1377, 2570]  # [3458, 1577, 3104, 3646, 1722]
    args.reshape_value = 3706
    args.data_shape = [1, 34, 109]

elif args.dataset_name == 'ml-10m':
    args.num_users = 35785
    args.num_items = 32525
    args.num_classes = 6

    args.recommender = 'LFM'
    args.learning_rate = 0.001
    args.num_epochs = 2000
    args.lambda_value = 0.005

    args.num_template = 716
    args.num_iterations_FGSM = 5
    args.target_item_index_list = [6945, 3018, 12125, 30245, 85]  # [2754, 13846, 841, 109, 2342]
    args.reshape_value = 32525
    args.data_shape = [1, 33, 985]


model.select_template_profiles()
model.train_MD()
model.poisoning_attack()






