from math import sqrt
import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn
import sys
from model.MMDNN.MMDNN import MMDNN


def init_para():
    initial_para = {'net': 'nan',  # TSDGRU or TSDCNN or TSD3DCNN AECNN GRGAN GRGAN_GRU
                    'dimension': 60,  # dimenision of the latent factors
                    'lambda': 50,  # regularization parameter 30
                    'maxIter': 500,  # the max iterations 300
                    'batchSize': 1024,
                    'epoch': 5,  # 默认是1000
                    'user_graph_len': 30,
                    'service_graph_len': 30,
                    'mutation rate': 0.2,
                    'dataset_path1': 'DataPreprocessing/ws_dream/',
                    'dataset_path2': 'dataset#1',
                    'random_state': 1,
                    'confidence': 0.1,  # remove the unreliable users/services
                    'run_speed': 0,  # 写代码测试时候用到的参数，值在0到1之间，0.9的时候代码跑会快一点（代表丢弃了90%的数据集）
                    'train_set': 0.7,  # 用来分训练集和测试集的
                    'valid_set': 0.1,
                    'GPU_Parallel': False,  # 多卡并行开关，Linux下才可以打开，在win下使用好像可能会导致速度变慢
                    'device': 'cuda:0',  # 选择一个默认gpu跑,这里选的是gpu0
                    'learning rate': 0.001,
                    'simplified_tree': True,
                    'split': False
                    }
    return initial_para


def comput_result(prediction, target):
    test_vec_x = np.where(target > 0)
    prediction = prediction[test_vec_x]
    target = target[test_vec_x]
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squared_error = []
    abs_error = []
    for val in error:
        squared_error.append(val * val)  # target-prediction之差平方
        abs_error.append(abs(val))  # 误差绝对值
    mse = float(sum(squared_error) / len(squared_error))  # 均方误差MSE
    mae = float(sum(abs_error) / len(abs_error))  # 平均绝对误差MAE
    rmse = sqrt(sum(squared_error) / len(squared_error))  # 均方根误差RMSE
    return mse, mae, rmse


# 固定随机数序列
def ini_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用 multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # 固定Numpy 的随机数.
    random.seed(seed)  # Python random module


# 计算结果
# 原来的名字叫testset_result(_test_dataloader, data_load)
def dataset_result(dataset_loader, preprocessed_dataset, t=63):
    """

    :param t:
    :param dataset_loader: 这里是用dataset分好的tensor
    :param preprocessed_dataset: 这里指预处理后的数据集类
    :return: 返回对应的mae和rmse
    """
    pred = preprocessed_dataset.predmat
    # pred = data_load.dataset  # 这个东西就是label
    pred_result = []
    label_result = []
    for _input_data, _target in dataset_loader:
        if preprocessed_dataset.para['dataset_path2'] == 'dataset#1':
            pred_result.append(np.array([pred[item[0], item[1]] for item in _input_data]))
            label_result.append(np.array(_target))
        elif preprocessed_dataset.para['dataset_path2'] == 'dataset#2':
            pred_result.append(np.array([pred[t, item[0], item[1]] for item in _input_data]))
            label_result.append(np.array(_target))

    pred_result = np.concatenate(pred_result)
    label_result = np.concatenate(label_result)
    _, _MAE, _RMSE = comput_result(pred_result, label_result)
    return _MAE, _RMSE


def init_net(para):
    #
    # 初始化网络
    if para['GPU_Parallel']:
        para['device'] = 'cuda:0'
    para['device'] = torch.device(para['device'] if torch.cuda.is_available() else "cpu")
    if para['net'] == 'TSDGRU':
        from model.TSDGRU.TwoStreamGRU import TSDGRN

        net = TSDGRN(para).to(para['device'])
    elif para['net'] == 'TSDCNN':
        from model.TSDCNN.TSDCNN import TSDCNN

        net = TSDCNN(para).to(para['device'])
    elif para['net'] == 'TSD3DCNN':
        from model.TSDCNN.TSD3DCNN import TSDCNN

        net = TSDCNN(para).to(para['device'])
    elif para['net'] == 'GRGAN':
        from model.GRGAN.GRGAN import Generator, Discriminator
        generator = Generator(para).to(para['device'])
        discriminator = Discriminator(para).to(para['device'])
    elif para['net'] == 'GRGAN_GRU':
        from model.GRGAN.GRGAN_GRU import Generator, Discriminator
        generator = Generator(para).to(para['device'])
        discriminator = Discriminator(para).to(para['device'])
        #
        # 多卡并行的代码
        if para['GPU_Parallel']:
            para['device'] = 'cuda:0'
            if torch.cuda.device_count() > 1:
                print('Use', torch.cuda.device_count(), 'GPUs')
                generator = nn.DataParallel(generator)
                discriminator = nn.DataParallel(discriminator)
        net = [generator, discriminator]

    elif para['net'] == 'AECNN':
        from model.AutoEncoder.AECNN import AECNN

        net = AECNN(para).to(para['device'])
    elif para['net'] == 'MMDNN':
        from model.MMDNN.MMDNN import MMDNN

        net = MMDNN(para).to(para['device'])
    else:
        print('Can\'t find the network called ' + para['net'])
        sys.exit(1)

    if para['net'] == 'GRGAN':
        criterion = torch.nn.MSELoss()
        optim_G = torch.optim.Adam(net[0].parameters(), lr=para['learning rate'], betas=(para['b1'], para['b2']))
        optim_D = torch.optim.Adam(net[1].parameters(), lr=para['learning rate'], betas=(para['b1'], para['b2']))
        optimizer = (optim_G, optim_D)
    else:
        if para['net'] == 'AECNN':
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
        optimizer = optim.Adam(net.parameters(), lr=para['learning rate'], weight_decay=1e-5)
        #
        # 多卡并行的代码
        if para['GPU_Parallel']:
            para['device'] = 'cuda:0'
            if torch.cuda.device_count() > 1:
                print('Use', torch.cuda.device_count(), 'GPUs')
                net = nn.DataParallel(net)

    return net, criterion, optimizer


def init_pred_net(para):
    #
    # 初始化MMDNN网络
    if para['GPU_Parallel']:
        para['device'] = 'cuda:0'
    para['device'] = torch.device(para['device'] if torch.cuda.is_available() else "cpu")

    net = MMDNN(para).to(para['device'])

    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    #
    # 多卡并行的代码
    if para['GPU_Parallel']:
        para['device'] = 'cuda:0'
        if torch.cuda.device_count() > 1:
            print('Use', torch.cuda.device_count(), 'GPUs')
            net = nn.DataParallel(net)

    return net, criterion, optimizer


#
def net_test(input_dataloader, preprocessed_data, para, net):
    # 验证网络效果的代码
    with torch.no_grad():
        output_ = []
        target_ = []
        if preprocessed_data.para['dataset_path2'] == 'dataset#1':
            for input_data, target in input_dataloader:
                user_mat = [preprocessed_data.user_load[item[0], :] for item in input_data]
                service_mat = [preprocessed_data.service_load[item[1], :] for item in input_data]
                user_tensor = torch.from_numpy(np.array(user_mat)).to(para['device'])
                service_tensor = torch.from_numpy(np.array(service_mat)).to(para['device'])
                if para['net'] != 'AECNN':
                    out_put = net(user_tensor.unsqueeze(2), service_tensor.unsqueeze(2))
                else:
                    out_put = net[0](user_tensor.unsqueeze(2), service_tensor.unsqueeze(2), net[1])
                output_.append(out_put.clone().detach_().cpu().numpy())
                target_.append(target.clone().detach_().cpu().numpy())
            output_ = np.concatenate(output_)
            target_ = np.concatenate(target_)
            _, _mae, _rmse = comput_result(output_, target_)
        elif preprocessed_data.para['dataset_path2'] == 'dataset#2':
            for input_data, target in input_dataloader:
                user_mat = [preprocessed_data.user_load[:, item[0], :] for item in input_data]
                service_mat = [preprocessed_data.service_load[:, item[1], :] for item in input_data]
                user_tensor = torch.from_numpy(np.array(user_mat)).to(para['device'])
                service_tensor = torch.from_numpy(np.array(service_mat)).to(para['device'])
                if para['net'] == 'GRGAN':
                    out_put = net(user_tensor.unsqueeze(2), service_tensor.unsqueeze(2), stage=2)
                else:
                    print('其他网络的时间序列代码还没写惹')
                    return 0

                output_.append(out_put.clone().detach_().cpu().numpy())
                target_.append(target.clone().detach_().cpu().numpy())
            output_ = np.concatenate(output_)
            target_ = np.concatenate(target_)
            _, _mae, _rmse = comput_result(output_, target_)
        """
                if para['net'] != 'AECNN':
            
        else:  # 这里是AECNN
            if preprocessed_data.para['dataset_path2'] == 'dataset#1':
                for input_data, target in input_dataloader:
                    user_mat = [preprocessed_data.mf_U[item[0], :] for item in input_data]
                    service_mat = [preprocessed_data.mf_S[item[1], :] for item in input_data]
                    user_tensor = torch.from_numpy(np.array(user_mat)).to(para['device']).float()
                    service_tensor = torch.from_numpy(np.array(service_mat)).to(para['device']).float()
                    out_put = net[0](user_tensor.unsqueeze(1).unsqueeze(1), service_tensor.unsqueeze(1).unsqueeze(1),
                                     net[1])
                    output_.append(out_put.clone().detach_().cpu().numpy())
                    target_.append(target.clone().detach_().cpu().numpy())
                output_ = np.concatenate(output_)
                target_ = np.concatenate(target_)
                _, _mae, _rmse = comput_result(output_, target_)
            elif preprocessed_data.para['dataset_path2'] == 'dataset#2':
                for input_data, target in input_dataloader:
                    user_mat = [preprocessed_data.user_load[:, item[0], :] for item in input_data]
                    service_mat = [preprocessed_data.service_load[:, item[1], :] for item in input_data]
                    user_tensor = torch.from_numpy(np.array(user_mat)).to(para['device'])
                    service_tensor = torch.from_numpy(np.array(service_mat)).to(para['device'])
                    gen_user_tenser = net[0](user_tensor.unsqueeze(2))
                    gen_service_tenser = net[1](service_tensor.unsqueeze(2))
                    out_put = torch.matmul(gen_user_tenser.unsqueeze(1), gen_service_tenser.unsqueeze(2)).squeeze()
                    output_.append(out_put.clone().detach_().cpu().numpy())
                    target_.append(target.clone().detach_().cpu().numpy())
                output_ = np.concatenate(output_)
                target_ = np.concatenate(target_)
                _, _mae, _rmse = comput_result(output_, target_)
        """

    return _mae, _rmse
