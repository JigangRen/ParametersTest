from DataPreprocessing.dataloader import Dataloader
from DataPreprocessing.dataset import tensor_dataloader
import torch
import numpy as np
from tqdm import trange
import sys
import func
from model.AutoEncoder.AECNN import FusinNet
import torch.nn as nn
import torch.optim as optim

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6

def train(para):

    func.ini_seed(para['random_state'])

    preprocessed_dataload = Dataloader(para)
    train_dataloader, valid_dataloader, test_dataloader = tensor_dataloader(preprocessed_dataload, para)

    net1, criterion1, optimizer1 = func.init_net(para)
    net2, criterion2, optimizer2 = func.init_net(para)

    print(f"Model_1 has {count_parameters(net1):.2f} million parameters")
    # 这行代码用来算参数量的，会导致网络跑不通，需要再用
    # summary(net.cuda(), input_size=[(42, 1, 20), (42, 1, 20)])
    #
    init_mae_train, _ = func.dataset_result(train_dataloader, preprocessed_dataload)
    print("The initial train MAE is %f" % init_mae_train)
    init_mae_test, _ = func.dataset_result(test_dataloader, preprocessed_dataload)
    print("The initial test MAE is %f" % init_mae_test)

    pbar = trange(para['epoch'], file=sys.stdout)
    for _ in pbar:
        for input_data, target in train_dataloader:
            if para['dataset_path2'] == 'dataset#1':
                user_mat = [preprocessed_dataload.user_load[item[0], :] for item in input_data]
                service_mat = [preprocessed_dataload.service_load[item[1], :] for item in input_data]
            elif para['dataset_path2'] == 'dataset#2':  # [时间，用户/服务，维度]
                user_mat = [preprocessed_dataload.user_load[:, item[0], :] for item in
                            input_data]
                service_mat = [preprocessed_dataload.service_load[:, item[1], :] for item
                               in input_data]
            else:
                print('error in AE_run line 62')
                return -1
            user_tensor = torch.from_numpy(np.array(user_mat)).to(para['device']).float()
            service_tensor = torch.from_numpy(np.array(service_mat)).to(para['device']).float()
            # print("user_tensor.shape:" )
            # print(user_tensor.size())
            # print("service_tensor.shape:")
            # print(service_tensor.size())
            # print("user_tensor.unsqueeze.shape:")
            # print( user_tensor.unsqueeze(2).size())
            # print("service_tensor.unsqueeze.shape:")
            # print(service_tensor.unsqueeze(2).size())

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            out_put1 = net1(user_tensor.unsqueeze(2))
            out_put2 = net2(service_tensor.unsqueeze(2))
            loss1 = criterion1(out_put1[2], user_tensor[:, 0, :])
            loss2 = criterion2(out_put2[2], service_tensor[:, 0, :])
            loss1.backward()
            loss2.backward()
            optimizer1.step()
            optimizer2.step()

            pbar.set_description("loss1 = %f,loss2 = %f" % (loss1, loss2))
    for p in net1.parameters():
        p.requires_grad = False  # 锁参数
    for p in net2.parameters():
        p.requires_grad = False  # 锁参数

    temp_best_mae=9999
    temp_best_rmse=9999
    a_best_mae=[]
    a_best_rmse=[]
    net = FusinNet(para).to(para['device'])
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=para['learning rate'], weight_decay=1e-5)
    print(f"Model_2 has {count_parameters(net):.2f} million parameters")

    if para['GPU_Parallel']:
        para['device'] = 'cuda:0'
        if torch.cuda.device_count() > 1:
            print('Use', torch.cuda.device_count(), 'GPUs')
            net = nn.DataParallel(net)

    pbar = trange(para['epoch'], file=sys.stdout)
    for _ in pbar:
        output_ = []
        target_ = []
        for input_data, target in train_dataloader:
            if para['dataset_path2'] == 'dataset#1':
                user_mat = [preprocessed_dataload.user_load[item[0], :] for item in input_data]
                service_mat = [preprocessed_dataload.service_load[item[1], :] for item in input_data]
            elif para['dataset_path2'] == 'dataset#2':  # [时间，用户/服务，维度]
                user_mat = [preprocessed_dataload.user_load[:, item[0], :] for item in
                            input_data]
                service_mat = [preprocessed_dataload.service_load[:, item[1], :] for item
                               in input_data]
            else:
                print('error in AE_run line 102')
                return -1
            user_tensor = torch.from_numpy(np.array(user_mat)).to(para['device']).float()
            service_tensor = torch.from_numpy(np.array(service_mat)).to(para['device']).float()

            optimizer.zero_grad()
            out_put = net(user_tensor.unsqueeze(2), service_tensor.unsqueeze(2), (net1, net2))
            loss = criterion(out_put, target.to(para['device']))
            loss.backward()
            optimizer.step()
            output_.append(out_put.clone().detach_().cpu().numpy())
            target_.append(target.clone().detach_().cpu().numpy())
        output_ = np.concatenate(output_)
        target_ = np.concatenate(target_)
        train_mse, train_mae, train_rmse = func.comput_result(output_, target_)

        test_mae, test_rmse = func.net_test(test_dataloader, preprocessed_dataload, para, (net, (net1, net2)))
        a_best_mae.append(test_mae)
        a_best_rmse.append(test_rmse)
        if(temp_best_mae>test_mae):
            temp_best_mae=test_mae
            temp_best_rmse=test_rmse
        pbar.set_description("train: MAE = %f test: MAE = %f  RMSE = %f" % ( train_mae, test_mae, test_rmse))
    return np.mean(a_best_mae),np.mean(a_best_rmse),temp_best_mae,temp_best_rmse

if __name__ == '__main__':
    initial_para = func.init_para()
    initial_para['net'] = 'AECNN'
    initial_para['device'] = 'cuda:1'
    initial_para['dataset_path2'] = 'dataset#1'
    initial_para['qos_attribute'] = 'rt'
    initial_para['density'] = 0.05
    initial_para['split'] = False
    initial_para['run_speed'] = 0
    initial_para['user_graph_len'] = 15
    initial_para['service_graph_len'] = 15

    for densi in range(5, 21, 5):
        best_mae=0
        best_rmse=0
        b_best_mae=0
        b_best_rmse=0
        print('\n\n\ndensi=', densi / 100)
        initial_para['density'] = densi / 100
        initial_para['qos_attribute'] = 'rt'
        b_best_mae,b_best_rmse,best_mae,best_rmse=train(initial_para)
        print('类型：'+initial_para['qos_attribute']+'\t密度：'+str(initial_para['density'])+'\tbest_mae：'+str(best_mae)+'\tbest_rmse：'+str(best_rmse)+"\t均值best_mae："+str(b_best_mae)+"\t均值best_rmse："+str(b_best_rmse))
        initial_para['qos_attribute'] = 'tp'
        b_best_mae,b_best_rmse,best_mae,best_rmse=train(initial_para)
        print('类型：'+initial_para['qos_attribute']+'\t密度：'+str(initial_para['density'])+'\tbest_mae：'+str(best_mae)+'\tbest_rmse：'+str(best_rmse)+"\t均值best_mae："+str(b_best_mae)+"\t均值best_rmse："+str(b_best_rmse))

