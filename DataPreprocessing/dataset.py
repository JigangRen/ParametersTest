import random
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class TensorDataset(Dataset):

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


# """
def tensor_dataloader(dataload, para):
    if not para['split']:
        if not para['net'] == 'AECNN——':
            if para['dataset_path2'] == 'dataset#1':
                target = torch.from_numpy(dataload.dataset)
            elif para['dataset_path2'] == 'dataset#2':
                t = para['pred_time']
                torch.manual_seed(para['random_state'] + t)
                target = torch.from_numpy(dataload.dataset[:, :, t])
            else:
                print('dataset error in tensor_dataloader')
                return -1
            target = target.reshape(-1)
            input_tenosr = [[u, s] for u in range(dataload.users_num) for s in range(dataload.services_num)]
            random.shuffle(input_tenosr)
            target_tensor = [target[item[0] * dataload.services_num + item[1]] for item in input_tenosr]
            length = int(len(input_tenosr) * para['train_set'])
            length1 = int(len(input_tenosr) * (para['train_set'] + para['valid_set']))
            # 训练集
            train_input_tenosr = torch.from_numpy(np.array(input_tenosr[:length]))
            train_target_tensor = torch.from_numpy(np.array(target_tensor[:length]))
            train_dataset = TensorDataset(train_input_tenosr, train_target_tensor)
            train_loader = DataLoader(train_dataset,
                                      batch_size=para['batchSize'],
                                      shuffle=True,
                                      num_workers=0)
            # 验证集
            valid_input_tenosr = torch.from_numpy(np.array(input_tenosr[length:length1]))
            valid_target_tensor = torch.from_numpy(np.array(target_tensor[length:length1]))
            valid_dataset = TensorDataset(valid_input_tenosr, valid_target_tensor)
            valid_loader = DataLoader(valid_dataset,
                                      batch_size=para['batchSize'],
                                      shuffle=True,
                                      num_workers=0)
            # 测试集
            test_input_tenosr = torch.from_numpy(np.array(input_tenosr[length1:]))
            test_target_tensor = torch.from_numpy(np.array(target_tensor[length1:]))
            test_dataset = TensorDataset(test_input_tenosr, test_target_tensor)
            test_loader = DataLoader(test_dataset,
                                     batch_size=para['batchSize'],
                                     shuffle=True,
                                     num_workers=0)

            return train_loader, valid_loader, test_loader
        else:
            if para['dataset_path2'] == 'dataset#1':
                target = torch.from_numpy(dataload.dataset)
            elif para['dataset_path2'] == 'dataset#2':
                t = para['pred_time']
                torch.manual_seed(para['random_state'] + t)
                target = torch.from_numpy(dataload.dataset[:, :, t])
            else:
                print('dataset error in tensor_dataloader')
                return -1
            target = target.reshape(-1)

            uesr_num = [u for u in range(dataload.users_num)]
            random.shuffle(uesr_num)
            length = int(dataload.users_num * para['train_set'])
            uesr_num_train = uesr_num[:length]
            uesr_num_test = uesr_num[length:]
            service_num = [s for s in range(dataload.services_num)]
            random.shuffle(service_num)
            length = int(dataload.services_num * para['train_set'])
            service_num_train = service_num[:length]
            service_num_test = service_num[length:]

            input_tenosr_train = [[u, s] for u in uesr_num_train for s in service_num_train]
            random.shuffle(input_tenosr_train)
            input_tenosr_test = [[u, s] for u in uesr_num_test for s in service_num_test]
            random.shuffle(input_tenosr_test)

            target_tensor_train = [target[item[0] * dataload.services_num + item[1]] for item in input_tenosr_train]
            target_tensor_test = [target[item[0] * dataload.services_num + item[1]] for item in input_tenosr_test]

            train_input_tenosr = torch.from_numpy(np.array(input_tenosr_train))
            train_target_tensor = torch.from_numpy(np.array(target_tensor_train))
            train_dataset = TensorDataset(train_input_tenosr, train_target_tensor)
            train_loader = DataLoader(train_dataset,
                                      batch_size=para['batchSize'],
                                      shuffle=True,
                                      num_workers=0)

            test_input_tenosr = torch.from_numpy(np.array(input_tenosr_test))
            test_target_tensor = torch.from_numpy(np.array(target_tensor_test))
            test_dataset = TensorDataset(test_input_tenosr, test_target_tensor)
            test_loader = DataLoader(test_dataset,
                                     batch_size=para['batchSize'],
                                     shuffle=True,
                                     num_workers=0)

            return train_loader, test_loader

    else:
        if para['dataset_path2'] == 'dataset#1':
            target_train = torch.from_numpy(dataload[0].dataset)
            target_test = torch.from_numpy(dataload[1].dataset)
        elif para['dataset_path2'] == 'dataset#2':
            t = para['pred_time']
            torch.manual_seed(para['random_state'] + t)
            target_train = torch.from_numpy(dataload[0].dataset[:, :, t])
            target_test = torch.from_numpy(dataload[1].dataset[:, :, t])
        else:
            print('dataset error in tensor_dataloader')
            return -1
        target_train = target_train.reshape(-1)
        target_test = target_test.reshape(-1)
        input_tenosr_train = [[u, s] for u in range(dataload[0].users_num) for s in range(dataload[0].services_num)]
        input_tenosr_test = [[u, s] for u in range(dataload[1].users_num) for s in range(dataload[1].services_num)]
        random.shuffle(input_tenosr_train)
        random.shuffle(input_tenosr_test)
        target_tensor_train = [target_train[item[0] * dataload[0].services_num + item[1]] for item in
                               input_tenosr_train]
        target_tensor_test = [target_test[item[0] * dataload[1].services_num + item[1]] for item in input_tenosr_test]

        train_input_tenosr = torch.from_numpy(np.array(input_tenosr_train))
        train_target_tensor = torch.from_numpy(np.array(target_tensor_train))
        train_dataset = TensorDataset(train_input_tenosr, train_target_tensor)
        train_loader = DataLoader(train_dataset,
                                  batch_size=para['batchSize'],
                                  shuffle=True,
                                  num_workers=0)

        test_input_tenosr = torch.from_numpy(np.array(input_tenosr_test))
        test_target_tensor = torch.from_numpy(np.array(target_tensor_test))
        test_dataset = TensorDataset(test_input_tenosr, test_target_tensor)
        test_loader = DataLoader(test_dataset,
                                 batch_size=para['batchSize'],
                                 shuffle=True,
                                 num_workers=0)

        return train_loader, test_loader
