import random
import numpy as np
from DataPreprocessing.NMF import NMF
import torch
import DataPreprocessing.lib_func.dataload as dataload


class DATALOADER:

    def __init__(self, para):
        print('[Start Data Preprocessing]\n')
        self.para = para
        self.initial_qos = dataload.load(self.para)

        if para['dataset_path2'] == 'dataset#1':
            self.users_num, self.services_num = self.initial_qos.shape
        elif para['dataset_path2'] == 'dataset#2':
            self.users_num, self.services_num, self.times_num = self.initial_qos.shape

        self.reliable_qos = self.initial_qos
        self.dataset = np.empty([1, 1])
        self.densi_qos = np.empty([1, 1])
        self.posi = []
        self.graph = []
        self.mf_U = np.empty([1, 1])
        self.mf_S = np.empty([1, 1])
        self.user_load = []
        self.service_load = []
        self.user_sim = []
        self.service_sim = []
        self.reliable_U = []
        self.reliable_S = []
        # print('Dataset loading has been completed....')

    def unreliable_remove(self, confidence):
        if self.para['dataset_path2'] == 'dataset#1':
            posi = np.where(self.reliable_qos <= 0)
            unreliable_services = []
            for item in np.unique(posi[1]):
                count = (np.sum(posi[1] == item))
                if count > int(self.users_num * confidence):
                    unreliable_services.append(item)
            self.reliable_qos = np.delete(self.reliable_qos, unreliable_services, axis=1)
            posi = np.where(self.reliable_qos <= 0)
            unreliable_users = []
            for item in np.unique(posi[0]):
                count = (np.sum(posi[0] == item))
                if count > int(self.services_num * confidence):
                    unreliable_users.append(item)
            self.reliable_qos = np.delete(self.reliable_qos, unreliable_users, axis=0)
            self.users_num, self.services_num = self.reliable_qos.shape
            print('Unreliable users and services have been removed. (users:', unreliable_users.__len__(), 'services:',
                  unreliable_services.__len__(), ')')
        elif self.para['dataset_path2'] == 'dataset#2':
            posi = np.where(self.reliable_qos <= 0)
            unreliable_services = []
            for item in np.unique(posi[1]):
                count = (np.sum(posi[1] == item))
                if count > int(self.users_num * self.times_num * confidence):
                    unreliable_services.append(item)
            self.reliable_qos = np.delete(self.reliable_qos, unreliable_services, axis=1)
            posi = np.where(self.reliable_qos <= 0)
            unreliable_users = []
            for item in np.unique(posi[0]):
                count = (np.sum(posi[0] == item))
                if count > int(self.services_num * self.times_num * confidence):
                    unreliable_users.append(item)
            self.reliable_qos = np.delete(self.reliable_qos, unreliable_users, axis=0)
            # temp = np.where(self.reliable_qos <= 0)
            posi = np.where(self.reliable_qos >= 19.9)
            unreliable_users = []
            for item in np.unique(posi[0]):
                unreliable_users.append(item)
            self.reliable_qos = np.delete(self.reliable_qos, unreliable_users, axis=0)
            self.users_num, self.services_num, _ = self.reliable_qos.shape
            print('Unreliable users and services have been removed. (users:', unreliable_users.__len__(), 'services:',
                  unreliable_services.__len__(), ')')

    def speed(self, run_speed):
        run_speed = 1 - run_speed
        random.seed(self.para['random_state'])
        services_list = list(range(0, self.services_num))
        random.shuffle(services_list)
        users_list = list(range(0, self.users_num))
        random.shuffle(users_list)

        data_set = []
        services_dataset = services_list[0:int(self.services_num * run_speed)]
        users_dataset = users_list[0:int(self.users_num * run_speed)]
        if self.para['dataset_path2'] == 'dataset#1':
            for u in users_dataset:
                for s in services_dataset:
                    data_set.append(self.reliable_qos[u, s])
            self.dataset = np.array(data_set).reshape([len(users_dataset), len(services_dataset)])
            self.users_num, self.services_num = self.dataset.shape
        elif self.para['dataset_path2'] == 'dataset#2':
            for t in range(self.times_num):
                for u in users_dataset:
                    for s in services_dataset:
                        data_set.append(self.reliable_qos[u, s, t])
            self.dataset = np.array(data_set).reshape([len(users_dataset), len(services_dataset), -1])
            self.users_num, self.services_num, _ = self.dataset.shape
        # print('Data remove has been completed...')

    def split(self):
        random.seed(self.para['random_state'])
        services_list = list(range(0, self.services_num))
        random.shuffle(services_list)
        users_list = list(range(0, self.users_num))
        random.shuffle(users_list)

        if self.para['mode'] == 'train':
            data_set = []
            services_dataset = services_list[0:int(self.services_num * self.para['train_set'])]
            users_dataset = users_list[0:int(self.users_num * self.para['train_set'])]
            if self.para['dataset_path2'] == 'dataset#1':
                for u in users_dataset:
                    for s in services_dataset:
                        data_set.append(self.dataset[u, s])
                self.dataset = np.array(data_set).reshape([len(users_dataset), len(services_dataset)])
                self.users_num, self.services_num = self.dataset.shape
            elif self.para['dataset_path2'] == 'dataset#2':
                for t in range(self.times_num):
                    for u in users_dataset:
                        for s in services_dataset:
                            data_set.append(self.dataset[u, s, t])
                self.dataset = np.array(data_set).reshape([len(users_dataset), len(services_dataset), -1])
                self.users_num, self.services_num, _ = self.dataset.shape

        elif self.para['mode'] == 'test':
            data_set = []
            services_dataset = services_list[int(self.services_num * self.para['train_set']):]
            users_dataset = users_list[int(self.users_num * self.para['train_set']):]
            if self.para['dataset_path2'] == 'dataset#1':
                for u in users_dataset:
                    for s in services_dataset:
                        data_set.append(self.dataset[u, s])
                self.dataset = np.array(data_set).reshape([len(users_dataset), len(services_dataset)])
                self.users_num, self.services_num = self.dataset.shape
            elif self.para['dataset_path2'] == 'dataset#2':
                for t in range(self.times_num):
                    for u in users_dataset:
                        for s in services_dataset:
                            data_set.append(self.dataset[u, s, t])
                self.dataset = np.array(data_set).reshape([len(users_dataset), len(services_dataset), -1])
                self.users_num, self.services_num, _ = self.dataset.shape
        else:
            print('???啊这，dataloader的split这里有问题')

    def density(self, dens):
        """
        这个地方是为了将完整的数据集随机转化为稀疏的数据集，返回数据集索引
        :return: 返回不完整的用户qos矩阵
        """
        x_qos_lost = self.dataset.copy()
        if self.para['dataset_path2'] == 'dataset#1':
            random.seed(self.para['random_state'])
            list_len = self.services_num * self.users_num
            qos_list = list(range(0, list_len))
            random.shuffle(qos_list)
            qos_list_have = qos_list[0:int(list_len * dens)]
            qos_list_drop = qos_list[int(list_len * dens):]

            posi_have = [[int(i / self.services_num), int(i % self.services_num)] for i in qos_list_have]
            posi_drop = [[int(i / self.services_num), int(i % self.services_num)] for i in qos_list_drop]

            for pos in posi_drop:
                x_qos_lost[pos[0], pos[1]] = 0
            posi_have = np.array(posi_have)
            posi_drop = np.array(posi_drop)
            self.densi_qos = x_qos_lost
            self.posi = (posi_have, posi_drop)
        elif self.para['dataset_path2'] == 'dataset#2':
            list_len = self.services_num * self.users_num
            posi_have = []
            posi_drop = []
            for t in range(self.times_num):
                random.seed(self.para['random_state'] + t)
                qos_list = list(range(0, list_len))
                random.shuffle(qos_list)
                qos_list_have = qos_list[0:int(list_len * dens)]  # 留下来的值
                qos_list_drop = qos_list[int(list_len * dens):]  # 扣掉的值

                _posi_have = [[int(i / self.services_num), int(i % self.services_num)] for i in qos_list_have]
                _posi_drop = [[int(i / self.services_num), int(i % self.services_num)] for i in qos_list_drop]

                for pos in _posi_drop:
                    x_qos_lost[pos[0], pos[1], t] = 0
                posi_have.append(_posi_have)
                posi_drop.append(_posi_drop)
            self.densi_qos = x_qos_lost
            posi_have = np.array(posi_have)
            posi_drop = np.array(posi_drop)
            self.posi = (posi_have, posi_drop)
        else:
            print('密度这边有问题')
            return -1
        # print('Density processing has been completed...')

        return 0

    def nmf(self, dimension):
        self.para['dimension'] = dimension
        if self.para['dataset_path2'] == 'dataset#1':
            self.mf_U, self.mf_S = NMF.predict(self.densi_qos, self.para)
            self.reliable_U, self.reliable_S = NMF.predict(self.dataset, self.para)
            # print('Non-negative matrix factorization has been completed...')
        elif self.para['dataset_path2'] == 'dataset#2':
            self.mf_U, self.mf_S = NMF.predict(self.densi_qos[:, :, 0], self.para)
            self.mf_U = self.mf_U[:, :, None]
            self.mf_S = self.mf_S[:, :, None]
            for t in range(1, self.times_num):
                U_train, S_train = NMF.predict(self.densi_qos[:, :, t], self.para)
                # print('Non-negative matrix factorization has been completed...')
                self.mf_U = np.concatenate((self.mf_U, U_train[:, :, None]), axis=2)
                self.mf_S = np.concatenate((self.mf_S, S_train[:, :, None]), axis=2)

            self.reliable_U, self.reliable_S = NMF.predict(self.dataset[:, :, 0], self.para)
            self.reliable_U = self.reliable_U[:, :, None]
            self.reliable_S = self.reliable_S[:, :, None]
            for t in range(1, self.times_num):
                U_train, S_train = NMF.predict(self.dataset[:, :, t], self.para)
                # print('Non-negative matrix factorization has been completed...')
                self.reliable_U = np.concatenate((self.reliable_U, U_train[:, :, None]), axis=2)
                self.reliable_S = np.concatenate((self.reliable_S, S_train[:, :, None]), axis=2)

    def bipartite_graph2tree(self):
        self.graph = dataload.bipartite_graph2tree(self.posi[0], self.densi_qos.shape)

    def _simplified_tree(self, graph, users_sim_matrix, services_sim_matrix):

        # para['user_graph_len'])
        user_slim_graph = {}
        for i in range(self.users_num):
            temp_list = [users_sim_matrix[i, item] for item in graph[0][str(i)]]
            temp_list = np.array(temp_list)
            element_num = len(temp_list)
            if element_num > 2 * self.para['user_graph_len']:
                posi = np.argsort(temp_list)[(element_num - self.para['user_graph_len']):]
                if self.para['mutation rate'] != 0:
                    posi2 = np.argsort(temp_list)[
                            :int(self.para['user_graph_len'] * self.para['mutation rate'])]
                    posi_new = np.append(posi[::-1], posi2[::-1]).reshape(-1)
                else:
                    posi_new = posi[::-1].reshape(-1)
                temp_list = [graph[0][str(i)][item] for item in posi_new]
                user_slim_graph[str(i)] = temp_list
            else:
                temp1 = np.argsort(users_sim_matrix[i])[
                        users_sim_matrix.shape[0] - self.para['user_graph_len']:].tolist()
                if self.para['mutation rate'] != 0:
                    temp2 = np.argsort(users_sim_matrix[i])[
                            :int(self.para['user_graph_len'] * self.para['mutation rate'])].tolist()
                    temp = np.append(temp1[::-1], temp2[::-1]).reshape(-1)
                else:
                    temp = np.array(temp1)[::-1].reshape(-1)
                user_slim_graph[str(i)] = temp

        service_slim_graph = {}
        for i in range(self.services_num):
            temp_list = [services_sim_matrix[i, item] for item in graph[1][str(i)]]
            temp_list = np.array(temp_list)
            element_num = len(temp_list)
            if element_num > 2 * self.para['service_graph_len']:
                posi = np.argsort(temp_list)[(element_num - self.para['service_graph_len']):]
                if self.para['mutation rate'] != 0:
                    posi2 = np.argsort(temp_list)[:int(self.para['service_graph_len'] * self.para['mutation rate'])]
                    posi_new = np.append(posi[::-1], posi2[::-1]).reshape(-1)
                else:
                    posi_new = posi[::-1].reshape(-1)
                temp_list = [graph[1][str(i)][item] for item in posi_new]
                service_slim_graph[str(i)] = temp_list
            else:
                temp1 = np.argsort(services_sim_matrix[i])[
                        (services_sim_matrix.shape[0] - self.para['service_graph_len']):].tolist()
                if self.para['mutation rate'] != 0:
                    temp2 = np.argsort(services_sim_matrix[i])[
                            :int(self.para['service_graph_len'] * self.para['mutation rate'])].tolist()
                    temp = np.append(temp1[::-1], temp2[::-1]).reshape(-1)
                else:
                    temp = np.array(temp1)[::-1].reshape(-1)
                service_slim_graph[str(i)] = temp

        inputdata = (user_slim_graph, service_slim_graph)

        return inputdata

    def simplified_tree(self):
        # self.mf_S, self.mf_U
        if self.para['simplified_tree']:
            if self.para['dataset_path2'] == 'dataset#1':
                self.user_sim = dataload.matrix_sim(self.mf_U)
                self.service_sim = dataload.matrix_sim(self.mf_S)
            elif self.para['dataset_path2'] == 'dataset#2':
                self.user_sim = dataload.matrix_sim(np.hstack(self.mf_U.transpose((1, 0, 2))))  #
                self.service_sim = dataload.matrix_sim(np.hstack(self.mf_S.transpose((1, 0, 2))))  #
            input_data = self._simplified_tree(self.graph, self.user_sim, self.service_sim)

            self.user_load = self.mf_U[input_data[0]['0'], :]
            self.user_load = np.expand_dims(self.user_load, axis=0)
            for i in range(1, self.users_num):
                temp = self.mf_U[input_data[0][str(i)], :]
                temp = np.expand_dims(temp, axis=0)
                self.user_load = np.concatenate((self.user_load, temp), axis=0)

            self.service_load = self.mf_S[input_data[1]['0'], :]
            self.service_load = np.expand_dims(self.service_load, axis=0)
            for i in range(1, self.services_num):
                temp = self.mf_S[input_data[1][str(i)], :]
                temp = np.expand_dims(temp, axis=0)
                self.service_load = np.concatenate((self.service_load, temp), axis=0)

            self.user_load = self.user_load.astype('float32')
            self.service_load = self.service_load.astype('float32')
        else:
            i = self.para['pred_time'] - self.para['duration']
            self.user_load = self.mf_U[:, :, i]
            self.user_load = np.expand_dims(self.user_load, axis=0)
            self.service_load = self.mf_S[:, :, i]
            self.service_load = np.expand_dims(self.service_load, axis=0)
            for i in range(self.para['pred_time'] - self.para['duration'] + 1, self.para['pred_time'] + 1):
                temp = self.mf_U[:, :, i]
                temp = np.expand_dims(temp, axis=0)
                self.user_load = np.concatenate((self.user_load, temp), axis=0)
                temp = self.mf_S[:, :, i]
                temp = np.expand_dims(temp, axis=0)
                self.service_load = np.concatenate((self.service_load, temp), axis=0)
            self.user_load = self.user_load.astype('float32')
            self.service_load = self.service_load.astype('float32')

            print('ok')
        print('ok')


class Dataloader(DATALOADER):

    def __init__(self, para):
        print('==========================================================================')
        super(Dataloader, self).__init__(para)
        self.unreliable_remove(para['confidence'])
        self.speed(para['run_speed'])
        if para['split']:
            self.split()
        self.density(para['density'])
        self.nmf(para['dimension'])

        self.bipartite_graph2tree()
        # print('Bipartite graph transfer to users/services graph has been completed...')
        self.simplified_tree()
        # print('Simplified tree has been created...')
        if self.para['dataset_path2'] == 'dataset#1':
            self.predmat = np.dot(self.mf_U, self.mf_S.T)
        elif self.para['dataset_path2'] == 'dataset#2':
            self.predmat = []
            for t in range(self.times_num):
                self.predmat.append(np.dot(self.mf_U[:, :, t], self.mf_S[:, :, t].T))
            self.predmat = np.array(self.predmat)
        para['services_num'] = self.services_num
        para['users_num'] = self.users_num
        print('==========================================================================')

    def update(self, u_feature, s_feature):
        self.mf_U = u_feature
        self.mf_S = s_feature
        self.user_sim = dataload.matrix_sim(self.mf_U)  # , self.para['user_graph_len'])
        self.service_sim = dataload.matrix_sim(self.mf_S)  # , self.para['service_graph_len'])
        self.simplified_tree()
        self.predmat = np.dot(self.mf_U, self.mf_S.T)

    def pred_tensor(self, batch_data):
        batch_size = batch_data.shape[0]
        x1 = []
        x2 = []
        x3 = []
        x4 = []
        for i in range(batch_size):
            data = batch_data[i]
            temp1 = self.compute_x1(data)
            temp2 = self.compute_x2(data)
            temp3 = self.compute_x3(data)
            temp4 = self.compute_x4(data)
            x1.append(temp1)
            x2.append(temp2)
            x3.append(temp3)
            x4.append(temp4)
        x1 = torch.cat(x1).view(batch_size, -1).float()
        x2 = torch.cat(x2).view(batch_size, -1)
        x3 = torch.cat(x3).view(batch_size, -1)
        x4 = torch.from_numpy(np.array(x4)).unsqueeze(1)

        return x1.to(torch.float32), x2.to(torch.float32), x3.to(torch.float32), x4.to(torch.float32)

    def compute_x1(self, data):
        x_qos = torch.from_numpy(self.predmat)
        x1 = torch.cat([x_qos[int(data[0]), :], x_qos[:, int(data[1])]], dim=0)
        return x1

    def compute_x2(self, data):

        # np.argsort(temp_list)
        sort_pcc1 = self.user_sim[int(data[0]), :]
        sort_pcc1 = np.argsort(sort_pcc1)[-self.para['user_graph_len']:]
        sort_pcc2 = self.service_sim[int(data[1]), :]
        sort_pcc2 = np.argsort(sort_pcc2)[-self.para['service_graph_len']:]
        x2 = []
        for u in sort_pcc1:
            for s in sort_pcc2:
                x2.append(self.predmat[u, s])
        x2 = torch.from_numpy(np.array(x2))
        return x2

    def compute_x3(self, data):
        p = self.mf_U[data[0], :]
        q = self.mf_S[data[1], :]
        x3 = np.concatenate([p, q])
        x3 = torch.from_numpy(np.array(x3))
        return x3

    def compute_x4(self, data):
        x4 = self.predmat[data[0], data[1]]
        return x4
