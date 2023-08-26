import random
import numpy as np
from DataPreprocessing.NMF import NMF
import func


def preprocess(matrix, _para):
    if _para['qos_attribute'] == 'rt':
        matrix = np.where(matrix == 0, -0.01, matrix)
        matrix = np.where(matrix >= 19.9, -0.01, matrix)
    elif _para['qos_attribute'] == 'tp':
        matrix = np.where(matrix == 0, -0.01, matrix)
    return matrix


def test_nmf_for_wsdream(para_):
    if para_['dataset_path2'] == 'dataset#1':
        datafile = para_['dataset_path1'] + para_['dataset_path2'] + '/' + para_['qos_attribute'] + 'Matrix.txt'
        data_matrix = np.loadtxt(datafile)
    elif para_['dataset_path2'] == 'dataset#2':
        datafile = para_['dataset_path1'] + para_['dataset_path2'] + '/' + para_['qos_attribute'] + 'data.txt'
        data_matrix = -1 * np.ones((142, 4500, 64))
        fid = open(datafile, 'r')
        for line in fid:
            data = line.split(' ')
            rt = float(data[3])
            if rt > 0:
                data_matrix[int(data[0]), int(data[1]), int(data[2])] = rt
        fid.close()
        data_matrix = preprocess(data_matrix, para_)
    else:
        print('data load error!')
        return -1
    initial_qos = preprocess(data_matrix, para_)
    # 下面这个用来生成稀疏矩阵
    if para_['dataset_path2'] == 'dataset#1':
        x_qos_lost = initial_qos.copy()
        users_num, services_num = initial_qos.shape
        list_len = services_num * users_num
        posi_have = []
        posi_drop = []
        random.seed(para_['random_state'])
        qos_list = list(range(0, list_len))
        random.shuffle(qos_list)
        qos_list_have = qos_list[0:int(list_len * para_['density'])]  # 留下来的值
        qos_list_drop = qos_list[int(list_len * para_['density']):]  # 扣掉的值

        _posi_have = [[int(i / services_num), int(i % services_num)] for i in qos_list_have]  # 留下来的位置
        _posi_drop = [[int(i / services_num), int(i % services_num)] for i in qos_list_drop]  # 扣掉的位置

        for pos in _posi_drop:
            x_qos_lost[pos[0], pos[1]] = 0
        posi_have.append(_posi_have)
        posi_drop.append(_posi_drop)
        densi_qos = x_qos_lost

        mf_U, mf_S = NMF.predict(densi_qos, para_)
        predmat = np.dot(mf_U, mf_S.T)
        target = initial_qos
        mse, mae, rmse = func.comput_result(predmat, target)
        print(mae)
    elif para_['dataset_path2'] == 'dataset#2':
        x_qos_lost = initial_qos.copy()
        users_num, services_num, times_num = initial_qos.shape
        list_len = services_num * users_num
        posi_have = []
        posi_drop = []
        for t in range(2):
            random.seed(para_['random_state'] - t)
            qos_list = list(range(0, list_len))
            random.shuffle(qos_list)
            qos_list_have = qos_list[0:int(list_len * para_['density'])]  # 留下来的值
            qos_list_drop = qos_list[int(list_len * para_['density']):]  # 扣掉的值

            _posi_have = [[int(i / services_num), int(i % services_num)] for i in qos_list_have]  # 留下来的位置
            _posi_drop = [[int(i / services_num), int(i % services_num)] for i in qos_list_drop]  # 扣掉的位置

            for pos in _posi_drop:
                x_qos_lost[pos[0], pos[1], t] = 0
            posi_have.append(_posi_have)
            posi_drop.append(_posi_drop)
        densi_qos = x_qos_lost

        mf_U, mf_S = NMF.predict(densi_qos[:, :, 0], para_)
        predmat = np.dot(mf_U, mf_S.T)
        target = initial_qos[:, :, 0]
        mse, mae, rmse = func.comput_result(predmat, target)
        print(mae)


if __name__ == '__main__':
    initial_para = func.init_para()
    initial_para['net'] = 'AECNN'
    initial_para['device'] = 'cuda:0'
    initial_para['dataset_path2'] = 'dataset#1'
    initial_para['qos_attribute'] = 'rt'
    initial_para['density'] = 0.05
    initial_para['split'] = False
    initial_para['run_speed'] = 0
    test_nmf_for_wsdream(initial_para)
