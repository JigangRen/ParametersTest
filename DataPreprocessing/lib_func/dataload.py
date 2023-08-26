import numpy as np


# ======================================================#
# Function to load the dataset
# ======================================================#
def load(_para):
    if _para['dataset_path2'] == 'dataset#1':
        datafile = _para['dataset_path1'] + _para['dataset_path2'] + '/' + _para['qos_attribute'] + 'Matrix.txt'
        data_matrix = np.loadtxt(datafile)
    elif _para['dataset_path2'] == 'dataset#2':
        datafile = _para['dataset_path1'] + _para['dataset_path2'] + '/' + _para['qos_attribute'] + 'data.txt'
        data_matrix = -1 * np.ones((142, 4500, 64))
        fid = open(datafile, 'r')
        for line in fid:
            data = line.split(' ')
            rt = float(data[3])
            if rt > 0:
                data_matrix[int(data[0]), int(data[1]), int(data[2])] = rt
        fid.close()
        data_matrix = preprocess(data_matrix, _para)
    else:
        print('data load error!')
        return -1
    return data_matrix


# ======================================================#
# Function to preprocess the dataset which
# deletes the invalid values
# ======================================================#
def preprocess(matrix, _para):
    if _para['qos_attribute'] == 'rt':
        matrix = np.where(matrix == 0, -0.01, matrix)
        matrix = np.where(matrix >= 19.9, -0.01, matrix)
    elif _para['qos_attribute'] == 'tp':
        matrix = np.where(matrix == 0, -0.01, matrix)
    return matrix


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom == 0:
        sim = 0
    else:
        sim = num / denom
    return sim

# pcc
def pcc(x, y):
    # 计算均值
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # 计算协方差
    cov_xy = np.sum((x - mean_x) * (y - mean_y))
    cov_xx = np.sum((x - mean_x) ** 2)
    cov_yy = np.sum((y - mean_y) ** 2)

    # 计算PCC
    pcc = cov_xy / (np.sqrt(cov_xx) * np.sqrt(cov_yy))

    return pcc

# jaccard_similarity
def jaccard_similarity(set1, set2):
    set1=set(set1)
    set2=set(set2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / float(union) if union !=0 else 0

def matrix_sim(matrix):  # , max_len):
    item_num, dim_num = matrix.shape
    sim_mat = np.zeros([item_num, item_num])
    for i in range(item_num):
        frature1 = matrix[i, :]
        for j in range(i + 1, item_num):
            feature2 = matrix[j, :]
            sim_mat[i][j] = cos_sim(frature1, feature2)
    sim_mat = sim_mat + sim_mat.T + np.eye(item_num)
    # sim_mat = np.argsort(sim_mat)[:, (item_num - max_len):]
    return sim_mat


def bipartite_graph2tree(posi_have, shape):
    users_dict = {}
    services_dict = {}
    for item in posi_have:
        try:
            users_dict[str(item[0])]
        except KeyError:
            users_dict[str(item[0])] = [item[1]]
        else:
            users_dict[str(item[0])].append(item[1])

        try:
            services_dict[str(item[1])]
        except KeyError:
            services_dict[str(item[1])] = [item[0]]
        else:
            services_dict[str(item[1])].append(item[0])
    users_graph = {}
    for i in range(0, shape[0]):
        try:
            users_graph[str(i)]
        except KeyError:
            users_graph[str(i)] = []

        try:
            temp = users_dict[str(i)]
        except KeyError:
            continue
        else:
            for s in temp:
                try:
                    users_graph[str(i)] = users_graph[str(i)] + services_dict[str(s)]
                except KeyError:
                    continue
            users_graph[str(i)] = list(set(users_graph[str(i)]))

    services_graph = {}
    for i in range(0, shape[1]):
        try:
            services_graph[str(i)]
        except KeyError:
            services_graph[str(i)] = []
        try:
            temp = services_dict[str(i)]
        except KeyError:
            continue
        else:
            for u in temp:
                try:
                    services_graph[str(i)] = services_graph[str(i)] + users_dict[str(u)]
                except KeyError:
                    continue
            services_graph[str(i)] = list(set(services_graph[str(i)]))

    return users_graph, services_graph
