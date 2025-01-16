import numpy as np 
from collections import defaultdict
from data.data import Data
from data.graph import Graph
import scipy.sparse as sp
import pandas as pd

class Interaction(Data, Graph):
    def __init__(self, conf, training,val,test):
        Graph.__init__(self)
        Data.__init__(self, conf, training,val,test)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.val_set = defaultdict(dict)
        self.val_set_item = set()
        self.test_set = defaultdict(dict)
        self.test_set_item = set()

        self.__generate_set()  # 生成训练集和测试集的用户-物品对
        self.user_num = len(self.user)  # 用户总数
        self.item_num = len(self.item)  # 物品总数
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()


    def __generate_set(self):
        for user, item in self.training_data:
            if user not in self.user:
                user_id = len(self.user)
                self.user[user] = user_id
                self.id2user[user_id] = user
            if item not in self.item:
                item_id = len(self.item)
                self.item[item] = item_id
                self.id2item[item_id] = item
            self.training_set_u[user][item] = 1
            self.training_set_i[item][user] = 1

        for user, item in self.test_data:
            if user in self.user and item in self.item:
                self.test_set[user][item] = 1
                self.test_set_item.add(item)

        for user, item in self.val_data:
            if user in self.user and item in self.item:
                self.val_set[user][item] = 1
                self.val_set_item.add(item)


    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        n_nodes = self.user_num + self.item_num
        user_np = np.array([self.user[pair[0]] for pair in self.training_data])  # 使用映射后的用户索引
        item_np = np.array([self.item[pair[1]] for pair in self.training_data]) + self.user_num  # 使用映射后的物品索引
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        user_np_keep, item_np_keep = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_mat.shape[0])),
                                shape=(adj_mat.shape[0] + adj_mat.shape[1], adj_mat.shape[0] + adj_mat.shape[1]),
                                dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        row = np.array([self.user[pair[0]] for pair in self.training_data])
        col = np.array([self.item[pair[1]] for pair in self.training_data])
        entries = np.ones(len(row), dtype=np.float32)
        return sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)

    def get_user_id(self, u):
        return self.user.get(u)

    def get_item_id(self, i):
        return self.item.get(i)

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)
    
    def val_size(self):
        return len(self.val_set), len(self.val_set_item), len(self.val_data)
    
    def contain(self, u, i):
        return u in self.user and i in self.training_set_u[u]

    def contain_user(self, u):
        return u in self.user

    def contain_item(self, i):
        return i in self.item

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())
    def get_positive_item(self, u):
        # 返回用户 u 的一个正样本物品的内部索引
        # 这里简单地返回用户交互过的第一个物品
        history_item_ids, _ = self.user_rated(u)
        if len(history_item_ids) > 0:
            return history_item_ids[0]
        else:
            # 如果没有正样本，随机返回一个物品索引
            return np.random.randint(self.item_num)

    def row(self, u):
        k, v = self.user_rated(self.id2user[u])
        vec = np.zeros(self.item_num, dtype=np.float32)
        for item, rating in zip(k, v):
            vec[self.item[item]] = rating
        return vec

    def col(self, i):
        k, v = self.item_rated(self.id2item[i])
        vec = np.zeros(self.user_num, dtype=np.float32)
        for user, rating in zip(k, v):
            vec[self.user[user]] = rating
        return vec

    def matrix(self):
        m = np.zeros((self.user_num, self.item_num), dtype=np.float32)
        for u, u_id in self.user.items():
            vec = np.zeros(self.item_num, dtype=np.float32)
            k, v = self.user_rated(u)
            for item, rating in zip(k, v):
                vec[self.item[item]] = rating
            m[u_id] = vec
        return m
