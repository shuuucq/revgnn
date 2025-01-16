"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""
import torch
from torch import nn
from torch import optim
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score
import os
import json
import random
# ====================Concat Embedding=============================
class EmbeddingCombiner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbeddingCombiner, self).__init__()
        self.reducer = nn.Linear(input_dim, output_dim)

    def forward(self, lightgcn_emb, scibert_emb):
        combined_emb = torch.cat([lightgcn_emb, scibert_emb], dim=-1)
        reduced_emb = self.reducer(combined_emb)
        return reduced_emb  
    
def set_random_seed(seed=42):
    random.seed(seed)  # 设置 Python 内置的随机数生成器种子
    np.random.seed(seed)  # 设置 NumPy 随机数生成器种子
    torch.manual_seed(seed)  # 设置 CPU 上的 PyTorch 随机数生成器种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 CUDA 上的随机数生成器种子
        torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 上的随机数生成器种子
    torch.backends.cudnn.deterministic = True  # 使 cudnn 在算法上确定性
    torch.backends.cudnn.benchmark = False  # 禁用 cudnn 基准模式




# ====================end Metrics=============================
# =========================================================
@staticmethod
def load_sci_embeddings(filepath):
 # 假设嵌入数据存储在 JSON 文件中
    with open(filepath, 'r') as f:
        data = json.load(f)
    embedding_dict = {key: torch.tensor(value, dtype=torch.float32) for key, value in data.items()}
    return embedding_dict

@staticmethod
def load_data(train_file):
    training_data = []
    with open(train_file, 'r') as f:
        for line in f:
            data = line.strip().split()  # Assuming user and items are space-separated
            user = data[0]  # First column is user ID
            items = data[1:]  # Remaining columns are item IDs
            for item in items:
                training_data.append((user, item))  # Add each user-item pair to training data
    return training_data