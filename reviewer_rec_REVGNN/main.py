import world
import torch
import time
import dataloader
from dataloader import BasicDataset
from torch import nn
import numpy as np
import scipy.sparse as sp
from torch.nn import functional as F
import world
import utils
from world import cprint
import json
from tensorboardX import SummaryWriter
import time
from os.path import join
import register
from register import dataset

# 训练函数，使用 BCE 损失
def BCE_train_original(dataset, recommend_model, epoch, optimizer, w=None):
    Recmodel = recommend_model
    Recmodel.train()

    S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()  # 用户id
    posItems = torch.Tensor(S[:, 1]).long()  # 正样本
    negItems = torch.Tensor(S[:, 2]).long()  # 负样本

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bce_batch_size'] + 1
    aver_loss = 0.
    
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            utils.minibatch(users, posItems, negItems, batch_size=world.config['bce_batch_size'])):
        
        # 计算 BCE 损失
        loss = Recmodel.bce_loss(batch_users, batch_pos, batch_neg)
        aver_loss += loss.item()
        
        # 反向传播和优化步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失到 TensorBoard
        if world.tensorboard and w is not None:
            w.add_scalar(f'BCELoss/BCE', loss.item(), epoch * int(len(users) / world.config['bce_batch_size']) + batch_i)
    
    aver_loss = aver_loss / total_batch
    return f"loss{aver_loss:.3f}"

def main():
    # 设置种子
    utils.set_seed(world.seed)
    print(">>SEED:", world.seed)

    # 初始化模型
    world.model_name = 'lgn'
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)

    # 初始化优化器
    optimizer = torch.optim.Adam(Recmodel.parameters(), lr=world.config['lr'])

    # 初始化 TensorBoard 写入器
    writer = SummaryWriter(log_dir=world.config.get('tensorboard_log_dir', 'runs'))

    # 训练模型
    train_epochs = world.config.get('train_epochs', 200)
    for epoch in range(1, train_epochs + 1):
        loss_info = BCE_train_original(dataset, Recmodel, epoch, optimizer, writer)
        print(f"Epoch {epoch}: {loss_info}")

    if writer is not None:
        writer.close()

    # 获取真实用户和物品的 ID
    user_ids = dataset.trainUser  # 获取训练集中所有用户的真实ID
    item_ids = dataset.trainItem  # 获取训练集中所有物品的真实ID

    user_indices = torch.Tensor(user_ids).long().to(world.device)  # 用户的真实ID
    item_indices = torch.Tensor(item_ids).long().to(world.device)  # 物品的真实ID

    # 获取嵌入
    user_embeddings, pos_embeddings, neg_embeddings, item_embeddings = Recmodel.getEmbedding(user_indices, item_indices, item_indices)

    # 将用户和物品的嵌入转化为字典形式，并保存真实的ID
    user_embedding_dict = {str(user_id.item()): user_embeddings[i].cpu().tolist() for i, user_id in enumerate(user_indices)}
    item_embedding_dict = {str(item_id.item()): item_embeddings[i].cpu().tolist() for i, item_id in enumerate(item_indices)}

    # 保存嵌入为 JSON 文件
    with open('user_embeddings_real_ids.json', 'w') as f:
        json.dump(user_embedding_dict, f, indent=4)

    with open('item_embeddings_real_ids.json', 'w') as f:
        json.dump(item_embedding_dict, f, indent=4)

    print("User and Item embeddings with real IDs saved successfully.")

if __name__ == '__main__':
    main()
