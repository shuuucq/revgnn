import world
import torch
import time
from dataloader import BasicDataset
from torch import nn
import torch.nn.functional as F
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from base.torch_interface import TorchGraphInterface
from base.recommender import Recommender

class BasicModel(nn.Module):
    def __init__(self, conf=None, data=None):
        super(BasicModel, self).__init__()
        self.conf = conf
        self.data = data

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self, conf=None, data=None):
        super(PairWiseModel, self).__init__(conf, data)

    def bpr_loss(self, users, pos, neg):
        raise NotImplementedError

class LightGCN(Recommender):
    def __init__(self, conf, train_set, test_set):
        super(LightGCN, self).__init__(conf, train_set, test_set)
        self.n_layers = int(self.config['LightGCN']['n_layer'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lRate)

    def train(self):
        for epoch in range(self.maxEpoch):
            total_loss = 0
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                user_emb, pos_emb, neg_emb = self.get_embeddings(user_idx, pos_idx, neg_idx)

                pos_scores = torch.sigmoid(torch.sum(user_emb * pos_emb, dim=1))
                neg_scores = torch.sigmoid(torch.sum(user_emb * neg_emb, dim=1))

                bce_loss = self.compute_bce_loss(pos_scores, neg_scores)
                batch_loss = bce_loss + self.regularization_loss(user_idx, pos_idx, neg_idx)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                total_loss += batch_loss.item()

                if n % 100 == 0:
                    print(f'Epoch {epoch+1}, Batch {n}, Loss: {batch_loss.item()}')

            self.user_emb, self.item_emb = self.model()
            print(f'Epoch {epoch+1}, Total Loss: {total_loss}')

    def get_embeddings(self, user_idx, pos_idx, neg_idx):
        rec_user_emb, rec_item_emb = self.model()
        user_emb = rec_user_emb[user_idx]
        pos_emb = rec_item_emb[pos_idx]
        neg_emb = rec_item_emb[neg_idx]
        return user_emb, pos_emb, neg_emb

    def compute_bce_loss(self, pos_scores, neg_scores):
        pos_loss = torch.nn.functional.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        neg_loss = torch.nn.functional.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        return pos_loss + neg_loss

    def regularization_loss(self, user_idx, pos_idx, neg_idx):
        return l2_reg_loss(self.reg, 
                           self.model.embedding_dict['user_emb'][user_idx], 
                           self.model.embedding_dict['item_emb'][pos_idx], 
                           self.model.embedding_dict['item_emb'][neg_idx]) / self.batch_size

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model()

    def predict(self, user_idx):
        score = torch.matmul(self.user_emb[user_idx], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

    def forward(self, user_indices, item_indices):
        return self.model(user_indices, item_indices)


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict
    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings


class SimGCL(Recommender):
    def __init__(self, conf, train_set, test_set):
        super(SimGCL, self).__init__(conf, train_set, test_set)
        args = self.config['SimGCL']
        self.cl_rate = float(args['lambda'])
        self.eps = float(args['eps'])
        self.n_layers = int(args['n_layer'])
        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)

        # 初始化最佳用户和物品嵌入
        self.best_user_emb = None
        self.best_item_emb = None
        self.user_clusters = None  # 初始化用户簇
    
    def cluster_users(self, user_emb):
        """
        基于用户嵌入生成用户簇。
        """
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=10)  # 假设划分为10个簇
        self.user_clusters = kmeans.fit(user_emb.cpu().detach().numpy())

    def sample_negative_from_clusters(self, user_indices):
        """
        基于用户簇采样负样本，确保用户和负样本来自不同的簇。
        """
        neg_samples = []
        for user_idx in user_indices:
            cluster_id = self.user_clusters.labels_[user_idx]  # 当前用户的簇 ID
            other_cluster_users = [i for i, label in enumerate(self.user_clusters.labels_) if label != cluster_id]

            # 从其他簇中随机采样一个用户作为负样本
            neg_sample_idx = np.random.choice(other_cluster_users)
            neg_samples.append(neg_sample_idx)

        return torch.tensor(neg_samples).long().to(world.device)

    def train_model(self, init_user_emb, init_item_emb): 
        model = self.model.cuda()
        total_epoch_loss = torch.tensor(0.0).to(world.device)  # 初始化总损失

        # 对用户进行聚类，生成用户簇
        self.cluster_users(init_user_emb)  # 在训练开始时基于用户嵌入进行聚类

        for epoch in range(self.maxEpoch):
            epoch_loss = torch.tensor(0.0).to(world.device)  # 每个epoch的损失
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()

                # 获取用户和物品嵌入
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                
                # 计算推荐损失 (BPR损失)
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                # 使用簇生成负样本
                neg_user_idx = self.sample_negative_from_clusters(user_idx)  # 通过簇采样负样本

                # 计算对比学习损失，正样本和负样本
                cl_loss = self.cl_rate * self.cal_cl_loss(user_idx, pos_idx, neg_user_idx)
                
                # 计算正则化损失
                total_l2_loss = l2_reg_loss(self.reg, user_emb, pos_item_emb)
                
                # 总损失
                batch_loss = rec_loss + total_l2_loss + cl_loss
                
                # 更新每个epoch的损失值
                epoch_loss += batch_loss

                if n % 100 == 0 and n > 0:
                    print(f'training: {epoch + 1}, batch {n}, rec_loss: {rec_loss.item()}, cl_loss: {cl_loss.item()}')

            # 将当前epoch的损失添加到总损失
            total_epoch_loss += epoch_loss

        # 返回整个训练过程的损失
        return total_epoch_loss / self.maxEpoch

    def cal_cl_loss(self, user_idx, pos_idx, neg_user_idx):
        """
        计算对比学习损失，包括正负样本。
        """
        # 获取用户和物品的视图
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)

        # 计算用户的正样本损失（基于不同视图）
        user_cl_loss = InfoNCE(user_view_1[user_idx], user_view_2[user_idx], 0.2)

        # 计算用户的负样本损失（基于簇）
        neg_user_view_1 = user_view_2[neg_user_idx]
        user_neg_cl_loss = InfoNCE(user_view_1[user_idx], neg_user_view_1, 0.2)

        # 计算物品的对比学习损失
        item_cl_loss = InfoNCE(item_view_1[pos_idx], item_view_2[pos_idx], 0.2)

        # 返回总的对比学习损失
        return user_cl_loss + user_neg_cl_loss + item_cl_loss


    def getUsersRating(self, users, reviewer_record_dict):
        """
        计算用户对候选物品的评分。
        """
        all_users, all_items = self.best_user_emb, self.best_item_emb
        user_indices = self.map_user_ids_to_indices(users).to(world.device)
        users_emb = all_users[user_indices]
        candidate_items = torch.arange(all_items.size(0)).to(world.device)
        candidate_emb = all_items[candidate_items]

        # 计算注意力加权评分
        scores = []
        for user_idx in user_indices:
            user_idx_str = str(int(user_idx))
            if user_idx_str in reviewer_record_dict:
                hist_items = torch.tensor([int(paper_id) for paper_id in reviewer_record_dict[user_idx_str]]).long().to(world.device)
                hist_emb = all_items[hist_items]
                similarity = torch.matmul(candidate_emb, hist_emb.T)
                attention_weights = torch.softmax(similarity, dim=1)
                weighted_hist_emb = torch.matmul(attention_weights, hist_emb)
                combined_emb = torch.cat([users_emb, candidate_emb, weighted_hist_emb], dim=-1)
            else:
                combined_emb = torch.cat([users_emb, candidate_emb, torch.zeros_like(candidate_emb)], dim=-1)
            scores.append(combined_emb)

        # 使用 MLP 计算最终分数
        W_r1, W_r2, b_r1, b_r2 = self.W_r1, self.W_r2, self.b_r1, self.b_r2
        final_scores = torch.sigmoid(W_r2(torch.sigmoid(W_r1(scores) + b_r1)) + b_r2)
        return final_scores

    def save(self):
        # 保存最佳嵌入
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model()

    def predict(self, u):
        # 基于用户嵌入进行预测
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()
    
class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings
    
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)