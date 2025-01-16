
import torch
from torch import nn
import numpy as np
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
import scipy.sparse as sp
import torch.nn.functional as F
from  RecommendationSystem import RecommendationSystem
from tqdm import tqdm  # 确保你导入的是 tqdm 的类
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from random import shuffle,choice
import numpy as np
import networkx as nx
import scipy.sparse
from scipy.linalg import fractional_matrix_power, inv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random
class ClusterLayer:
    def __init__(self, n_clusters, embedding_dim):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.kmeans = KMeans(n_clusters=n_clusters)

    def fit(self, embeddings):
        embeddings = embeddings.detach().cpu().numpy()
        self.kmeans.fit(embeddings)

    def get_cluster_labels(self, embeddings):
        embeddings = embeddings.detach().cpu().numpy()
        cluster_labels = self.kmeans.predict(embeddings)
        return cluster_labels

    def get_cluster_centers(self):
        return torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float32)
    
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


class LightGCN(RecommendationSystem):
    def __init__(self, conf,logger):
        super(LightGCN, self).__init__(conf,logger)
        self.config = conf
        args = self.config['LightGCN']
        self.n_layers = int(args['n_layer'])
        self.norm_adj=self.data.norm_adj
        self.Graph=TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def computer(self,embedding_user, embedding_item):#图卷积传播   
        users_emb = embedding_user.weight
        items_emb = embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]       
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        user_all_embeddings =  light_out[:self.data.user_num]
        item_all_embeddings =  light_out[self.data.user_num:]
        return  user_all_embeddings, item_all_embeddings
    
    def forward(self, embedding_user, embedding_item):
        rec_user_emb, rec_item_emb = self.computer(embedding_user, embedding_item)
        scores = torch.matmul(rec_user_emb, rec_item_emb.T)  
        scores = torch.relu(scores)
        labels = torch.tensor(self.data.interaction_mat.toarray()).float().to(self.device)
        loss = F.binary_cross_entropy(scores, labels, reduction='mean')
        print("LightGCN_loss:", loss)
        return loss, rec_user_emb, rec_item_emb

  
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)


class Model(nn.Module):
    def __init__(self, n_in, n_h, nb_classes):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = Readout()

        self.sigm = nn.Sigmoid()
        self.fc = nn.Linear(n_h, nb_classes)
        self.sigm_fc = nn.Sigmoid()

    def forward(self, seq1, seq2, diff, sparse):
        # 计算图卷积输出
        h_mask = self.gcn2(seq2, diff, sparse)
        h_2 = self.gcn2(seq1, diff, sparse)
        
        h_mask = torch.relu(h_mask[0].unsqueeze(0))  # 对 h_mask 应用 ReLU 激活
        h_2 = torch.relu(h_2[0].unsqueeze(0))  # 对 h_2 应用 ReLU 激活

        return h_mask, h_2

    def embed(self, seq1, diff, sparse):
        h_2 = self.gcn2(seq1, diff, sparse)
        h_2 = torch.relu(h_2[0].unsqueeze(0))  # 对 h_2 应用 ReLU 激活
        return h_2.detach()

class fc_layer(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(fc_layer, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret
    
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()   

class UCGL(RecommendationSystem):
    def __init__(self, conf, logger):
        super(UCGL, self).__init__(conf, logger)
        self.alpha = 0.0001  # Regularization parameter
        self.mvg = Model(n_in=932, n_h=128, nb_classes=2)
        n_clusters = 5
        n_h = 128
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_h))
        self.bn = nn.BatchNorm1d(64, affine=False)
        self.f=nn.Linear(932,64)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def embed(self, seq, adj, diff, sparse):
        seq = seq.to(self.adj.device)  # Ensure seq is on the same device as adj
        h_1 = self.mvg.gcn1(seq, adj, sparse)
        h = self.mvg.gcn2(seq, diff, sparse)
        return (h + h_1).detach()

    def compute_q(self, bf, mask_fts, bd, sparse):
        h_mask, h = self.mvg(bf, mask_fts, bd, sparse)
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(h.reshape(-1, h.shape[2]).unsqueeze(1) - self.cluster_layer, 2),
            2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return h_mask, h, q
    
    def off_diagonal(self,x):
    # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
   
    def forward(self, user_emb, item_emb):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sparse=False
        features = torch.cat([user_emb, item_emb], dim=0)
        adj = self.data.ui_adj.toarray()  # Convert to dense matrix (NumPy array)
        features = features.clone().detach().to(device).unsqueeze(0)
        adj = torch.tensor(adj[np.newaxis], dtype=torch.float).to(device)
        mask_num = 200
        features_mask = features.clone() 

        for i in range(features_mask.shape[0]):
            idx = torch.randint(0, features_mask.shape[1], (mask_num,))
            features_mask[i, idx] = 0  # Set these features to 0 (mask them)
        features_mask_array = np.array(features_mask.squeeze(0).detach().cpu())
        features_mask=torch.tensor( features_mask_array[np.newaxis], dtype=torch.float).to(device)
        h2 = self.mvg.embed(features, adj, sparse=False)
        kmeans = KMeans(n_clusters=6)
        y_pred = kmeans.fit_predict(h2.data.squeeze().cpu().numpy()) 
        self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

        h_mask, h_2_sour, q = self.compute_q(features, features_mask, adj, sparse)
        p = target_distribution(q)
        y_pred = kmeans.fit_predict(h2.data.squeeze().cpu().numpy()) 
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')

        temperature = 0.5
        y_sam = torch.LongTensor(y_pred) # Ensure y_sam is on the same device
        neg_size = 2
        class_sam = []
        for m in range(np.max(y_pred) + 1):
            class_del = (y_sam.cpu() != m)  # Select nodes that are not of class `m
            class_neg = torch.nonzero(class_del).squeeze()  # Efficient selection
            neg_sam_id = torch.randint(len(class_neg), (neg_size,))
            class_sam.append(class_neg[neg_sam_id])
        out = h_2_sour.squeeze()
        out = out / out.max()  # 对 out 进行归一化
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        neg_samp = torch.zeros(neg.shape[0], int(neg_size))
        for n in range(np.max(y_pred) + 1):
            neg_samp[np.where(y_sam.cpu() == n)] = neg.cpu().index_select(1, class_sam[n])[np.where(y_sam.cpu() == n)]
        neg_samp = neg_samp.cuda()
        Ng = neg_samp.sum(dim=-1)
        out_mask = h_mask.squeeze()  # Masked embeddings
        out_mask = out_mask / out_mask.max()  # 
        pos = torch.exp(torch.mm(out, out_mask.t().contiguous())).cuda()
        cl_loss = (-torch.log(pos / (pos + Ng))).mean()
        # bpr_loss = -torch.log(torch.sigmoid(pos - neg) + 1e-10).mean()

        print("  cl_loss :",  cl_loss)
        print("kl_loss:",kl_loss)
        # print("feature_loss:",feature_loss)
        loss =  kl_loss + cl_loss 
        user_all_embeddings, item_all_embeddings = torch.split(out, [self.data.user_num, self.data.item_num])
        return loss , user_all_embeddings, item_all_embeddings



    
class Interaction_Oriented_Recommendation_Decoder(RecommendationSystem):
    def __init__(self, conf,logger):
        super(Interaction_Oriented_Recommendation_Decoder, self).__init__(conf,logger)
        self.config = conf
        args = self.config['IOR_Decoder']
        self.batch_size = int(args['batch_size'])
        self.pos_weight = float(args['pos_weight'])


    def custom_collate_fn(self, batch): 
        scholar_embs = torch.stack([b['scholar_emb'] for b in batch])
        max_reviewed_len = max(len(b['reviewed_submissions']) for b in batch)
        padded_reviewed_embs = torch.zeros(len(batch), max_reviewed_len, 128).to(self.device) 
        for i, b in enumerate(batch):
            reviewed_embs = b['reviewed_submissions']
            if len(reviewed_embs) > 0:
                padded_reviewed_embs[i, :len(reviewed_embs)] = torch.stack(reviewed_embs)
        labels = torch.stack([b['label'] for b in batch])
        
        return {
            'scholar_emb': scholar_embs,
            'reviewed_ids': [b['reviewed_ids'] for b in batch], 
            'reviewed_submissions': padded_reviewed_embs,  
            'label': labels
        }
            

    def predict(self, scholar_emb, all_submission_embs, history_embs):
        batch_size = scholar_emb.shape[0]
        history_embs = torch.stack([emb.clone().to(self.device) for emb in history_embs])
        all_submission_embs = all_submission_embs.to(self.device)
        history_embs_expanded = history_embs.unsqueeze(2)  # [batch_size, num_histories, 1, emb_size]
        all_submission_embs_expanded = all_submission_embs.unsqueeze(0).unsqueeze(1)  # [batch_size, 1, num_submissions, emb_size]

        mutualinteraction_embs = torch.kron(history_embs_expanded, all_submission_embs_expanded)  # 克罗内克积
        attention_weights = self.mlp_attention(mutualinteraction_embs.view(-1, mutualinteraction_embs.size(-1)))  # 展平输入到 MLP)
        attention_weights = attention_weights.view(batch_size, -1, all_submission_embs.size(0))  # [batch_size, num_histories, num_submissions]
        attention_weights = torch.softmax(attention_weights, dim=1)  
        weighted_histories = (history_embs_expanded * attention_weights.unsqueeze(-1)).sum(dim=1)  # [batch_size, emb_size]

        scholar_emb_expanded = scholar_emb.unsqueeze(1).expand(-1, all_submission_embs.size(0), -1)  # [batch_size, num_submissions, emb_size]
        all_submission_embs_expanded = all_submission_embs.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_submissions, emb_size]
        final_representation = torch.cat((scholar_emb_expanded, weighted_histories, all_submission_embs_expanded), dim=-1)  # [batch_size, num_submissions, 3 * emb_size]
        predictions = self.mlp_final(final_representation)  # [batch_size * num_submissions, 1]
        return predictions
    


    def forward(self, users_emb, items_emb, dataset):    
        self.dataset = dataset
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.custom_collate_fn)
        total_loss = 0
        # Wrap dataloader with tqdm
        for batch in tqdm(dataloader, desc="Training", unit="batch"):
            scholar_emb = batch['scholar_emb'].to(self.device)
            reviewed_embs = batch['reviewed_submissions'].to(self.device)
            labels = batch['label'].float().to(self.device)  
            # print("reviewed_embs.shape",reviewed_embs.shape)
            labels = labels.unsqueeze(-1)  # 添加一个新的维度，变成 [1, 4000, 1]
            # labels = labels.expand(-1, -1,1)  
            num_neg = (labels == 0).sum().item()  
            num_pos = (labels == 1).sum().item()  
            pos_weight = torch.tensor(min(num_neg / num_pos, self.pos_weight)).to(self.device)
            predictions = self.predict(scholar_emb, users_emb.to(self.device), reviewed_embs)
            loss = F.binary_cross_entropy(predictions, labels, weight=pos_weight, reduction='mean')
            # print(loss)
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Decoder_Loss = {avg_loss:.8f}")  # 打印平均损失
        return avg_loss


        



   