
import torch
import time
from os.path import join
import json
from util.conf import ModelConf
from torch.utils.data import Dataset, DataLoader
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation
from torch import nn
from utils import *
from tqdm import tqdm  # 确保这是在顶部导入的唯一 tqdm 相关代码
import pickle
import torch.nn.init as init
import logging

GPU = torch.cuda.is_available()
set_random_seed(42)
class InteractionDataset(Dataset):
    def __init__(self, reviewed_histories, interaction_matrix):
        self.reviewed_histories = reviewed_histories
        self.interaction_matrix = interaction_matrix
        self.samples = []

        for scholar_id, data in reviewed_histories.items():
            scholar_emb = data['scholar_emb']  
            reviewed_submissions = data['reviewed_submissions']
            reviewed_ids = data['reviewed_ids']
            self.samples.append((scholar_id, scholar_emb,reviewed_ids,reviewed_submissions))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scholar_id, scholar_emb, reviewed_ids, reviewed_submissions = self.samples[idx]
        label = torch.zeros(self.interaction_matrix.shape[0], dtype=torch.float32)
        for i,submission_id in enumerate(reviewed_ids):
            if scholar_id < self.interaction_matrix.shape[1] and submission_id < self.interaction_matrix.shape[0]:
                label[submission_id] = torch.tensor(self.interaction_matrix[submission_id, scholar_id], dtype=torch.float32)

        return {
            'scholar_emb': scholar_emb,
            'reviewed_ids': reviewed_ids, 
            'reviewed_submissions': reviewed_submissions,  
            'label': label
        }


class RecommendationSystem(nn.Module):
    def __init__(self, config,logger):
        super(RecommendationSystem, self).__init__() 
        self.config = config
        self.device = torch.device('cuda:{}'.format(0) if GPU else "cpu")
        self.paper_embedding_dict = load_sci_embeddings(self.config['paper_embeddings'])
        self.reviewer_embedding_dict = load_sci_embeddings(self.config['reviewer_embeddings'])
        self.training_set = load_data(self.config['train'])
        self.val_set = load_data(self.config['valid'])
        self.test_set = load_data(self.config['test'])
        self.LightGCN_emb_size=self.config['LightGCN_emb_size']
        self.emb_size=128
        self.hidden_size=self.config['hidden_size']
        self.ranking = self.config['item.ranking.topN']
        self.logger = logger 

        self.data = Interaction(self.config, self.training_set,self.val_set,self.test_set)
        self.bestPerformance = []
        self.topN = [int(num) for num in self.ranking]
        self.max_N = max(self.topN)
        self.to(self.device)
        self.f = nn.Sigmoid()

        self.mlp_attention = nn.Sequential(
            nn.Linear(self.emb_size*self.emb_size , 36),
            nn.PReLU(),
            nn.Linear(36, 1)
        )

        self.mlp_final = nn.Sequential(
            nn.Linear(self.emb_size*3, 200),
            nn.PReLU(),
            nn.Linear(200, 1),
            nn.PReLU(),
            nn.Sigmoid()
        )
        
        # self.mlp_attention = nn.Sequential(
        #     nn.Linear(self.emb_size*self.emb_size , 36),
        #     nn.ReLU(),
        #     nn.Linear(36, 1)
        # )

        # self.mlp_final = nn.Sequential(
        #     nn.Linear(self.emb_size*3, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 1),
        #     nn.ReLU(),
        #     nn.Sigmoid()
        # )

        self.init_weights()

    def init_weights(self):
            # 初始化用户和物品的嵌入层
            self.users_emb = torch.nn.Embedding(num_embeddings=self.data.user_num, embedding_dim=self.LightGCN_emb_size)
            self.items_emb = torch.nn.Embedding(num_embeddings=self.data.item_num, embedding_dim=self.LightGCN_emb_size)

            # 使用正态分布初始化嵌入层的权重
            nn.init.normal_(self.users_emb.weight, std=0.1)
            nn.init.normal_(self.items_emb.weight, std=0.1)

            self.apply(self.init_layer_weights)

    def init_layer_weights(self, m):
        if isinstance(m, nn.Linear):
            # Use Kaiming initialization instead of Xavier initialization
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize biases to 0

    def initialize_models(self,LightGCN,UCGL,Decoder):
        self.LightGCN_model = LightGCN(conf=ModelConf('Conf.yaml'),
                                     logger=self.logger ).to(self.device)
        self.UCGL_model = UCGL(conf=ModelConf('Conf.yaml'),logger=self.logger
                                    ).to(self.device)
        self.Decoder_model = Decoder(conf=ModelConf('Conf.yaml'),logger=self.logger
                                    ).to(self.device)

    def create_optimizer(self):
        # 为每个模型的参数定义不同的学习率
        lightgcn_params = list(self.LightGCN_model.parameters())
        ucgl_params = list(self.UCGL_model.parameters())
        decoder_params = list(self.Decoder_model.parameters())
        # combiner_params = list(self.combiner.parameters())

        optimizer_params = [
            {'params': lightgcn_params, 'lr': self.config['LightGCN_lr']},
            {'params': ucgl_params, 'lr': self.config['Simgcl_lr']},
            {'params': decoder_params, 'lr':self.config['decoder_lr']},
            {'params': self.users_emb.weight, 'lr': 0.002},  # 确保嵌入层的权重在优化器中
            {'params': self.items_emb.weight, 'lr': 0.002}   # 确保嵌入层的权重在优化器中
        ]

        # 创建优化器
        return torch.optim.Adam(optimizer_params)


    def generate_embedding_dicts(self, light_users_emb, light_items_emb):
        LightGCN_user_embedding_dict = {
            str(self.data.id2user[user_id]): light_users_emb[i]
            for i, user_id in enumerate(self.data.user.values())
        }
        LightGCN_item_embedding_dict = {
            str(self.data.id2item[item_id]): light_items_emb[i]
            for i, item_id in enumerate(self.data.item.values())
        }
        return LightGCN_user_embedding_dict, LightGCN_item_embedding_dict

    
    def combine_embeddings(self, lightgcn_emb, scibert_emb):
        scibert_emb = scibert_emb.to(lightgcn_emb.device)
        combined_emb = torch.cat([lightgcn_emb, scibert_emb], dim=-1)
        # combined_emb = self.combiner(lightgcn_emb, scibert_emb)
        return combined_emb

    
    def combine_embeddings_dicts(self, lightgcn_dict, scibert_dict):
        combined_embeddings = []
        keys = []
        # 假设 lightgcn_dict 和 scibert_dict 具有相同的键集合
        for key in lightgcn_dict:
            if key in scibert_dict:
                # 直接使用张量，不转换为列表
                lightgcn_emb = lightgcn_dict[key].to(self.device).unsqueeze(0)  # (1, emb_dim_lightgcn)
                scibert_emb = scibert_dict[key].to(self.device).unsqueeze(0)    # (1, emb_dim_scibert)
                combined_emb = self.combine_embeddings(lightgcn_emb, scibert_emb)  # (1, combined_emb_dim)
                combined_embeddings.append(combined_emb)
                keys.append(key)
            else:
                print(f"Warning: No matching embedding in SciBERT for ID {key}")
        
        if not combined_embeddings:
            raise ValueError("No embeddings were combined. Check your dictionaries.")
        
        # 拼接所有合并后的嵌入，形成一个批量张量
        combined_embeddings_tensor = torch.cat(combined_embeddings, dim=0)  # 形状: (num_keys, combined_emb_dim)
        return combined_embeddings_tensor


    def get_reviewer_histories_embeddings(self, simgcl_user_emb, simgcl_item_emb):
        reviewed_histories = {}
        for item_id in self.data.item.values():  # item scholar
            external_item_id = self.data.id2item[item_id] 
            item_emb = simgcl_item_emb[item_id]
            external_user_ids, _ = self.data.item_rated(external_item_id)  # scholar交互过的submission
            user_ids = []  # 修改为包含用户嵌入和用户 ID 的列表
            user_embs = []
            for external_user_id in external_user_ids:
                internal_user_id = self.data.user[external_user_id]
                user_emb = simgcl_user_emb[internal_user_id]
                user_ids.append(internal_user_id)
                user_embs.append(user_emb)

            if user_ids:  # 仅在有用户数据的情况下添加到字典
                reviewed_histories[item_id] = {
                    'scholar_emb': item_emb, 
                    'reviewed_ids':user_ids,
                    'reviewed_submissions':user_embs  # 将包含用户 ID 和嵌入的列表添加到字典
                } 
        return reviewed_histories

        
    def train_model(self, epochs):
        # 开启梯度异常检测
        torch.autograd.set_detect_anomaly(True)
        self.joint_optimizer = self.create_optimizer()
        self.train()
        
        for epoch in range(epochs):
            self.joint_optimizer.zero_grad()

            lightgcn_loss, light_users_emb, light_items_emb = self.LightGCN_model(self.users_emb, self.items_emb)
            LightGCN_user_embedding_dict, LightGCN_item_embedding_dict = self.generate_embedding_dicts(light_users_emb, light_items_emb)
            combined_users_emb = self.combine_embeddings_dicts(LightGCN_user_embedding_dict, self.paper_embedding_dict)
            combined_items_emb = self.combine_embeddings_dicts(LightGCN_item_embedding_dict, self.reviewer_embedding_dict)

            simgcl_loss, sim_user_emb, sim_item_emb = self.UCGL_model(combined_users_emb, combined_items_emb)
            self.logger.info(f"SimGCL Loss: {simgcl_loss.item()}")
            reviewed_histories = self.get_reviewer_histories_embeddings(sim_user_emb, sim_item_emb)

            interaction_dataset = InteractionDataset(
                reviewed_histories=reviewed_histories,
                interaction_matrix=self.data.interaction_mat
            )
            decoder_loss = self.Decoder_model(sim_user_emb, sim_item_emb, dataset=interaction_dataset)
            self.logger.info(f"Decoder Loss: {decoder_loss}")

            total_loss = lightgcn_loss + decoder_loss + simgcl_loss
            total_loss.backward()

            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(self.Decoder_model.parameters(), max_norm=1.0)

            self.joint_optimizer.step()

            # 每10轮进行评估
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    self.fast_evaluation(epoch, sim_user_emb, sim_item_emb,dataset=interaction_dataset)


    # def predict(self,u,dec_user_emb,reviewed_histories):  
    #     user_id = self.data.get_user_id(u)
    #     print(user_id)
    #     submission_emb = dec_user_emb[user_id].clone().detach().float().to(self.device)
    #     predictions = []

    #     for item_id, item_data in enumerate(reviewed_histories.values()):
    #         scholar_emb = torch.tensor(item_data['scholar_emb']).to(self.device)
    #         history_embs = [torch.tensor(emb).to(self.device) for emb in item_data['reviewed_submissions']]
    #         history_embs_tensor = torch.stack(history_embs)  

    #         attention_weights = []
    #         interaction_emb = torch.kron(history_embs_tensor, submission_emb)
    #         attention_weights = self.mlp_attention(interaction_emb)
    #         attention_weights = torch.softmax(attention_weights.squeeze(-1), dim=0) 
    #         attention_weights = attention_weights.unsqueeze(1) 

    #         weighted_sum = (history_embs_tensor * attention_weights).sum(dim=0)
    #         weighted_hist = weighted_sum.view(1, -1) 
    #         scholar_emb = scholar_emb.unsqueeze(0) if scholar_emb.dim() == 1 else scholar_emb
    #         submission_emb = submission_emb.unsqueeze(0) if submission_emb.dim() == 1 else submission_emb

    #         combined_representation = torch.cat((scholar_emb, weighted_hist, submission_emb), dim=1)  # 应该是 [1, 384]
    #         prediction = self.mlp_final(combined_representation)
    #         # print("1:", prediction.shape)
    #         # print(" prediction :", prediction )
    #         predictions.append(prediction)

    #     predictions = torch.cat(predictions, dim=0)
    #     return predictions.view(-1, 1)




    # def predict(self,u,dec_user_emb,dec_item_emb): 
    #     user_id = self.data.get_user_id(u)
    #     submission_emb = dec_user_emb[user_id].clone().detach().float().to(self.device) 
    #     rating = self.f(torch.matmul(submission_emb, dec_item_emb.t()))
    #     return rating

    def test(self, sim_user_emb,sim_item_emb):
        rec_list = {}
        user_count = len(self.data.test_set)  

        reviewed_histories = self.get_reviewer_histories_embeddings(sim_user_emb, sim_item_emb)

        for user in tqdm(self.data.test_set, total=user_count, desc="Processing users"):
            # candidates = self.predict(user, dec_user_emb,dec_item_emb)
            user_id = self.data.get_user_id(user)
            submission_emb = sim_user_emb[user_id].clone().detach().float().to(self.device)
            candidates = self.Decoder_model.test(submission_emb,reviewed_histories)
            rated_list, _ = self.data.user_rated(user)  
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8 
            if isinstance(candidates, torch.Tensor):
                candidates_np = candidates.detach().cpu().numpy().ravel() 
            else:
                candidates_np = candidates.ravel()  
            ids, scores = find_k_largest(self.max_N, candidates_np)
            print(ids)
            print(scores)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))

        print('')  # 换行
        return rec_list


    def evaluate(self, rec_list):
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for user in self.data.val_set:
            line = user + ':' + ''.join(
                f" ({item[0]},{item[1]}){'*' if item[0] in self.data.test_set[user] else ''}"
                for item in rec_list[user]
            )
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        out_dir = self.output
        file_name = f"{self.config['model']['name']}@{current_time}-top-{self.max_N}items.txt"
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        file_name = f"{self.config['model']['name']}@{current_time}-performance.txt"
        self.result = ranking_evaluation(self.data.val_set, rec_list, self.topN)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        print(f'The result of {self.model_name}:\n{"".join(self.result)}')

    def fast_evaluation(self, epoch,dec_user_emb,dec_item_emb):
        print('Evaluating the model...')
        rec_list = self.test(dec_user_emb,dec_item_emb)
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])

        performance = {k: float(v) for m in measure[1:] for k, v in [m.strip().split(':')]}

        if self.bestPerformance:
            count = sum(1 if self.bestPerformance[1][k] > performance[k] else -1 for k in performance)
            if count < 0:
                self.bestPerformance = [epoch + 1, performance]
                # self.save()
        else:
            self.bestPerformance = [epoch + 1, performance]
            # self.save()

        print('-' * 80)
        print(f'Real-Time Ranking Performance (Top-{self.max_N} Item Recommendation)')
        measure_str = ', '.join([f'{k}: {v}' for k, v in performance.items()])
        print(f'*Current Performance*\nEpoch: {epoch + 1}, {measure_str}')
        bp = ', '.join([f'{k}: {v}' for k, v in self.bestPerformance[1].items()])
        print(f'*Best Performance*\nEpoch: {self.bestPerformance[0]}, {bp}')
        print('-' * 80)
        self.save_results_to_file(epoch, performance)
        return measure

    def save_results_to_file(self, epoch, performance):
        with open('evaluation_results.txt', 'a') as f:
            f.write(f'Epoch: {epoch + 1}\n')
            for k, v in performance.items():
                f.write(f'{k}: {v}\n')
            f.write('-' * 80 + '\n')