from data.data import Data
from util.logger import Log
from os.path import abspath
from time import strftime, localtime, time
from data.ui_graph import Interaction

import os
import time
import torch.nn as nn
from os.path import abspath

class Recommender(nn.Module):
    def __init__(self, conf, training_set,val_set, test_set, **kwargs):
        # 初始化 nn.Module
        super(Recommender, self).__init__()
        
        # 保存配置和数据集
        self.config = conf
        self.training_set = training_set
        self.test_set = test_set
        self.test_set = val_set
        self.data = Data(self.config, training_set, val_set, test_set)
       
        # 模型参数
        model_config = self.config['model']
        self.model_name = model_config['name']
        self.ranking = self.config['item.ranking.topN']
        self.emb_size = int(self.config['embedding.size'])
        self.maxEpoch = int(self.config['max.epoch'])
        self.batch_size = int(self.config['batch.size'])
        self.lRate = float(self.config['learning.rate'])
        self.reg = float(self.config['reg.lambda'])
        self.output = self.config['output']
        self.data = Interaction(conf, training_set, val_set,test_set)
        # 日志记录
        current_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        self.model_log = Log(self.model_name, f"{self.model_name} {current_time}")

        self.result = []
        self.recOutput = []

    def initializing_log(self):
        """初始化日志文件，记录模型配置"""
        self.model_log.add('### model configuration ###')
        config_items = self.config.config
        for k in config_items:
            self.model_log.add(f"{k}={str(config_items[k])}")

    def print_model_info(self):
        """打印模型信息"""
        print('Model:', self.model_name)
        print('Training Set:', abspath(self.config['training.set']))
        print('Test Set:', abspath(self.config['test.set']))
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Learning Rate:', self.lRate)
        print('Batch Size:', self.batch_size)
        print('Regularization Parameter:', self.reg)

        model_name = self.config['model']['name']
        if self.config.contain(model_name):
            args = self.config[model_name]
            parStr = '  '.join(f"{key}:{args[key]}" for key in args)
            print('Specific parameters:', parStr)

    # 以下是需要由具体模型类实现的方法，暂时保持为空实现
    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self, rec_list):
        pass

    def execute(self):
        """执行完整的推荐过程：初始化日志、打印模型信息、训练、测试和评估"""
        self.initializing_log()
        self.print_model_info()
        print('Initializing and building model...')
        self.build()
        print('Training Model...')
        self.train()
        print('Testing...')
        rec_list = self.test()
        print('Evaluating...')
        self.evaluate(rec_list)
