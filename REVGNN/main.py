import os
import torch
import logging
from logging.handlers import TimedRotatingFileHandler
from RecommendationSystem import RecommendationSystem
from model import LightGCN,UCGL, Interaction_Oriented_Recommendation_Decoder
from util.conf import ModelConf
from datetime import datetime

# 设置日志保存的文件夹
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 创建日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 获取当前日期
current_date = datetime.now().strftime("%Y-%m-%d")
log_filename = os.path.join(log_dir, f'model_training_{current_date}.log')

# 设置按时间划分的日志文件处理器（按天分割）
file_handler = TimedRotatingFileHandler(log_filename, when='midnight', interval=1, backupCount=7)
file_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将 handler 添加到 logger
logger.addHandler(file_handler)

GPU = torch.cuda.is_available()

if __name__ == '__main__':
    config = ModelConf('Conf.yaml')  # Assuming this is defined
    device = torch.device("cuda" if GPU else "cpu")
    
    logger.info(f"Configuration loaded: {config.config}")  # 打印 config 字典内容，而不是 config 对象
    trainer = RecommendationSystem(config, logger)
    trainer.initialize_models(LightGCN, UCGL, Interaction_Oriented_Recommendation_Decoder)
    trainer.to(device)
    trainer.train_model(config.config['train_epoch'])  # Train for 100 epochs
