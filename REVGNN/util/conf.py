import os
import yaml
import logging

class ModelConf(object):
    def __init__(self, file, logger=None):
        self.config = {}
        self.logger = logger  # 将 logger 作为参数传递给类
        self.read_configuration(file)

    def __getitem__(self, item):
        if not self.contain(item):
            print('Parameter ' + item + ' is not found in the configuration file!')
            exit(-1)
        return self.config[item]

    def contain(self, key):
        return key in self.config

    def read_configuration(self, file):
        if not os.path.exists(file):
            print('Config file is not found!')
            raise IOError
        with open(file, 'r') as f:
            try:
                self.config = yaml.safe_load(f)
                # 记录配置字典内容，确保 logger 可用
                if self.logger:
                    self.logger.info(f"Configuration loaded: {self.config}")
            except yaml.YAMLError as exc:
                print(f"Error in configuration file: {exc}")
                raise IOError
