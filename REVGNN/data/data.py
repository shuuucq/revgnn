class Data(object):
    def __init__(self, conf, training,val, test):
        self.config = conf
        self.training_data = training
        self.val_data = val
        self.test_data = test #can also be validation set if the input is for validation







