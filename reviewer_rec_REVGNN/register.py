import world
import dataloader
import model
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'aminer', 'citeulike', 'amazon', 'ml-1m', 'reviewer_rec']:
    dataset = dataloader.Loader_train(path='get_paper_embedding/dataset_4k')
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'lgn': model.LightGCN
}
