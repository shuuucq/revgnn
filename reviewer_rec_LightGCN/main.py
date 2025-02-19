import world
import utils
from world import cprint
import torch
from tensorboardX import SummaryWriter
import time
import procedure
from os.path import join
import register
from register import dataset

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

world.model_name = 'lgn'
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = r'./checkpoints/lgn-reviewer_rec-3-64.pth.tar'
print(f"load and save to {weight_file}")

# Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
Neg_k = 5

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

tag = []
note = ''
tag.append('GF-CF-' + world.model_name)
note += 'GF-CF-' + world.model_name
tag.append(world.dataset)
note += world.dataset
note += "-" + str(world.seed)
t_start = time.time()

if __name__ == '__main__':

    try:
        if world.simple_model != 'none':
            epoch = 0
            cprint("[TEST]")
            results = procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            t_epoch = time.time() - t_start
            wandb_log = {}
            wandb_log["Time"] = t_epoch
            wandb_log["Epoch"] = epoch
            for k in range(len(world.topks)):
                wandb_log["recall@"+str(world.topks[k])] = results['recall'][k]
                wandb_log["precision@"+str(world.topks[k])] = results['precision'][k]
                wandb_log["ndcg@"+str(world.topks[k])] = results['ndcg'][k]
                wandb_log["hr@"+str(world.topks[k])] = results['hr'][k]


        else:
            print('NOT SIMPLE MODEL')
            for epoch in range(world.TRAIN_epochs):
                t_epoch_start = time.time()
                output_information = procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
                print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
                if epoch % 10 == 0:
                    cprint("[TEST]")
                    results = procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                    t_epoch = time.time() - t_epoch_start
                    wandb_log = {}
                    wandb_log["Time"] = t_epoch
                    wandb_log["Epoch"] = epoch
                    for k in range(len(world.topks)):
                        wandb_log["recall@"+str(world.topks[k])] = results['recall'][k]
                        wandb_log["precision@"+str(world.topks[k])] = results['precision'][k]
                        wandb_log["ndcg@"+str(world.topks[k])] = results['ndcg'][k]
                        wandb_log["hr@"+str(world.topks[k])] = results['hr'][k]
                    # wandb.log(wandb_log)
                torch.save(Recmodel.state_dict(), weight_file)
    finally:
        if world.tensorboard:
            w.close()
