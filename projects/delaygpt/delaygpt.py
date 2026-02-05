"""
Trains a difference equation predictive model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F

import numpy as np
import bisect
import pandas as pd
import time

from mingpt.cts_model import ContinuousGPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from helpers.delaydiff import delayed_logistic_mult

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/delaygpt'

    # data
    C.data = DelayDataset.get_default_config()

    # model
    C.model = ContinuousGPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class DelayDataset(Dataset):
    """
    Emits batches of floats
    data is List[np.array of shape (n,) or shape (n, input_dim)], of length num_traj
    get_item always returns x, y that are Float tensors with shape (n, input_dim), only time Float comes up
    Note must be Float and not double because model weights are Floats not doubles
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.block_size = self.config.block_size
        self.input_dim = data[0].shape[1] if len(data[0].shape) > 1 else 1 # get input_dim if array is matrix, else 1
        
        # List[num of windows per traj], window_size = block_size + 1
        self.window_counts = [len(traj) - self.block_size for traj in data]
        # List[index of first window of each trajectory] + [total # of windows] 
        # (shift index 0 by the number of windows in the traj) or equivalently
        # [0] + List[number of windows after combining first (n-1) trajectories]
        self.offsets = np.cumsum([0] + self.window_counts) # + is list concatenation
        
        # treating inputs as discrete
        # chars = np.sort(np.unique(np.concatenate(data))) # data is a list of np.arrays
        # data_size, vocab_size = len(data), len(chars)
        # print('data has %d numbers, %d unique.' % (data_size, vocab_size)) # wrong data_size, gives number of paths
        # self.vocab_size = vocab_size

    # def get_vocab_size(self):
    #     return self.vocab_size
    
    def get_input_dim(self):
        return self.input_dim
    
    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, idx):
        # grab a window of (block_size + 1) numbers from the data, return as tensor
        traj_id = bisect.bisect_right(self.offsets, idx) - 1
        # bisect treats elements of len n list as walls between n+1 buckets
        # right and left says to put the walls in the left or right bucket they are adjacent to
        # everything after index 0 before next one gets labelled as bucket 1, subtract 1 to 0-index
        local_idx = idx - self.offsets[traj_id]

        if self.input_dim == 1:
            window = self.data[traj_id][local_idx : local_idx + self.block_size + 1] # array of shape (block size + 1,)
            x = torch.tensor(window[:-1], dtype=torch.float).unsqueeze(1)
            y = torch.tensor(window[1:], dtype=torch.float).unsqueeze(1)
        else:
            window = self.data[traj_id][local_idx : local_idx + self.block_size + 1, :] # array of shape (block size + 1,)
            x = torch.tensor(window[:-1, :], dtype=torch.float)
            y = torch.tensor(window[1:, :], dtype=torch.float)
        return x, y

# -----------------------------------------------------------------------------

# Config workflow
    # config = CN containing a CN for system, data, model, trainer
    # 1) fn sets it to defaults, other than changes for work_dir, model_type, learning_rate
    # 2) overwrite any args received from the command line
    # use data to make dataset, which infers / sets 2 params about data (input_dim, block_size)
    # 3) set params of model based on these 2 params about data
    # make model
    # make trainer
    # done changing config, log and print config
    # ***unexpected behavior now: log and print config before making model so 
    # model doesn't fill in params from model_type when logging config, will cause errors in logic
    # when making another model from the config, since only allowed to have model type or params filled in (XOR)

if __name__ == '__main__':
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    
    set_seed(config.system.seed) # util function sets seed for libraries

    ########## Construct the training dataset ##########
    train_n = int(1e6)
    test_n = int(1e4)
    train_params = [ # train on 2 paths of one equation
        {"r": 2.26, "delay": 1, "x_init": [0.1, 0.1]},
        {"r": 2.26, "delay": 1, "x_init": [0.1, 0.15]}
    ]
    test_params = [ # test on 2 paths of same equation
        {"r": 2.26, "delay": 1, "x_init": [0.1, 0.2]},
        {"r": 2.26, "delay": 1, "x_init": [0.1, 0.175]}
    ]
    train_trajs = delayed_logistic_mult(train_params, train_n) # fn: List[Dict{delay:, r:, x_init:}] -> List[np.array of shape (n,)]
    test_trajs = delayed_logistic_mult(test_params, test_n)
    
    train_dataset = DelayDataset(config.data, train_trajs)
    test_dataset = DelayDataset(config.data, test_trajs)

    ########## Construct the model #####################
    config.model.input_dim = train_dataset.get_input_dim()
    config.model.block_size = train_dataset.get_block_size()
    print(config)
    setup_logging(config)
    model = ContinuousGPT(config.model)

    ########## Construct the trainer object ############
    trainer = Trainer(config.trainer, model, train_dataset)

    # helper function for iteration callback
    # dataset -> list containing a loss for each batch of data -> average of these losses
    # list length is max_batches or entire dataset
    def eval_split(device, dataset, max_batches=None):
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        losses = []
        for b, (x, y) in enumerate(loader):
            loss = model(x.to(device), y.to(device)).loss.item()
            # need to move data (usually on cpu) to device where the model weights are, so can apply model
            # loss is a 0-dim tensor attached to a computational graph, just get the number
            losses.append(loss)
            if max_batches is not None and b + 1 >= max_batches:
                break
        return sum(losses) / len(losses)

    # iteration callback
    top_score = float("inf") # define a global variable
    training_logs = []
    def batch_end_callback(trainer):
        global top_score # tells whatever is calling this function to update the global

        if trainer.iter_num % 10 == 0:
            # print a train score for a single batch
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
            entry = {
                'iter': trainer.iter_num,
                'loss': trainer.loss.item(),
                'top_score': top_score,
                'iter_time': trainer.iter_dt, # time for THIS iteration
                'timestamp': time.time()      # wall clock time
            }
            training_logs.append(entry)

        if trainer.iter_num % 500 == 0:
            # print a score based on the average train and test score for several batches
            model.eval()
            with torch.no_grad():
                train_score = eval_split(trainer.device, train_dataset, max_batches=5)
                test_score  = eval_split(trainer.device, test_dataset,  max_batches=5)
            print(f"Train final score: avg loss = {train_score:.4f}")
            print(f"Test final score: avg loss = {test_score:.4f}")
            score = train_score + test_score
            # save the model if this is the best score we've seen so far
            if score < top_score:
                top_score = score
                print(f"saving model with new top score of {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()
            
            # save the training log
            df = pd.DataFrame(training_logs)
            df.to_csv("training_log.csv", index=False)

    trainer.set_callback('on_batch_end', batch_end_callback)

    ########## Run the optimization ####################
    trainer.run() 
