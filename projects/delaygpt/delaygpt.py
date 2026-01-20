"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F

import numpy as np
import bisect

from mingpt.model import GPT
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
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class DelayDataset(Dataset):
    """
    Emits batches of floats
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
        
        # treating inputs as discrete
        chars = np.sort(np.unique(np.concatenate(data))) # data is a list of np.arrays
        data_size, vocab_size = len(data), len(chars)
        print('data has %d numbers, %d unique.' % (data_size, vocab_size)) # wrong data_size, gives number of paths
        self.vocab_size = vocab_size
        
        self.window_counts = [len(traj) - self.block_size for traj in data]
        self.offsets = np.cumsum([0] + self.window_counts)

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) numbers from the data
        traj_id = bisect.bisect_right(self.offsets, idx) - 1
        local_idx = idx - self.offsets[traj_id]
        chunk = self.data[traj_id][local_idx : local_idx + self.block_size + 1]
        # return as tensors
        x = torch.tensor(chunk[:-1]*10000, dtype=torch.long)
        y = torch.tensor(chunk[1:]*10000, dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    set_seed(config.system.seed)

    # construct the training dataset
    train_n = int(1e5)
    train_params = [
        {"r": 2.26, "delay": 1, "x_init": [0.1, 0.1]},
        {"r": 2.26, "delay": 1, "x_init": [0.1, 0.15]}
    ]
    test_n = int(250)
    test_params = [
        {"r": 2.26, "delay": 1, "x_init": [0.1, 0.2]},
        {"r": 2.26, "delay": 1, "x_init": [0.1, 0.175]}
    ]

    train_xs = delayed_logistic_mult(train_params, train_n)
    test_xs = delayed_logistic_mult(test_params, test_n)
    train_dataset = DelayDataset(config.data, [row for row in train_xs])
    test_dataset = DelayDataset(config.data, [row for row in test_xs])

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    print(config)
    setup_logging(config)
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # helper function for the evaluation of a model
    def eval_split(trainer, split, max_batches=None):
        dataset = {'train': train_dataset, 'test': test_dataset}[split]
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)

        losses = []
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            _, loss = model(x, y)
            losses.append(loss.item())
            if max_batches is not None and b + 1 >= max_batches:
                break

        avg_loss = sum(losses) / len(losses)
        print(f"{split} final score: avg loss = {avg_loss:.4f}")
        return avg_loss

    # iteration callback
    top_score = 0
    def batch_end_callback(trainer):
        global top_score

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                train_score = eval_split(trainer, 'train', max_batches=5)
                test_score  = eval_split(trainer, 'test',  max_batches=5)
            score = train_score + test_score
            # save the model if this is the best score we've seen so far
            if score > top_score:
                top_score = score
                print(f"saving model with new top score of {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
