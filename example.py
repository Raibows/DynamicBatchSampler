# Project: https://github.com/Raibows/DynamicBatchSampler
# Author: Raibows@GitHub https://github.com/Raibows
# Licensed under the GNU GENERAL PUBLIC LICENSE, Version 3
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import random
import copy
import time
import torch
import logging
from torch import nn
from tqdm import tqdm
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
from argparse import ArgumentParser
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as ddp

from DBSampler import DynamicBatchSampler

logging.basicConfig(level=logging.INFO, datefmt="%m/%d/%Y %H:%M:%S", format='%(asctime)s - %(levelname)s - %(name)s\n%(message)s')
PAD_TOKEN_ID = 0
VOCAB_SIZE = 128
class MyDataset(Dataset):
    # 2 classes classification
    def __init__(self, vocab_size, sample_num, max_len):
        self.x = [[random.randint(1, vocab_size - 1) for _ in range(random.randint(1, max_len))] for _ in range(sample_num)]
        self.y = [random.randint(0, 1) for _ in range(sample_num)]
        # the len_dict contains the number of tokens of each sample
        self.len_dict = [len(k) for k in self.x]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.len_dict[item]

class MyModel(nn.Module):
    def __init__(self, vocab_size, classes, word_dim):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_dim)
        self.mlp = nn.Linear(word_dim, classes)

    def forward(self, x):
        embeds = self.embedding.forward(x)
        avg_embeds = torch.sum(embeds, dim=1) / embeds.shape[1]
        logits = self.mlp.forward(torch.tanh(avg_embeds))
        
        return logits


def list_copy(*args):
    return [list(copy.deepcopy(a)) for a in args]

def padding_collator(batch):
    # padding your batch
    x, y, x_len = zip(*batch)
    x, y, x_len = list_copy(x, y, x_len)
    max_len = max(x_len)
    for idx in range(len(x)):
        pad_len = max_len - len(x[idx])
        if pad_len > 0:
            x[idx] += [PAD_TOKEN_ID] * pad_len
    x, y = torch.tensor(x, dtype=torch.int32), torch.tensor(y, dtype=torch.long)
    return x, y

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def runner(rank, world_size, port):
    setup(rank, world_size, port)
    dataset = MyDataset(VOCAB_SIZE, 10000, 64)
    # create your loader here
    batch_sampler = DynamicBatchSampler(world_size, rank, dataset.len_dict, 64, max_batch_size=32, max_batch_tokens=128, shuffle=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=padding_collator)
    # create your loader done
    model = MyModel(VOCAB_SIZE, 2, 64)
    model.to(rank)
    ddp_model = ddp(model, device_ids=[rank])
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion.to(rank)

    for epoch in range(2):
        # VERY IMPORTANT BELOW
        batch_sampler.set_epoch(epoch)
        # VERY IMPORTANT ABOVE
        with tqdm(total=len(loader), disable=rank != 0, desc=f'epoch {epoch}') as pbar:
            for i, (x, y) in enumerate(loader):
                optimizer.zero_grad()
                x, y = x.to(rank), y.to(rank)
                logits = model.forward(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                pbar.set_postfix_str(f"loss {loss.item():.4f} bsz {x.shape[0]}")
                pbar.update(1)
                # for intuition
                time.sleep(0.3)

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    gpu_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(runner, args=(gpu_num, 12345), nprocs=gpu_num, join=True)