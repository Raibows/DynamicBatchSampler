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


import random
import torch
import logging
import math
import torch.distributed as dist
from torch.utils.data import Sampler



class DynamicBatchSampler(Sampler):
    def __init__(self, num_replicas, rank, length_dict, num_buckets=128, min_len=0, max_len=1024,
                 max_batch_tokens=None, max_batch_size=None, shuffle=True, seed=0, drop_last=False,) -> None:
        """
        A dynamic batch sampler supports DDP for robust training
        :param num_replicas: int
            the world size (i.e. the num of gpus), set it to 1 if you are using single gpu
        :param rank: int
            the rank of the gpu (see PyTorch DDP docs for details), set it to 0 if you are using single gpu
        :param length_dict: dict or list
            to get the token num of a sample, {idx: the token num of that sample}
        :param num_buckets: int
            the smaller the num_buckets, the richer the permutation in one batch
            it is not ordering and there is no difference with the PyTorch default sampler if num_buckets set to 1
            it is going to be deterministic hence lost the advantage in robust training if num_buckets set to len(dataset)
            the best param is related with your dataset length distribution, set it carefully
        :param min_len: int
            skip the sample whose length < min_len
        :param max_len: int
            skip the sample whose length > max_len
        :param max_batch_tokens: int or None
            max_batch_tokens and max_batch_size determine the usage of gpu memory and the 'real batch size' together
        :param max_batch_size: int or None
            max_batch_size and max_batch_tokens determine the usage of gpu memory and the 'real batch size' together
        :param shuffle: bool
        :param seed: int
        :param drop_last: bool
        """
        super(DynamicBatchSampler, self).__init__(None)
        if dist.is_available() and not num_replicas > rank >= 0:
            raise RuntimeError(f"rank should be in the [0, {num_replicas - 1}]")
        if not dist.is_available():
            assert num_replicas == 1 and rank == 0, "rank and num_replicas have to be set to 1 if you are not in multi gpu(DDP) mode"
        assert max_batch_tokens is not None or max_batch_size is not None, "you have to specify one of [max_batch_tokens, max_batch_size] to decide the 'real batch size'"
        self.max_batch_tokens = max_batch_tokens if max_batch_tokens is not None else float('Inf')
        self.max_batch_size = max_batch_size if max_batch_size is not None else float('Inf')
        assert self.max_batch_size >= 1
        assert max_len >= self.max_batch_tokens >= min_len
        random.seed(seed)
        self.num_replicas = num_replicas
        self.rank = rank
        self.length_dict = length_dict
        self.num_buckets = num_buckets
        self.min_len = min_len
        self.max_len = max_len
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.__epoch = 0
        self.__logger = logging.getLogger('sampler')
        self.__per_gpu_batch_num = 0
        self.__batches = []

    def __len__(self):
        return self.__per_gpu_batch_num

    def __iter__(self):
        for batch in self.__batches[self.rank:len(self.__batches):self.num_replicas]:
            yield batch

    def set_epoch(self, epoch: int):
        self.__epoch = epoch
        self.__batches = self.__prepare_batches()

    def __is_full(self, tokens_in_all, batch):
        if len(batch) == self.max_batch_size:
            return True
        if tokens_in_all > self.max_batch_tokens:
            return True
        return False

    def __prepare_batches(self):
        if self.rank == 0:
            self.__logger.info(f"starting prepare batches of epoch {self.__epoch} shuffle {self.shuffle}")
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.__epoch)
            indices = torch.randperm(len(self.length_dict), generator=g).tolist()
        else:
            # try not to re-prepare batches
            if self.__batches is not None: return self.__batches
            indices = list(range(len(self.length_dict)))
        batches = []
        buckets = [[] for _ in range(self.num_buckets)]
        buckets_max_len = [0 for _ in range(self.num_buckets)]
        for idx in indices:
            idx_len = self.length_dict[idx]
            if not self.max_len >= idx_len >= self.min_len:
                if self.rank == 0:
                    self.__logger.warning(f"ignored one sample with index {idx}, length {idx_len} not in the interval [{self.min_len}, {self.max_len}]")
                continue
            idx_bkt = math.floor((idx_len - self.min_len) / (self.max_len - self.min_len + 1) * self.num_buckets)
            buckets_max_len[idx_bkt] = max(buckets_max_len[idx_bkt], idx_len)
            # +1 is make sure it will judge correctly whether it is full if you add this sample
            tokens_in_all = (len(buckets[idx_bkt]) + 1) * buckets_max_len[idx_bkt]
            if self.__is_full(tokens_in_all, buckets[idx_bkt]):
                batches.append(buckets[idx_bkt])
                buckets[idx_bkt] = []
                buckets_max_len[idx_bkt] = 0
            # add the sample to the bucket that contains samples all have similar length
            buckets[idx_bkt].append(idx)

        # process the leftover samples, try to group them to a batch
        # leftover samples are ascending by length
        leftover_batch = []
        leftover_max_len = 0
        leftover_indices = [idx for bkt in buckets for idx in bkt]
        for idx in leftover_indices:
            idx_len = self.length_dict[idx]
            leftover_max_len = max(leftover_max_len, idx_len)
            tokens_in_all = (len(leftover_batch) + 1) * leftover_max_len
            if self.__is_full(tokens_in_all, leftover_batch):
                batches.append(leftover_batch)
                leftover_batch = []
                leftover_max_len = 0
            leftover_batch.append(idx)

        # whether to drop last
        if len(leftover_batch) > 0:
            if self.drop_last:
                if self.rank == 0:
                    self.__logger.warning(f"dropped the leftover batch size {len(leftover_batch)}")
            else:
                batches.append(leftover_batch)

        self.__per_gpu_batch_num = math.ceil(len(batches) / self.num_replicas)
        total_batch_num = self.__per_gpu_batch_num * self.num_replicas
        dummy_batch_num = total_batch_num - len(batches)
        if dummy_batch_num <= len(batches):
            dummy_batches = random.sample(batches, k=dummy_batch_num)
        else:
            if self.rank == 0:
                self.__logger.warning(f"repeated batches will exist because the dummy_batch_num is larger than len(batches)")
            dummy_batches = [random.choice(batches) for _ in range(dummy_batch_num)]
        batches += dummy_batches

        # rich the batch permutation and is the key reason to improve the model's robustness
        if self.shuffle:
            random.shuffle(batches)
        return batches








