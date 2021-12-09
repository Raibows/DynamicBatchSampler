# Dynamic Batch Sampler

Yet another dynamic batch sampler for variable sequence data (e.g., most of the data in NLP) in PyTorch. Supports both **single gpu** and **multi-gpu** training (DDP, Distributed Data Parallel). 

<p align=center>
    <b><big>Rubust. Easy to use. Efficient for distributed training.</big></b>
</p>

Efficient for training because it will cluster the input data that have similar length to reduce the padding as much as possible. Besides, the batch size will dynamically change to utilize the gpu memory due to every batch is determined by the number of total tokens of a batch. Further to explain, your gpu memory usage is decided by the number of total tokens of a batch but not the batch size! So a better way to process variable sequence data like text sentence is to let the batch size change! See the parameters to have a thorough comprehension.

Welcome to issue or pull requests! Please star my project if it helps you.


## Requirements

```
python 3.5+
PyTorch 1.x
```

## Quick Start

1. clone this repo or just copy the file ``DBSampler.py`` to your root directory of your project.

2. define your padding collator to pad the batch

   ```python
   def padding_collator(batch):
       # padding to the max length of the batch
    	# batch =  [(x, y, x_len), ...]  
       x, y, x_len = zip(*batch)
       x, y, x_len = list_copy(x, y, x_len)
       max_len = max(x_len)
       for idx in range(len(x)):
           pad_len = max_len - len(x[idx])
           if pad_len > 0:
               x[idx] += [PAD_TOKEN_ID] * pad_len
       x, y = torch.tensor(x, dtype=torch.int32), torch.tensor(y, dtype=torch.long)
       return x, y
   ```

3. create your dataloader like this

   ```python
   from DBSampler import DynamicBatchSampler
   dataset = MyDataset()
   batch_sampler = DynamicBatchSampler(world_size, rank, dataset.len_dict, 64, max_batch_size=32, max_batch_tokens=128, shuffle=True)
   # you don't need to pass other params (e.g., batch_size, shuffle and sampler) 
   dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=padding_collator)
   ```

4. start your training

   ```python
   for epoch in range(10):
       # very important for the data shuffle
       batch_sampler.set_epoch(epoch)
       for x, y in loader:
           ...
   ```

A text classification demo could be found at ``example.py``, run it with ``python example.py --device 0,1,2,3`` and watch the batch size change. You can find more in [PyTorch DDP docs](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) if you are not familiar with multi-gpu training.

## Parameters

- **num_replicas**: int

  The world size (i.e. the num of gpus), set it to 1 if you are using single gpu.

- **rank**: int

  The rank of the gpu (see PyTorch DDP docs for details), set it to 0 if you are using single gpu.

- **length_dict**: dict or list, {idx: the length of that sample}

  To get the token num (length) of a sample.

- **num_buckets**: int

  The smaller the ``num_buckets``, the richer the permutation in one batch. It is not ordering and there is no difference with the PyTorch default sampler if ``num_buckets`` is set to 1. It is going to be deterministic hence lost the advantage in robust training if ``num_buckets`` is set to ``len(dataset)``. The best param is related with your dataset length distribution, set it carefully.

- **min_len**: int

  Skip the sample whose ``length < min_len``

- **max_len**: int

  Skip the sample whose ``length > max_len``

- **max_batch_tokens**: int

  ``max_batch_tokens`` and ``max_batch_size`` determine the usage of gpu memory and the real batch size together.

- **max_batch_size**: int

  ``max_batch_size`` and ``max_batch_tokens`` determine the usage of gpu memory and the real batch size together. In details, the gpu memory usage is up to your model, optimizer and the batch. The size of batch is up to the total number of tokens in this batch. It will pack to a batch when reaches the ``max_batch_tokens`` or ``max_batch_size``. Empirically, set ``max_batch_size`` as large as possible benefits for both training speed and model performance. Note that a dynamic change in real batch size is  a key point for robust training.

- **shuffle**: bool

  Whether to shuffle the data after every epoch. It is strongly suggested to set ``True`` for train dataset to use the power of ``DynamicBatchSampler``.

- **seed**: int

  Set the seed for reproducibility.

- **drop_last**: bool

  Whether to drop the last not packed batch. Note that some dummy batches will exist for syncing multi-gpu.

## Reference

[huggingface/transformers](https://huggingface.co/transformers/)

[facebook/fairseq](https://github.com/pytorch/fairseq)

## License

Under GPLv3.
