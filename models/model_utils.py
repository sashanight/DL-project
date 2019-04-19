import torch
import torch.nn as nn
import numpy as np

def mean_pool(x, lengths):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_()
    for i in range(x.size(1)):
        out[i] = torch.mean(x[:lengths[i],i,:], 0)
    return out

def max_pool(x, lengths):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_()
    for i in range(x.size(1)):
        out[i,:] = torch.max(x[:lengths[i],i,:], 0)[0]
    return out

def min_pool(x, lengths):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_()
    for i in range(x.size(1)):
        out[i] = torch.min(x[:lengths[i],i,:], 0)[0]
    return out


def pool(out, lengths, params):
    if params.pooling == "mean":
        out = mean_pool(out, lengths)
    elif params.pooling == "max":
        out = max_pool(out, lengths)
    elif params.pooling == "min":
        out = min_pool(out, lengths)
    else:
        raise ValueError("No valid pooling operation specified!")
    return out


def param_init(model, opts):
    if opts.init == "normal":
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.normal_(p)
    elif opts.init == "uniform":
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.uniform_(p, a=-0.1, b=0.1)
    elif opts.init == "kaiming":
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)
    elif opts.init == "xavier":
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

                
def embed(batch, params, se_params, to_reverse=False):
    input_seq = torch.LongTensor(params.max_seq_len, len(batch)).zero_()

    cur_max_seq_len = 0
    for i, l in enumerate(batch):
        j = 0
        if to_reverse:
            l.reverse()
        for k, w in enumerate(l):
            if k == params.max_seq_len:
                break
            input_seq[j][i] = se_params.word2id[w]
            j += 1
        if j > cur_max_seq_len:
            cur_max_seq_len = j

    input_seq = input_seq[:cur_max_seq_len]
    out = se_params.lut(input_seq)
    if params.gpu:
        out = out.cuda()

    lengths = [len(i) if len(i) < params.max_seq_len else params.max_seq_len for i in batch]
    lengths = torch.from_numpy(np.array(lengths))

    return lengths, out