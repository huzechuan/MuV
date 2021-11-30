import time
import sys
import os
from pathlib import Path
import copy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from configparser import ConfigParser
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torchtext.data import (
    Iterator
)
from torchtext.vocab import Vocab
from collections import Counter, OrderedDict

def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum

    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M

    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)

class configparser(ConfigParser):
    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=None)

    def optionxform(self, optionstr):
        return optionstr


def log_info(info, dynamic=False):
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if dynamic:
        print(f'\r{now_time} '
              f'{info} ', end='')
    else:
        print(f'{now_time} {info} ')
    sys.stdout.flush()


def revlut(lut):
    return {v: k for k, v in lut.items()}


def init_embedding(embedding):
    bias = np.sqrt(3.0 / embedding.size(0))
    nn.init.uniform_(embedding, -bias, bias)
    return embedding


def init_embeddings(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)

def init_tensor(tens, std=0.1545):
    """Initialize linear transformation
        """
    bias = std
    # bias = 0.1545# np.sqrt(6.0 / (tens.size(0) + tens.size(1)))
    nn.init.normal_(tens, 0, bias)

def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def preprocess(x):
    return None


def bert_pad(minibatch, include_length=True):
    padded = pad_sequence(minibatch)
    max_len = padded.size(0)
    lengths = []
    mask = []

    if include_length:
        for x in minibatch:
            length = x.size(0)
            lengths.append(length)
            mask.append([1] * length + [0] * (max_len - length))

        mask = torch.BoolTensor(mask).transpose(0, 1)
        return padded, lengths, mask

    return padded


def batch_index_select(input, dim, index):
    # for ii in range(1, len(index.shape)):
    #     if ii != dim:
    #         index = index.unsqueeze(ii)
    # expanse = list(input.shape)
    # expanse[0] = -1
    # expanse[dim] = -1
    # index = index.expand(expanse)
    index_sizes = list(index.size())
    index_sizes.append(input.size(dim - 1))
    index = index.view(-1)

    return torch.index_select(input, dim, index).view(index_sizes)


def select_labeled_data(dataset, labeled_set, select_idx):
    labeled_set.examples = []
    for idx in select_idx:
        ex = dataset.examples[idx]
        labeled_set.examples.append(ex)
    for ex in labeled_set.examples:
        dataset.examples.remove(ex)
    return dataset, labeled_set

def randomly_select_unlabeled_data(dataset, percent=1.0):
    cand = len(dataset)
    if percent > 1.0:
        num_unlabeled = int(percent)
    else:
        num_unlabeled = int(percent * cand)
    unlabeled_data_index = np.random.choice(cand, num_unlabeled, replace=False).tolist()
    examples = []
    for id in unlabeled_data_index:
        ex = dataset.examples[id]
        examples.append(ex)
    dataset.examples = examples

    return dataset

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def construct_pseudo_data(train_set, labeled_set, test_set, params, net, device, vote=False):
    pretrain_models = []
    pretrain_models_path = Path(params.model_path) / params.dataset
    models = os.listdir(pretrain_models_path)
    train, _ = Iterator.splits((train_set, None), batch_sizes=(50, 0),
                                       shuffle=False, sort_within_batch=False, sort_key=lambda x: len(x.BERT))
    dev, _ = Iterator.splits((labeled_set, None), batch_sizes=(50, 0),
                             shuffle=False, sort_within_batch=False, sort_key=lambda x: len(x.BERT))
    test, _ = Iterator.splits((test_set, None), batch_sizes=(50, 0),
                              shuffle=False, sort_within_batch=False, sort_key=lambda x: len(x.BERT))

    source_models = clone(net, len(params.source_language))
    for i, s in enumerate(params.source_language):
        for model in models:
            if model.startswith(s) and model.endswith('.pt'):
                model_path_ = Path(params.model_path) / params.dataset / model
                # log_info(model_path_)
                model_config = torch.load(model_path_, map_location=device)
                source_models[i].load_state_dict(model_config['model_state_dict'])
    del pretrain_models, model_config, net

    source_models.to(device)
    examples = [list() for _ in range(3)]
    itos = test_set.fields['label'].vocab.itos
    for dataset, data_iter, k in zip((train_set, labeled_set, test_set), (train, dev, test), (0, 1, 2)):
    # for dataset, data_iter, k in zip((labeled_set, test_set), (dev, test), (1, 2)):
        begin = 0
        for index, batch in enumerate(data_iter):
            end = len(batch.BERT[0]) + begin
            label_set = []
            for model in source_models:
                with torch.no_grad():
                    model.eval()
                    sentences, lengths, mask = batch.BERT
                    labels, _ = batch.label

                    scores = model(sentences, lengths)
                    decoded = model.decode(scores)
                label_set.append(decoded.tolist())

            if not vote:
                tmp_examples = [copy.deepcopy(dataset.examples[begin:end]) for _ in range(len(params.source_language))]
                ii, jj = range(3), range(50)
                for tmp_example, pseudo_label, i in zip(tmp_examples, label_set, ii):
                    for example, pseudo_l, lens, j in zip(tmp_example, pseudo_label, lengths.tolist(), jj):
                        label_s = []
                        for label_id in pseudo_l[:lens]:
                            label_s.append(itos[label_id])
                        example.pseudo_label = label_s
                        # if example.label != label_s:
                        #     print("No equal")
                    examples[k].extend(tmp_example)
            else:
                from collections import Counter
                from random import random
                tmp_examples = copy.deepcopy(dataset.examples[begin:end])
                label_set = np.array(label_set)
                label_set = label_set.transpose((1, 2, 0)).tolist()
                for example, pseudo_labels, lens in zip(tmp_examples, label_set, lengths.tolist()):
                    vote_labels = []
                    for i in range(lens):
                        tmp = Counter(pseudo_labels[i])
                        top_one = tmp.most_common(1)
                        if top_one[0][1] == 1: # randomly select
                            tmp_label = np.random.choice(pseudo_labels[i])
                        else:
                            tmp_label = top_one[0][0]
                        vote_labels.append(itos[tmp_label])
                    example.pseudo_label = vote_labels
                examples[k].extend(tmp_examples)

            begin = end

        # examples[i] = tmp
    train_set.examples = examples[0]
    labeled_set.examples = examples[1]
    test_set.examples = examples[2]
    return train_set, labeled_set, test_set

def build_vocab():
    vocab = Vocab(Counter(), )
    vocab.itos = []
    vocab.stoi.clear()
    vocab.stoi.update({str(i): i for i in range(-1, 201)})
    return vocab
