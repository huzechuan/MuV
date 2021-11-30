import argparse
from torchtext.data import (
    Field
)
from data_preprocess.preprocess import (
    TransfomerField
)
from data_preprocess import preprocess
from data_preprocess.params import (Direct)
from data_preprocess.utils import (
    configparser,
    log_info
)
from typing import Dict, List
from pathlib import Path
import torch
import pickle
import os
import numpy as np

def select_data(train_set, percent):
    log_info('Selecting labeled data...')
    cand = len(train_set)
    if params.percent_of_labeled_data > 1.0:
        num_labeled_sents = int(percent)
    else:
        num_labeled_sents = int(percent * cand)
    labeled_data_index = np.random.choice(cand, num_labeled_sents, replace=False).tolist()
    labeled_data_index.sort()
    return labeled_data_index

def build_config_file(params: Direct, device: torch.device):
    BERT: Field = TransfomerField(device=device)
    label: Field = Field(pad_token='<pad>', include_lengths=True, batch_first=True)

    if params.task in ['parsing', 'pos']:
        fields: List = [(None, None), ('BERT', BERT), (None, None), (('gold', 'label'), (label, label))] + [(None, None)] * 6
    else:
        fields: List = [('BERT', BERT), (('gold', 'label'), (label, label))]

    separator = '\t' if params.task in ['pos', 'POS', 'parsing'] else ' '
    spliter = getattr(preprocess, params.dataset)
    train_set, val_set, test_set = spliter.splits(fields=fields, path=Path(params.dataset) / params.language,
                                                  separator=separator)

    langs = [params.name]
    label.build_vocab(train_set.label)
    vocab = label.vocab
    labeled_data = select_data(train_set, params.percent_of_labeled_data)


    file_path = Path('./config/config.pkl')
    if not (os.path.exists(file_path)):
        log_info('No config file exists.')
        dic = {params.dataset: {'vocab': vocab, params.language: labeled_data}}
    else:
        log_info('There already exist a config file.')
        f = open(file_path, 'rb')
        dic = pickle.load(f)
        dic[params.dataset]['vocab'] = vocab
        dic[params.dataset][params.language] = labeled_data
        f.close()

    # with open(Path('./config/config.pkl'), 'rb') as f:
    #     data = pickle.load(f)
    #     print(data)
    #     print(data[params.dataset]['vocab'].itos)
    # exit(0)
    log_info(f'New label set {langs} {len(vocab)}, {vocab.itos}')
    log_info(f'select {labeled_data}')

    for lang, lang_name in zip(params.target_languages, params.target_names):
        langs.append(lang_name)
        train_set, val_set, test_set = spliter.splits(fields=fields, path=Path(params.dataset) / lang,
                                                      separator=separator)
        label.build_vocab(train_set.label)
        vocab.extend(label.vocab)
        labeled_data = select_data(train_set, params.percent_of_labeled_data)
        dic[params.dataset][lang] = labeled_data
        log_info(f'New label set {langs} {len(vocab)}, {vocab.itos}')
        log_info(f'select {labeled_data}')

    with open(file_path, 'wb') as f:
        pickle.dump(dic, f)
        log_info(f'successfully build config: "{str(file_path)}"')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Learning with ncrf or triLinear-ncrf')
    parser.add_argument('--config', default='conll.config', help='Path of .config file.')
    parser.add_argument('--target_language', default='conll_03_german,conll_03_dutch,conll_03_spanish', help='target language name')
    parser.add_argument('--target_name', default='de,nl,es', help='target language name')
    parser.add_argument('--language', default='conll_03_english', help='source language corpus')
    parser.add_argument('--name', default='en', help='source language name')

    args = parser.parse_args()
    # config_name = args.config
    # cluster = args.cluster

    # config_file = Path('./config') / 'Direct' / Path(config_name)
    # log_info(config_file)
    # config = configparser()
    # config.read(config_file, encoding='utf-8')
    # 1. Read experiments' settings.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = Direct(args)
    data = build_config_file(params, device)
