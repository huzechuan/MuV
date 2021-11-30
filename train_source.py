import argparse
from torchtext.data import (
    Field, BucketIterator
)
from data_preprocess.preprocess import (
    Data, TransfomerField
)
from data_preprocess import preprocess
from data_preprocess.params import Direct
from data_preprocess.utils import (
    configparser,
    log_info, select_labeled_data
)
from model.top_layer import Softmax
from model.evaluator import eval_softmax
from typing import List
from pathlib import Path
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import time
import os
import pickle

def main(params: Direct, device: torch.device):
    BERT: Field = TransfomerField(device=device)
    label: Field = Field(pad_token='<pad>', include_lengths=True, batch_first=True)

    if params.task in ['parsing', 'pos']:
        fields: List = [(None, None), ('BERT', BERT), (None, None), (('gold', 'label'), (label, label))] + [(None, None)] * 6
    else:
        fields: List = [('BERT', BERT), (('gold', 'label'), (label, label))]

    separator: str = '\t' if params.task in ['pos', 'POS', 'parsing'] else ' '
    spliter = getattr(preprocess, params.dataset)
    train_set, val_set, test_set = spliter.splits(fields=fields, path=Path(params.dataset) / params.language,
                                                  separator=separator)
    train_set_target, dev_set_target, test_set_target = spliter.splits(fields=fields,
                                                              path=Path(params.dataset) / params.target_language,
                                                              separator=separator)
    config_path = Path('./config/config.pkl')
    assert os.path.exists(config_path), 'miss config file.'
    with open(config_path, 'rb') as f:
        dic = pickle.load(f)
        label.vocab = dic[params.dataset]['vocab']
        labeled_data_index = dic[params.dataset][params.target_language]

    labeled_data_index = sorted(labeled_data_index)
    setattr(params, 'labeled_data_index', labeled_data_index)
    train_set_tmp, labeled_set = select_labeled_data(train_set_target, dev_set_target, labeled_data_index)
    train_set.examples.extend(labeled_set.examples)

    train_iter, val_iter, test_iter, target_test_iter = BucketIterator.splits(
                (train_set, val_set, test_set, test_set_target), batch_sizes=(params.batch_size, 50, 50, 50), sort_key=lambda x: len(x.BERT))

    print('-' * 100)
    print('-' * 100)
    print(f'Target Corpus: {params.target_language}')
    print(f'Source Corpus: {params.language}')
    print(f'Corpus: "train {len(train_set)} val {len(val_set)} test {len(test_set)}"')
    print(f'Label set: {" ".join(label.vocab.itos)}')
    print(f'Labeled data: {" ".join(map(str, labeled_data_index))}')
    print('-' * 100)

    data = Data(train=train_iter, dev=val_iter, test=[target_test_iter, test_iter],
                label_dict=label.vocab, device=device)
    train(data, params)

def train(data, params):
    net = Softmax(model=params.BERT, num_labels=len(data.label_dict), dropout=0.1, device=data.device)
    # net.rand_init()
    net.cuda()

    # log_info(f'Model: "{net}"')
    print('-' * 100)
    print("Parameters:")
    print(f' - mini_batch_size: "{params.batch_size}"')
    print(f' - learning_rate: "{params.HP_BERT_lr}"')
    print(f' - L2: "{params.HP_L2}"')
    print(f' - max_epochs: "{params.max_epoch}"')
    print('-' * 100)
    total_number_of_batches = len(data.train)
    modulo = max(1, int(total_number_of_batches / 10))

    net_params = []
    if params.freeze:
        freezed_parameters = []
        # log_info(f'Freezing last three layers!')
        for name, value in dict(net.named_parameters()).items():
            if name.startswith('model.model.encoder.layer.0.') or name.startswith('model.model.encoder.layer.1.') \
                    or name.startswith('model.model.encoder.layer.2.') or name.startswith('model.model.embeddings'):
                net_params.append({'params': [value], 'lr': 0.0})
                freezed_parameters.append(name)
            else:
                net_params.append({'params': [value], 'lr': params.HP_BERT_lr})
        print('-' * 100)
        log_info(f' - Freeze parameters: {" | ".join(freezed_parameters)}')
        print('-' * 100)
    else:
        net_params = net.parameters()
    optimizer = AdamW(net_params, lr=params.HP_BERT_lr, betas=(0.9, 0.999), weight_decay=params.HP_L2)
    model_path = Path('./models') / params.dataset / params.method
    if not (os.path.exists(model_path) and os.path.isdir(model_path)):
        os.makedirs(model_path, exist_ok=True)

    save_name = f'{params.name}_{params.target_name}50.pt'

    eval_path = model_path / params.name
    if not (os.path.exists(eval_path) and os.path.isdir(eval_path)):
        os.makedirs(eval_path, exist_ok=True)
    evaluator = eval_softmax(params.metric, data.label_dict, eval_path)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_number_of_batches * 0.1,
                                                num_training_steps=total_number_of_batches * params.max_epoch)

    for epoch in range(params.max_epoch):
        net.train()
        epoch_loss = 0
        seen_batches = 0
        log_info('-' * 100)
        batch_time = 0

        for index, batch in enumerate(data.train):
            for group in optimizer.param_groups:
                learning_rate = group["lr"]
            start_time = time.time()
            net.zero_grad()
            sentences, lengths, mask = batch.BERT
            labels, l_lengths = batch.label
            scores, _, _ = net(sentences, lengths)

            loss = net.crit(scores, labels.cuda(device=device), mask)#/ params.batch_size
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss
            seen_batches += 1

            batch_time += time.time() - start_time
            if seen_batches % modulo == 0:
                log_info(
                    f"epoch {epoch + 1} - iter {seen_batches}/{total_number_of_batches} - loss "
                    f"{epoch_loss / seen_batches:.8f} learning_rate: {learning_rate:.8f} - samples/sec: {params.batch_size * modulo / batch_time:.2f}",
                    dynamic=False
                )
                batch_time = 0
            scheduler.step()

        epoch_loss = epoch_loss / (index + 1)
        # log_info(f'loss: {epoch_loss}.', dynamic=False)

        test_scores = []
        for test_iter in data.test:
            test_score, _, _, _ = evaluator.calc_score(net, test_iter, 'test')
            test_scores.append(str(round(test_score, 2)))
        test_score = f'{params.target_name}=' + ' '.join(test_scores)
        if True:
            log_info(f'Saving model to {str(model_path)}')
            if params.save_model:
                torch.save({
                    'label': data.label_dict,
                    f'labeled_data': params.labeled_data_index,
                    'task': params.task,
                    'name': params.name,
                    'target_name': params.target_name,
                    'metric': params.metric,
                    'model_state_dict': net.state_dict()
                }, model_path / save_name)

        log_info(f'loss: {epoch_loss} test_score {test_score}')
    # fout.writelines(f'loss={epoch_loss:.5f} test_score {test_score}\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Learning with ncrf or triLinear-ncrf')
    parser.add_argument('--config', default='conll.config', help='Path of .config file.')
    parser.add_argument('--target_language', default='conll_03_german,conll_03_dutch,conll_03_spanish', help='target language name')
    parser.add_argument('--target_name', default='de,nl,es', help='target language name')
    parser.add_argument('--language', default='conll_03_english', help='source language corpus')
    parser.add_argument('--name', default='en', help='source language name')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Read experiments' settings.
    params = Direct(args)
    target_langs = params.target_languages
    target_names = params.target_names
    for tl, tn in zip(target_langs, target_names):
        setattr(params, 'target_language', tl)
        setattr(params, 'target_name', tn)
        main(params, device)
