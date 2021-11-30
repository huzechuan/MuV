import argparse
import numpy as np
from torchtext.data import (
    Field, NestedField, BucketIterator, Iterator
)
from data_preprocess.preprocess import (
    Data, TransfomerField
)
from data_preprocess import preprocess
from data_preprocess.params import Params, MView
from data_preprocess.utils import (
    configparser, log_info, select_labeled_data, randomly_select_unlabeled_data
)
from model.top_layer import StackedLayer, Softmax, Assembled_MView
from model.evaluator import eval_softmax
from typing import Dict, List
from pathlib import Path
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import time
import os
import pickle

def main(params: MView, device):
    BERT: Field = TransfomerField(device=device)
    label: Field = Field(pad_token='<pad>', include_lengths=True, batch_first=True)

    if params.task in ['parsing', 'pos', 'POS']:
        fields: List = [(None, None), ('BERT', BERT), (None, None), (('gold', 'label'), (label, label))] + [(None, None)] * 6
    else:
        fields: List = [('BERT', BERT), (('gold', 'label'), (label, label))]

    separator = '\t' if params.task in ['pos', 'POS'] else ' '
    spliter = getattr(preprocess, params.dataset)
    train_set, labeled_set, test_set = spliter.splits(fields=fields, path=Path(params.dataset) / params.language,
                                                  separator=separator)

    config_path = Path('./config/config.pkl')
    assert os.path.exists(config_path), 'miss config file.'
    with open(config_path, 'rb') as f:
        dic = pickle.load(f)
        label.vocab = dic[params.dataset]['vocab']
        labeled_data_index = dic[params.dataset][params.language]

    # labeled_set = val_set
    labeled_data_index = sorted(labeled_data_index)
    train_set, labeled_set = select_labeled_data(train_set, labeled_set, labeled_data_index)

    if params.percent_of_unlabeled_data != 1.0:
        train_set = randomly_select_unlabeled_data(train_set, percent=params.percent_of_unlabeled_data)

    train_iter, labeled_iter, test_iter = BucketIterator.splits(
        (train_set, labeled_set, test_set), batch_sizes=(params.batch_size, params.batch_size, 50),
        sort_key=lambda x: len(x.BERT)
    )
    # labeled_iter, _ = BucketIterator.splits((labeled_set, None), batch_sizes=(params.batch_size, 0), sort_key=lambda x: len(x.BERT))

    print('-' * 100)
    print('-' * 100)
    print(f'Target Corpus: {params.language}')
    print(f'Source Models: {" ".join(params.source_language)}')
    print(f'Corpus: "train {len(train_set)}+{len(labeled_set)} test {len(test_set)}"')
    print(f'Label set: {" ".join(label.vocab.itos)}')
    print(f'Labeled data: {" ".join(map(str, labeled_data_index))}')
    print('-' * 100)

    data = Data(
        train=train_iter, labeled_data=labeled_iter, test=test_iter, label_dict=label.vocab,
        sources=params.source_language, labeled_data_index=labeled_data_index, device=device
        )

    trainer(data, params)

def step_one(net, optimizer, scheduler, modulo, batch_l, batch_u, start_time, total_number_of_batches,
              batch, if_labeled=False):
    for group in optimizer.param_groups:
        learning_rate = group["lr"]
    start_time = time.time()
    net.zero_grad()
    sentences, lengths, mask = batch.BERT
    labels, l_lengths = batch.label
    scores, source_scores = net(sentences, lengths, decode=False)

    loss = net.crit(scores, source_scores, labels.cuda(device=device), mask, if_labeled=if_labeled)  # / params.batch_size
    if isinstance(loss, int):
        return 0
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    optimizer.step()

    scheduler.step()
    return loss


def trainer(data: Data, params: MView):
    pretrain_models_root = Path(params.model_path)
    pretrain_models_path = [pretrain_models_root / f'{source_name}_{params.name}50.pt' for source_name in data.source_langs]

    net: Assembled_MView = Assembled_MView(
        Softmax(model=params.BERT, num_labels=len(data.label_dict), dropout=0.1, device=data.device),
        pretrain_models_path, consensus=params.consensus, reduce=params.aggregate_method,
        interpolation=params.interpolation, view_interpolation=params.view_interpolation, mu=params.mu,
        att_dropout=params.att_dropout, lang=params.language, device=data.device
    )
    net.cuda()

    # print(f'Model: "{net}"')
    print('-' * 100)
    print("Parameters:")
    print(f' - mini_batch_size: "{params.batch_size}"')
    print(f' - learning_rate: "{params.HP_lr}"')
    print(f' - bert_learning_rate: "{params.HP_BERT_lr}"')
    print(f' - max_epochs: "{params.max_epoch}"')
    print(f' - consensus loss: "{params.consensus}"')
    print(f' - U/L interpolation: "{params.interpolation}"')
    print(f' - view_interpolation: "{params.view_interpolation}"')
    print(f' - interpolation of symmetric consensus: "{params.mu}"')
    print(f' - sample rate of U/L: "{params.sample_rate}"')
    print(f' - attention dropout: "{params.att_dropout}"')
    print('-' * 100)

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
        for name, value in dict(net.named_parameters()).items():
            if 'source_models.weight_vector' in name or 'combine_layer.bilinears' in name:
                net_params.append({'params': [value], 'lr': params.HP_lr})
            else:
                net_params.append({'params': [value], 'lr': params.HP_BERT_lr})

    optimizer: AdamW = AdamW(net_params, lr=params.HP_BERT_lr, betas=(0.9, 0.999), weight_decay=params.HP_L2)

    model_path = Path('./models') / params.dataset / params.method
    if not (os.path.exists(model_path) and os.path.isdir(model_path)):
        os.makedirs(model_path, exist_ok=True)
    eval_path = model_path / params.name / params.consensus
    if not (os.path.exists(eval_path) and os.path.isdir(eval_path)):
        os.makedirs(eval_path, exist_ok=True)

    evaluator = eval_softmax(params.metric, data.label_dict, eval_path)

    for m in net.source_models.source_models:
        m.eval()
        test_score, _, _, _ = evaluator.calc_score(m, data.test, 'test')
        log_info(f'test_score {test_score}')

    total_number_of_batches = len(data.train)
    total_number_of_batches_labeled = len(data.labeled_data)
    modulo = max(1, int(total_number_of_batches / 10))
    modulo_labeled = max(1, int(total_number_of_batches_labeled / 5))

    total_batches = total_number_of_batches + total_number_of_batches_labeled
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_batches * 0.1,
                                                num_training_steps=total_batches * params.max_epoch)

    p = [1 - params.sample_rate, params.sample_rate]
    unlabeled_data = enumerate(data.train)
    labeled_data = enumerate(data.labeled_data)
    epoch_loss0, epoch_loss1 = 0, 0
    batch_u, batch_l = 0, 0
    total_u, total_l = 0, 0
    start_time = time.time()
    log_info('-' * 100)
    iters = 0
    for epoch in range(params.max_epoch * total_batches):
        net.train()
        U_or_L = np.random.choice(['U', 'L'], p=p)
        if U_or_L in 'U':
            try:
                index0, batch = next(unlabeled_data)
            except StopIteration:
                unlabeled_data = enumerate(data.train)
                index0, batch = next(unlabeled_data)

            batch_u += 1
            total_u += 1
            epoch_loss0 += step_one(net, optimizer, scheduler, modulo + modulo_labeled, batch_l, batch_u, start_time,
                                     total_number_of_batches, batch=batch, if_labeled=False)
        else:
            try:
                index1, batch = next(labeled_data)
            except StopIteration:
                labeled_data = enumerate(data.labeled_data)
                index1, batch = next(labeled_data)

            batch_l += 1
            total_l += 1
            epoch_loss1 += step_one(net, optimizer, scheduler, modulo_labeled + modulo, batch_l, batch_u, start_time,
                                     total_number_of_batches_labeled, batch=batch,
                                     if_labeled=True)

        if (batch_u + batch_l) % (modulo + modulo_labeled) == 0:
            batch_time = time.time() - start_time
            for group in optimizer.param_groups:
                learning_rate = group["lr"]
            log_info(
                f"epoch {iters} - iter ({batch_u}){batch_u + batch_l}/{total_batches} - "
                f"loss {epoch_loss0 / (batch_u + 1e-8):.4f}/{epoch_loss1 / (batch_l + 1e-8):.4f} - "
                f"learning_rate: {learning_rate:.8f} - "
                f"samples/sec: {(batch_u + batch_l) * params.batch_size / batch_time:.2f}",
                dynamic=False
            )

        if (epoch + 1) % total_batches == 0:
            log_info('-' * 100)
            loss_u, loss_l = epoch_loss0 / (batch_u + 1e-8), epoch_loss1 / (batch_l + 1e-8)
            batch_u, batch_l = 0, 0
            epoch_loss0, epoch_loss1 = 0, 0

            labeled_score, labeled_score_e, _, _ = evaluator.calc_score(net, data.labeled_data, 'val')

            test_score, test_score_e, _, _ = evaluator.calc_score(net, data.test, 'test')
            source_model_test_score, source_model_test_score_e, _, _ = evaluator.calc_score(net, data.test, 'test', view='source_model')
            log_info(f'loss_u {loss_u} loss_l {loss_l} test_score {test_score} labeled_score {labeled_score}', dynamic=False)
            log_info(f'source_model_test_score {source_model_test_score}')
            start_time = time.time()
            iters += 1
                                    
    if params.save_model:
        log_info(f'saving model... {str(model_path / "best_model.pt")}')
        save_name = f'{params.name}_{params.consensus}.pt'
        torch.save({
            'task': params.task,
            'metric': params.metric,
            'model_state_dict': net.state_dict()
        }, model_path / save_name)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Learning with ncrf or triLinear-ncrf')
    parser.add_argument('--config', default='conll_en.config', help='Path of .config file')
    parser.add_argument('--aggregate_method', default='att', help='aggregate multiple sources')
    parser.add_argument('--source_name', default='en,nl,es', help='source model prefix')
    parser.add_argument('--language', default='conll_03_german', help='target language corpus')
    parser.add_argument('--name', default='de', help='target language name')

    args = parser.parse_args()
    # config_name = args.config
    # cluster = args.cluster
    # gpu = 2
    # torch.cuda.set_device(gpu)
    # device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # config_file = Path('./config/mview') / Path(config_name)
    # log_info(config_file)
    # config = configparser()
    # config.read(config_file, encoding='utf-8')
    params = MView(args)

    # just run one ex.
    main(params, device)
