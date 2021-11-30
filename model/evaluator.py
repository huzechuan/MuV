import torch
import numpy as np
import itertools

import data_preprocess.utils as utils
from torch.autograd import Variable
from tqdm import tqdm
import sys
import os
import subprocess
import re
from pathlib import Path


class eval_batch:
    """Base class for evaluation, provide method to calculate f1 score and accuracy

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
    """

    def __init__(self, vocab, file_path=None, scheme='BIO', save_data=False, save_path=None):
        self.itos = vocab.itos
        self.file_path = file_path
        self.scheme = scheme
        self.save_data = save_data
        if self.save_data:
            self.save_path = save_path

    def reset_path(self, new_path):
        """ reset the path of target language.
            used for saving direct method's pseudo data.
        """
        self.save_path = new_path

    def reset(self):
        """
        re-set all states
        """
        self.correct_labels = 0
        self.total_labels = 0

    def calc_acc_batch(self, decoded_data, target_data, lengths, *args):
        """
        update statics for accuracy

        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        batch_decoded = torch.unbind(decoded_data.cpu(), 0)
        batch_targets = torch.unbind(target_data, 0)

        for decoded, target, lens in zip(batch_decoded, batch_targets, lengths.cpu()):
            gold = target % len(self.itos)
            # remove padding
            gold = gold[:lens].numpy()
            best_path = decoded[:lens].numpy()

            self.total_labels += lens.numpy()
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def write_result(self, decoded_data, target_data, lengths, sentences, fout):

        batch_decoded = torch.unbind(decoded_data.cpu(), 0)
        # batch_decoded = decoded_data
        batch_targets = torch.unbind(target_data, 0)
        idx2item = self.itos
        # lines = list()
        for predict, target, sentence, lens in zip(batch_decoded, batch_targets, sentences, lengths.cpu()):
            gold = target % len(self.itos)
            predict = predict[:lens].numpy()
            gold = gold[:lens].numpy()
            sentence = sentence[:lens]
            for i in range(lens):
                fout.write(f'{sentence[i]} '
                           f'{idx2item[predict[i]]} '
                           f'{idx2item[gold[i]]} '
                           f'\n')
            fout.write('\n')

    def call_conlleval(self, prefix):
        file_path = self.file_path / f'{prefix}.BIOES'
        file_path_to = self.file_path / f'{prefix}.BIO'
        if self.scheme == 'BIOES':
            tagSchemeConvert = subprocess.check_output(f'python tools/convertResultTagScheme.py {file_path} {file_path_to}',
                                                       shell=True,
                                                       timeout=200)
        # else:
        #     rename = subprocess.check_output(f'mv {file_path} {file_path_to}',
        #                                                shell=True,
        #                                                timeout=200)
        output = subprocess.check_output(f'perl tools/conlleval < {file_path_to}',
                                         shell=True,
                                         timeout=200).decode('utf-8')
        # if 'train' in prefix:
        if self.save_data:
            save_path = self.save_path / f'{prefix}.txt'
            save_pseudo_data = subprocess.check_output(f'mv {file_path_to} {save_path}',
                                                       shell=True,
                                                       timeout=200)
        else:
            delete = subprocess.check_output(f'rm -rf {file_path_to} {file_path}',
                                             shell=True,
                                             timeout=200).decode('utf-8')
        out = output.split('\n')[1]
        assert out.startswith('accuracy'), "Wrong lines"
        result = re.findall(r"\d+\.?\d*", out)
        return float(result[-1]), float(result[1]), float(result[2]), None

    def acc_score(self, *args):
        """
        calculate accuracy score based on statics
        """
        if 0 == self.total_labels:
            return 0.0
        accuracy = float(self.correct_labels) / self.total_labels
        return round(accuracy * 100, 2), None, None, None



class eval_single_softmax(eval_batch): # for self-training approaches
    """evaluation class for word level model (LSTM-CRF)

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, metric, dict, file_path, scheme='BIO', save_data=False):
        eval_batch.__init__(self, dict, file_path, scheme=scheme, save_data=save_data)

        self.eval_method = metric
        if 'f' in metric:
            # self.eval_b = self.calc_f1_batch
            # self.calc_s = self.f1_score
            self.eval_b = self.write_result
            self.calc_s = self.call_conlleval
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, net, dataset_loader, file_prefix=None, view='x'):
        """
        calculate score for pre-selected metrics

        args:
            ner_model: LSTM-CRF model
            dataset_loader: loader class for test set
        """
        if 'f' in self.eval_method:
            fout = open(self.file_path / f'{file_prefix}.BIO', 'w')
        else:
            fout = None

        with torch.no_grad():

            net.eval()
            self.reset()
            for index, batch in enumerate(dataset_loader):
                sentences, lengths, mask = batch.BERT
                labels, _ = batch.label
                if isinstance(sentences, tuple):
                    _, sentences = sentences
                scores, _, _ = net(sentences, lengths)
                decoded = net.decode(scores, view=view)
                self.eval_b(decoded, labels, lengths, sentences, fout)

        if 'f' in self.eval_method:
            fout.close()
        return self.calc_s(file_prefix)

class eval_tri_softmax(eval_batch): # for tri-training approaches
    """evaluation class for word level model (LSTM-CRF)

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, metric, dict, file_path, scheme='BIO', save_data=False):
        eval_batch.__init__(self, dict, file_path, scheme=scheme, save_data=save_data)

        self.eval_method = metric
        if 'f' in metric:
            # self.eval_b = self.calc_f1_batch
            # self.calc_s = self.f1_score
            self.eval_b = self.write_result
            self.calc_s = self.call_conlleval
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, net, dataset_loader, file_prefix=None, view='x'):
        """
        calculate score for pre-selected metrics

        args:
            ner_model: LSTM-CRF model
            dataset_loader: loader class for test set
        """
        if 'f' in self.eval_method:
            fout = open(self.file_path / f'{file_prefix}.BIO', 'w')
        else:
            fout = None

        with torch.no_grad():

            net.eval()
            self.reset()
            for index, batch in enumerate(dataset_loader):
                sentences, lengths, mask = batch.BERT
                labels, _ = batch.label
                if isinstance(sentences, tuple):
                    _, sentences = sentences
                scores = net(sentences, lengths, two_model=False)
                decoded = net.decode(scores, lengths, view=view)
                self.eval_b(decoded, labels, lengths, sentences, fout)

        if 'f' in self.eval_method:
            fout.close()
        return self.calc_s(file_prefix)


    def calc_acc_batch(self, decoded_data, target_data, lengths, *args):
        """
        update statics for accuracy

        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        batch_decoded = decoded_data
        batch_targets = torch.unbind(target_data, 0)

        for decoded, target, lens in zip(batch_decoded, batch_targets, lengths.cpu()):
            gold = target % len(self.itos)
            # remove padding
            gold = gold[:lens].numpy()
            best_path = decoded[:lens]

            self.total_labels += lens.numpy()
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def write_result(self, decoded_data, target_data, lengths, sentences, fout):

        batch_decoded = decoded_data
        batch_targets = torch.unbind(target_data, 0)
        idx2item = self.itos
        lines = list()
        for predict, target, sentence, lens in zip(batch_decoded, batch_targets, sentences, lengths.cpu()):
            gold = target % len(self.itos)
            predict = predict[:lens]
            gold = gold[:lens].numpy()
            sentence = sentence[:lens]
            for i in range(lens):
                # lines.append(f'{sentence[i]} '
                #              f'{idx2item[predict[i]]} '
                #              f'{idx2item[gold[i]]}\n')
                fout.write(f'{sentence[i]} '
                           f'{idx2item[predict[i]]} '
                           f'{idx2item[gold[i]]} '
                           f'\n')
            fout.write('\n')

class eval_softmax(eval_batch): # for Multi-view approaches
    """evaluation class for word level model (LSTM-CRF)

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, metric, dict, file_path, scheme='BIO', save_data=False):
        eval_batch.__init__(self, dict, file_path, scheme=scheme, save_data=save_data)

        self.eval_method = metric
        if 'f' in metric:
            # self.eval_b = self.calc_f1_batch
            # self.calc_s = self.f1_score
            self.eval_b = self.write_result
            self.calc_s = self.call_conlleval
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, net, dataset_loader, file_prefix=None, view='x', **kwargs):
        """
        calculate score for pre-selected metrics

        args:
            ner_model: LSTM-CRF model
            dataset_loader: loader class for test set
        """
        if 'f' in self.eval_method:
            fout = open(self.file_path / f'{file_prefix}.BIO', 'w')
        else:
            fout = None

        with torch.no_grad():
            net.eval()
            self.reset()
            for index, batch in enumerate(dataset_loader):
                sentences, lengths, mask = batch.BERT
                labels, _ = batch.label
                if isinstance(sentences, tuple):
                    _, sentences = sentences
                scores = net(sentences, lengths, mask=mask)
                decoded = net.decode(scores, view=view)
                # if view != 'x':
                #     decoded = net.decode(scores, masks=mask, view=view, labels=labels.cuda())
                # else:
                #     decoded = net.decode(scores, view=view)
                # decoded = net.decode(scores, view=view, gold=labels)
                self.eval_b(decoded, labels, lengths, sentences, fout)

        if 'f' in self.eval_method:
            fout.close()
        return self.calc_s(file_prefix)
