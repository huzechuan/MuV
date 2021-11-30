from torchtext.vocab import Vectors
from torchtext.data import Example, Dataset, Field
from torchtext.datasets import SequenceTaggingDataset
import data_preprocess
import data_preprocess.dataset as dataset
import torch
from pathlib import Path

class TransfomerField(Field):
    def __init__(self, device='cpu', use_crf=False, task='seq', method=None, **kwargs):
        """

        :param device:
        :param use_crf: default false
        :param task:
                    seq: sequence labeling
                    parsing: semantic dependency parsing, add <ROOT> word in the 0-th position
        :param kwargs:
        """
        self.device = device
        self.use_crf = use_crf
        self.num_sentences = 0
        self.task = task
        self.method = method
        super(TransfomerField, self).__init__(**kwargs)

    def preprocess(self, x):
        sent_id = self.num_sentences
        sentence = Sentence(text=x, sent_id=sent_id)
        self.num_sentences += 1
        return sentence

    def pad(self, minibatch):
        max_len = len(max(minibatch, key=len))
        if self.use_crf:
            mask = torch.zeros(
                [len(minibatch), max_len + 1],
                dtype=torch.bool,
                device=self.device,
            )
        else:
            mask = torch.zeros(
                [len(minibatch), max_len],
                dtype=torch.bool,
                device=self.device,
            )

        lengths = []
        sentences = []
        texts = []
        for s_id, sentence in enumerate(minibatch):
            lens = len(sentence)
            if self.use_crf:
                mask[s_id][:lens+1] = torch.ones(lens+1, dtype=torch.bool)
            else:
                mask[s_id][:lens] = torch.ones(lens, dtype=torch.bool)
            lengths.append(lens)
            texts.append(sentence.text)
            sentences.append(sentence)
        if self.method in ['self-training']:
            arr = sentences, texts, lengths, mask
        else:
            arr = texts, lengths, mask
        return arr

    def numericalize(self, arr, device=None):
        if self.method in ['self-training']:
            sentences, texts, lengths, mask = arr
            lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
            return (sentences, texts), lengths, mask
        else:
            texts, lengths, mask = arr
            lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
            return texts, lengths, mask


class CoNLL(SequenceTaggingDataset):

    # Universal Dependencies English Web Treebank.
    # Download original at http://universaldependencies.org/
    # License: http://creativecommons.org/licenses/by-sa/4.0/
    # urls = ['https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip']
    # dirname = 'en-ud-v2'
    name = 'CoNLL'

    @classmethod
    def splits(cls, fields, path=None, root="./data", train="train.txt",
               validation="dev.txt",
               test="test.txt", **kwargs):
        """Downloads and loads the Universal Dependencies Version 2 POS Tagged
        data.
        """
        data_folder = Path(root) / path
        return super(CoNLL, cls).splits(
            fields=fields, root=root, path=str(data_folder), train=train, validation=validation,
            test=test, **kwargs)


class UD(dataset.UDDataset):

    # Universal Dependencies English Web Treebank.
    # Download original at http://universaldependencies.org/
    # License: http://creativecommons.org/licenses/by-sa/4.0/
    # urls = ['https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip']
    # dirname = 'en-ud-v2'
    name = 'UD'

    @classmethod
    def splits(cls, fields, path=None, root="./data", train="train.txt",
               validation="dev.txt",
               test="test.txt", **kwargs):
        """Downloads and loads the Universal Dependencies Version 2.6 POS Tagged
        data.
        """
        data_folder = Path(root) / path

        for file in data_folder.iterdir():
            file_name = file.name
            if 'train.conllu150' in file_name:
                train = file_name
            if 'test.conllu150' in file_name:
                test = file_name
            if 'dev.conllu150' in file_name and validation is not None:
                validation = file_name
        return super(UD, cls).splits(
            fields=fields, root=root, path=str(data_folder), train=train, validation=validation,
            test=test, **kwargs)

class LangClassfier(dataset.LCDataset):

    # Universal Dependencies English Web Treebank.
    # Download original at http://universaldependencies.org/
    # License: http://creativecommons.org/licenses/by-sa/4.0/
    # urls = ['https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip']
    # dirname = 'en-ud-v2'
    name = 'LangClassfier'

    @classmethod
    def splits(cls, fields, path=None, root="./data", train="train.txt",
               validation="dev.txt",
               test="test.txt", tag=None, **kwargs):
        """Downloads and loads the Universal Dependencies Version 2.6 POS Tagged
        data.
        """
        data_folder = Path(root) / path
        if 'UD' in str(path):
            for file in data_folder.iterdir():
                file_name = file.name
                if 'train.conllu150' in file_name:
                    train = file_name
                if 'test.conllu150' in file_name:
                    test = file_name
                if 'dev.conllu150' in file_name and validation is not None:
                    validation = file_name
        return super(LangClassfier, cls).splits(
            fields=fields, root=root, path=str(data_folder), train=train, validation=validation,
            test=test, tag=tag, **kwargs)


class Data:
    def __init__(self, train=None, labeled_data=None, test=None, dev=None,
                 label_dict=None, embeds=None, sources=None, labeled_data_index=None, device='cpu',
                 **kwargs):
        self.train = train
        self.labeled_data = labeled_data
        self.test = test
        self.label_dict = label_dict
        self.dev = dev
        self.embeds = embeds
        self.source_langs = sources
        self.labeled_data_index = labeled_data_index
        self.device = device

        super(Data, self).__init__()

class Sentence(object):
    def __init__(self, input_id=None, start_ids=None, subword_lens=None, text=None,
                 BERT_embeds=None, length=None, sent_id=None,
                 **kwargs):
        self.input_id = input_id
        self.start_ids = start_ids
        self.subword_lens = subword_lens
        self.BERT_embeds = BERT_embeds
        self.sent_id = sent_id
        self.length = length
        self.text = text
        self.confidence = 0
        self.device = list()
        super(Sentence, self).__init__(**kwargs)

    def __len__(self):
        return len(self.text)
