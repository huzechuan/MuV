import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import data_preprocess.utils as utils
from typing import Tuple, List, Dict
# from .top_layer import Softmax

def init_lstm(input_lstm):
    """
    Initialize lstm
    author:: Liyuan Liu
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


class LSTM_Net(nn.Module):
    """LSTM_CRF model

    args:
        vocab_size: size of word dictionary
        num_labels: size of label set
        embedding_dim: size of word embedding
        hidden_dim: size of word-level blstm hidden dim
        rnn_layers: number of word-level lstm layers
        dropout_ratio: dropout ratio
        large_CRF: use CRF_L or not, refer model.crf.CRF_L and model.crf.CRF_S for more details
    """

    def __init__(self, tag_map: Dict, embedding_length: int,
                 hidden_dim: int, rnn_layers: int,
                 dropout_ratio: float, use_crf=False):
        super(LSTM_Net, self).__init__()
        self.embedding_dim: int = embedding_length
        self.hidden_dim: int = hidden_dim
        self.use_crf: bool = use_crf
        self.tag_map: Dict = tag_map

        # RNN Module
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim // 2,
                            num_layers=rnn_layers, bidirectional=True)
        self.rnn_layers: int = rnn_layers

        # Dropout Module
        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.num_labels: int = len(tag_map)
        self.top_layer = Softmax(hidden_dim, self.num_labels)
            # CrossEntropyLoss
        self.loss = nn.CrossEntropyLoss(reduction='sum')

        self.batch_size: int = None
        self.seq_length: int = None

    def set_batch_size(self, bsize):
        """
        set batch size
        """
        self.batch_size = bsize

    def set_batch_seq_size(self, sizes: Tuple):
        """
        set batch size and sequence length
        """
        if isinstance(sizes, Tuple):
            self.batch_size, self.seq_length = sizes
        else:
            self.batch_size, self.seq_length = sizes.size(1), sizes.size(0)

    def rand_init(self):
        """
        random initialization

        args:
            init_embedding: random initialize embedding or not
        """
        init_lstm(self.lstm)
        if self.use_crf:
            self.top_layer.rand_init()

    def crit(self, scores, tags, masks):
        loss = self.top_layer.crit(scores, tags, masks)

        return loss

    def forward(self, feats, lengths, hidden=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''
        # embeds = self.embeddings.embed(feats)
        embeds = feats
        self.set_batch_seq_size(embeds)
        total_len = self.seq_length#feats[0].total_len

        d_embeds = self.dropout1(embeds)

        d_embeds = pack_padded_sequence(input=d_embeds, lengths=lengths, batch_first=False, enforce_sorted=False)
        lstm_out, hidden = self.lstm(d_embeds, hidden)
        # lstm_out = lstm_out.view(-1, self.hidden_dim)
        lstm_out, batch_lens = pad_packed_sequence(lstm_out, batch_first=False, padding_value=0.0, total_length=total_len)

        d_lstm_out = self.dropout2(lstm_out)

        score = self.top_layer(d_lstm_out)

        if self.use_crf:
            score = score.view(self.seq_length, self.batch_size, self.num_labels, self.num_labels)
        else:
            score = score.view(self.seq_length, self.batch_size, self.num_labels)

        return score#, hidden

    def decode(self, scores):
        _, tags = torch.max(scores, 2)

        return tags