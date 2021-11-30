from pathlib import Path

import torch.nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from typing import List, Tuple, Union, Dict

from .parser.biaffine_attention import BiaffineAttention
from .parser import BiLSTM, Biaffine, MLP
from .parser.dropout import IndependentDropout, SharedDropout
from .parser import alg
from tqdm import tqdm
import numpy as np
import pdb
import copy
import time
from .metric import Metric
import sys

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)

import torch.nn.functional as F

# Part of Codes are from https://github.com/yzhangcs/biaffine-parser
class SemanticDependencyParser(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            tag_dictionary: Dict,
            # tag_type: str,
            use_crf: bool = False,
            use_rnn: bool = True,
            train_initial_hidden_state: bool = False,
            punct: bool = False,  # ignore all punct in default
            tree: bool = False,  # keep the dpendency with tree structure
            n_mlp_arc=500,
            n_mlp_rel=100,
            mlp_dropout=.33,
            use_sib=True,
            use_gp=True,
            use_cop=False,
            iterations=3,
            # binary=False,
            # is_mst=False,
            rnn_layers: int = 3,
            lstm_dropout: float = 0.33,
            dropout: float = 0.0,
            word_dropout: float = 0.33,
            locked_dropout: float = 0.5,
            pickle_module: str = "pickle",
            interpolation: float = 0.5,
            factorize_interpolation: float = 0.025,
            config=None,
            use_decoder_timer=True,
            debug=False,
            diagonal: bool = False,
            is_srl: bool = False,
            device: str = 'cpu'
    ):
        """
		Initializes a SequenceTagger
		:param hidden_size: number of hidden states in RNN
		:param embeddings: word embeddings used in tagger
		:param tag_dictionary: dictionary of tags you want to predict
		:param tag_type: string identifier for tag type
		:param use_crf: if True use CRF decoder, else project directly to tag space
		:param use_rnn: if True use RNN layer, otherwise use word embeddings directly
		:param rnn_layers: number of RNN layers
		:param dropout: dropout probability
		:param word_dropout: word dropout probability
		:param locked_dropout: locked dropout probability
		:param distill_crf: CRF information distillation
		:param crf_attention: use CRF distillation weights
		:param biaf_attention: use bilinear attention for word-KD distillation
		"""

        super(SemanticDependencyParser, self).__init__()
        self.binary = False
        self.is_mst = True
        self.tree = True
        self.debug = False
        self.biaf_attention = False
        self.token_level_attention = False
        self.use_language_attention = False
        self.use_language_vector = False
        self.use_crf = use_crf
        self.use_decoder_timer = False
        self.sentence_level_loss = False
        self.train_initial_hidden_state = train_initial_hidden_state
        # add interpolation for target loss and distillation loss

        self.interpolation = interpolation
        self.debug = debug
        # self.use_rnn = use_rnn
        self.encoder = encoder
        self.hidden_size = self.encoder.bilstm.hidden_size

        # self.rnn_layers: int = rnn_layers
        # extract embedding, etc, Multilingual BERT
        # encoder, etc, BiLSTM
        self.config = config
        self.punct = punct
        self.punct_list = ['``', "''", ':', ',', '.', 'PU', 'PUNCT']
        # self.tree = tree
        # set the dictionaries
        self.tag_dictionary: Dict = tag_dictionary
        # self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)

        self.diagonal = diagonal

        # initialize the network architecture
        # self.nlayers: int = rnn_layers
        self.hidden_word = None

        # dropouts

        self.pickle_module = pickle_module

        # rnn_input_dim: int = self.embeddings.embedding_length

        # self.bidirectional = True
        # self.rnn_type = "LSTM"
        # if not self.use_rnn:
        #     self.bidirectional = False
        # bidirectional LSTM on top of embedding layer
        num_directions = 1

        # hiddens
        self.n_mlp_arc = n_mlp_arc
        self.n_mlp_rel = n_mlp_rel
        self.mlp_dropout = mlp_dropout
        # self.lstm_dropout = lstm_dropout
        # Initialization of Biaffine Parser
        # self.embed_dropout = IndependentDropout(p=word_dropout)
        # if self.encoder.name == 'bilstm':
        #     self.lstm_dropout_func = SharedDropout(p=self.lstm_dropout)
        mlp_input_hidden = self.hidden_size * 2
        # else:
        #     mlp_input_hidden = rnn_input_dim

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=mlp_input_hidden,
                             n_hidden=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_arc_d = MLP(n_in=mlp_input_hidden,
                             n_hidden=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=mlp_input_hidden,
                             n_hidden=n_mlp_rel,
                             dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=mlp_input_hidden,
                             n_hidden=n_mlp_rel,
                             dropout=mlp_dropout)
        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=self.tagset_size,
                                 bias_x=True,
                                 bias_y=True,
                                 diagonal=self.diagonal, )
        # self.binary = binary
        # the Second Order Parts

        # self.pad_index = pad_index
        # self.unk_index = unk_index
        self.rel_criterion = nn.CrossEntropyLoss()
        self.arc_criterion = nn.CrossEntropyLoss()

        self.to(device)

    def _init_model_with_state_dict(state, embed_net):
        use_dropout = 0.0 if not "use_dropout" in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if not "use_word_dropout" in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if not "use_locked_dropout" in state.keys()
            else state["use_locked_dropout"]
        )

        model = SemanticDependencyParser(
            # embed_net=state["embed_net"],
            encoder=state["encoder"],
            tag_dictionary=state["tag_dictionary"],
            # tag_type=state["tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            tree=state["tree"],
            punct=state["punct"],
            train_initial_hidden_state=state["train_initial_hidden_state"],
            n_mlp_arc=state["n_mlp_arc"],
            n_mlp_rel=state["n_mlp_rel"],
            mlp_dropout=state["mlp_dropout"],
            use_sib=state["use_sib"],
            use_gp=state["use_gp"],
            use_cop=state["use_cop"],
            iterations=state["iterations"],
            rnn_layers=state["rnn_layers"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            config=state['config'] if "config" in state else None,
            diagonal=False if 'diagonal' not in state else state['diagonal'],
        )
        model.load_state_dict(state["state_dict"])
        return model

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            # "embed_net": self.embed_net,
            "encoder": self.encoder,
            "tag_dictionary": self.tag_dictionary,
            # "tag_type": self.tag_type,
            "use_crf": self.use_crf,
            "use_rnn": self.use_rnn,
            "tree": self.tree,
            "punct": self.punct,
            "train_initial_hidden_state": self.train_initial_hidden_state,
            "n_mlp_arc": self.n_mlp_arc,
            "n_mlp_rel": self.n_mlp_rel,
            "mlp_dropout": self.mlp_dropout,
            "n_mlp_sec": self.n_mlp_sec,
            "use_sib": self.use_sib,
            "use_gp": self.use_gp,
            "use_cop": self.use_cop,
            "iterations": self.iterations,
            "rnn_layers": self.rnn_layers,
            "dropout": self.use_dropout,
            "word_dropout": self.use_word_dropout,
            "locked_dropout": self.use_locked_dropout,
            "config": self.config,
            "diagonal": self.diagonal,
        }
        return model_state

    def forward(self, sentences, lengths, mask=None):
        # self.zero_grad()
        # lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        # longest_token_sequence_in_batch: int = max(lengths)

        # self.embeddings.embed(sentences)
        # features, _ = self.embed_net(sentences)

        # sentence_tensor = self.embed_dropout(features)[0]

        # if self.use_rnn:
        #     x = pack_padded_sequence(sentence_tensor, lengths, True, False)
        #     x, _ = self.rnn(x)
        #     sentence_tensor, _ = pad_packed_sequence(x, True, total_length=sentence_tensor.shape[1])
        #     sentence_tensor = self.lstm_dropout_func(sentence_tensor)

        # mask = self.sequence_mask(torch.tensor(lengths), longest_token_sequence_in_batch).cuda().type_as(
        #     sentence_tensor)
        encoded_x, encoded_sentence = self.encoder(sentences, lengths)
        self.mask = mask
        # mask = words.ne(self.pad_index)
        # lens = mask.sum(dim=1)

        # get outputs from embedding layers
        # x = sentence_tensor
        x = encoded_x
        # apply MLPs to the BiLSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        # set the scores that exceed the length of each sentence to -inf
        # if not self.binary:
        #     s_arc.masked_fill_(~mask.unsqueeze(1).bool(), float(-1e9))
        return s_arc, s_rel, x, encoded_sentence


    def forward_loss(
            self, s_arc, s_rel, gold_arc, gold_rel
    ) -> torch.tensor:
        # s_arc, s_rel = self.forward(data_points)
        # lengths = [len(sentence.tokens) for sentence in data_points]
        # longest_token_sequence_in_batch: int = max(lengths)

        # max_len = features.shape[1]
        # mask=self.sequence_mask(torch.tensor(lengths), max_len).cuda().type_as(features)
        loss = self._calculate_loss(s_arc, s_rel, gold_arc, gold_rel, self.mask)
        return loss

    def _calculate_loss(
            self, arc_scores: torch.tensor, rel_scores: torch.tensor, arcs, rels, mask: torch.tensor,
            return_arc_rel=False,
    ) -> float:
        if self.binary:
            root_mask = mask.clone()
            root_mask[:, 0] = 0
            binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)
            # arc_mat=
            if hasattr(sentences, self.tag_type + '_arc_tags'):
                arc_mat = getattr(sentences, self.tag_type + '_arc_tags').to(flair.device).float()
            else:
                arc_mat = torch.stack(
                    [getattr(sentence, self.tag_type + '_arc_tags').to(flair.device) for sentence in sentences],
                    0).float()
            if hasattr(sentences, self.tag_type + '_rel_tags'):
                rel_mat = getattr(sentences, self.tag_type + '_rel_tags').to(flair.device).long()
            else:
                rel_mat = torch.stack(
                    [getattr(sentence, self.tag_type + '_rel_tags').to(flair.device) for sentence in sentences],
                    0).long()

            arc_loss = self.arc_criterion(arc_scores, arc_mat)
            rel_loss = self.rel_criterion(rel_scores.reshape(-1, self.tagset_size), rel_mat.reshape(-1))
            arc_loss = (arc_loss * binary_mask).sum() / binary_mask.sum()

            rel_mask = (rel_mat > 0) * binary_mask
            num_rels = rel_mask.sum()
            if num_rels > 0:
                rel_loss = (rel_loss * rel_mask.view(-1)).sum() / num_rels
            else:
                rel_loss = 0
        # rel_loss = (rel_loss*rel_mat.view(-1)).sum()/rel_mat.sum()
        else:
            # if hasattr(sentences, self.tag_type + '_arc_tags'):
            #     arcs = getattr(sentences, self.tag_type + '_arc_tags').to(flair.device).long()
            # else:
            #     arcs = torch.stack(
            #         [getattr(sentence, self.tag_type + '_arc_tags').to(flair.device) for sentence in sentences],
            #         0).long()
            # if hasattr(sentences, self.tag_type + '_rel_tags'):
            #     rels = getattr(sentences, self.tag_type + '_rel_tags').to(flair.device).long()
            # else:
            #     rels = torch.stack(
            #         [getattr(sentence, self.tag_type + '_rel_tags').to(flair.device) for sentence in sentences],
            #         0).long()
            self.arcs = arcs
            self.rels = rels
            mask[:, 0] = False
            # mask = mask.bool()
            gold_arcs = arcs[mask]
            rel_scores, rels = rel_scores[mask], rels[mask]
            rel_scores = rel_scores[torch.arange(len(gold_arcs)), gold_arcs]

            if self.use_crf:
                arc_loss, arc_probs = crf(arc_scores, mask, arcs)
                arc_loss = arc_loss / mask.sum()
                rel_loss = self.rel_criterion(rel_scores, rels)

            # =============================================================================================
            # dist=generate_tree(arc_scores,mask,is_mst=self.is_mst)
            # labels = dist.struct.to_parts(arcs[:,1:], lengths=mask.sum(-1)).type_as(arc_scores)
            # log_prob = dist.log_prob(labels)
            # if (log_prob>0).any():

            #   log_prob[torch.where(log_prob>0)]=0
            #   print("failed to get correct loss!")
            # if self.token_loss:
            #   arc_loss = - log_prob.sum()/mask.sum()
            # else:
            #   arc_loss = - log_prob.mean()

            # self.dist=dist

            # rel_loss = self.rel_criterion(rel_scores, rels)
            # if self.token_loss:
            #   rel_loss = rel_loss.mean()
            # else:
            #   rel_loss = rel_loss.sum()/len(sentences)

            # if self.debug:
            #   if rel_loss<0 or arc_loss<0:
            #       pdb.set_trace()
            # =============================================================================================
            else:
                arc_scores, arcs = arc_scores[mask], arcs[mask]
                arc_loss = self.arc_criterion(arc_scores, arcs)

                # rel_scores, rels = rel_scores[mask], rels[mask]
                # rel_scores = rel_scores[torch.arange(len(arcs)), arcs]

                rel_loss = self.rel_criterion(rel_scores, rels)
        if return_arc_rel:
            return (arc_loss, rel_loss)
        loss = 2 * ((1 - self.interpolation) * arc_loss + self.interpolation * rel_loss)

        # score = torch.nn.functional.cross_entropy(features.view(-1,features.shape[-1]), tag_list.view(-1,), reduction='none') * mask.view(-1,)

        # if self.sentence_level_loss or self.use_crf:
        #   score = score.sum()/features.shape[0]
        # else:
        #   score = score.sum()/mask.sum()

        #   score = (1-self.posterior_interpolation) * score + self.posterior_interpolation * posterior_score
        return loss

    def evaluate(
            self,
            data_loader,
            out_path: Path = None,
            embeddings_storage_mode: str = "cpu",
            prediction_mode: bool = False,
    ):
        # data_loader.assign_embeddings()
        with torch.no_grad():
            if self.binary:
                eval_loss = 0

                batch_no: int = 0

                # metric = Metric("Evaluation")
                # sentence_writer = open('temps/'+str(uid)+'_eval'+'.conllu','w')
                lines: List[str] = []
                utp = 0
                ufp = 0
                ufn = 0
                ltp = 0
                lfp = 0
                lfn = 0
                for batch in data_loader:
                    batch_no += 1

                    arc_scores, rel_scores = self.forward(batch)
                    mask = self.mask
                    root_mask = mask.clone()
                    root_mask[:, 0] = 0
                    binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)

                    arc_predictions = (arc_scores.sigmoid() > 0.5) * binary_mask
                    rel_predictions = (rel_scores.softmax(-1) * binary_mask.unsqueeze(-1)).argmax(-1)
                    if not prediction_mode:
                        arc_mat = torch.stack(
                            [getattr(sentence, self.tag_type + '_arc_tags').to(flair.device) for sentence in batch],
                            0).float()
                        rel_mat = torch.stack(
                            [getattr(sentence, self.tag_type + '_rel_tags').to(flair.device) for sentence in batch],
                            0).long()
                        loss = self._calculate_loss(arc_scores, rel_scores, batch, mask)
                        if self.is_srl:
                            # let the head selection fixed to the gold predicate only
                            binary_mask[:, :, 0] = arc_mat[:, :, 0]
                            arc_predictions = (arc_scores.sigmoid() > 0.5) * binary_mask

                        # UF1
                        true_positives = arc_predictions * arc_mat
                        # (n x m x m) -> ()
                        n_predictions = arc_predictions.sum()
                        n_unlabeled_predictions = n_predictions
                        n_targets = arc_mat.sum()
                        n_unlabeled_targets = n_targets
                        n_true_positives = true_positives.sum()
                        # () - () -> ()
                        n_false_positives = n_predictions - n_true_positives
                        n_false_negatives = n_targets - n_true_positives
                        # (n x m x m) -> (n)
                        n_targets_per_sequence = arc_mat.sum([1, 2])
                        n_true_positives_per_sequence = true_positives.sum([1, 2])
                        # (n) x 2 -> ()
                        n_correct_sequences = (n_true_positives_per_sequence == n_targets_per_sequence).sum()
                        utp += n_true_positives
                        ufp += n_false_positives
                        ufn += n_false_negatives

                        # LF1
                        # (n x m x m) (*) (n x m x m) -> (n x m x m)
                        true_positives = (rel_predictions == rel_mat) * arc_predictions
                        correct_label_tokens = (rel_predictions == rel_mat) * arc_mat
                        # (n x m x m) -> ()
                        # n_unlabeled_predictions = tf.reduce_sum(unlabeled_predictions)
                        # n_unlabeled_targets = tf.reduce_sum(unlabeled_targets)
                        n_true_positives = true_positives.sum()
                        n_correct_label_tokens = correct_label_tokens.sum()
                        # () - () -> ()
                        n_false_positives = n_unlabeled_predictions - n_true_positives
                        n_false_negatives = n_unlabeled_targets - n_true_positives
                        # (n x m x m) -> (n)
                        n_targets_per_sequence = arc_mat.sum([1, 2])
                        n_true_positives_per_sequence = true_positives.sum([1, 2])
                        n_correct_label_tokens_per_sequence = correct_label_tokens.sum([1, 2])
                        # (n) x 2 -> ()
                        n_correct_sequences = (n_true_positives_per_sequence == n_targets_per_sequence).sum()
                        n_correct_label_sequences = (
                        (n_correct_label_tokens_per_sequence == n_targets_per_sequence)).sum()
                        ltp += n_true_positives
                        lfp += n_false_positives
                        lfn += n_false_negatives

                        eval_loss += loss

                    if out_path is not None:
                        masked_arc_scores = arc_scores.masked_fill(~binary_mask.bool(), float(-1e9))
                        # if self.target
                        # lengths = [len(sentence.tokens) for sentence in batch]

                        # temp_preds = eisner(arc_scores, mask)
                        if not self.is_mst:
                            temp_preds = eisner(arc_scores, root_mask.bool())
                        for (sent_idx, sentence) in enumerate(batch):
                            if self.is_mst:
                                preds = MST_inference(torch.softmax(masked_arc_scores[sent_idx], -1).cpu().numpy(),
                                                      len(sentence), binary_mask[sent_idx].cpu().numpy())
                            else:
                                preds = temp_preds[sent_idx]

                            for token_idx, token in enumerate(sentence):
                                if token_idx == 0:
                                    continue

                                # append both to file for evaluation
                                arc_heads = torch.where(arc_predictions[sent_idx, token_idx] > 0)[0]
                                if preds[token_idx] not in arc_heads:
                                    val = torch.zeros(1).type_as(arc_heads)
                                    val[0] = preds[token_idx].item()
                                    arc_heads = torch.cat([arc_heads, val], 0)
                                if len(arc_heads) == 0:
                                    arc_heads = masked_arc_scores[sent_idx, token_idx].argmax().unsqueeze(0)
                                rel_index = rel_predictions[sent_idx, token_idx, arc_heads]
                                rel_labels = [self.tag_dictionary.get_item_for_index(x) for x in rel_index]
                                arc_list = []
                                for i, label in enumerate(rel_labels):
                                    if '+' in label:
                                        labels = label.split('+')
                                        for temp_label in labels:
                                            arc_list.append(str(arc_heads[i].item()) + ':' + temp_label)
                                    else:
                                        arc_list.append(str(arc_heads[i].item()) + ':' + label)
                                eval_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                                    token_idx,
                                    token.text,
                                    'X',
                                    'X',
                                    'X',
                                    'X=X',
                                    str(token_idx - 1),
                                    'root' if token_idx - 1 == 0 else 'det',
                                    '|'.join(arc_list),
                                    'X',
                                )
                                lines.append(eval_line)
                            lines.append("\n")
                eval_loss /= batch_no
                UF1 = self.compute_F1(utp, ufp, ufn)
                LF1 = self.compute_F1(ltp, lfp, lfn)
                if out_path is not None:
                    with open(out_path, "w", encoding="utf-8") as outfile:
                        outfile.write("".join(lines))
                if prediction_mode:
                    return None, None

                result = Result(
                    main_score=LF1,
                    log_line=f"\nUF1: {UF1} - LF1 {LF1}",
                    log_header="PRECISION\tRECALL\tF1",
                    detailed_results=f"\nUF1: {UF1} - LF1 {LF1}",
                )
            else:
                if prediction_mode:
                    eval_loss, metric = self.dependency_evaluate(data_loader, out_path=out_path,
                                                                 prediction_mode=prediction_mode)
                    return eval_loss, metric
                else:
                    eval_loss, metric = self.dependency_evaluate(data_loader, out_path=out_path)

                UAS = metric.uas
                LAS = metric.las
                # result = Result(main_score=LAS, log_line=f"\nUAS: {UAS} - LAS {LAS}",
                #                 log_header="PRECISION\tRECALL\tF1", detailed_results=f"\nUAS: {UAS} - LAS {LAS}", )
            # return result, eval_loss
        return UAS, LAS

    def compute_F1(self, tp, fp, fn):
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        return 2 * (precision * recall) / (precision + recall + 1e-12)

    @torch.no_grad()
    def dependency_evaluate(self, loader, out_path=None, prediction_mode=False):
        # self.model.eval()

        loss, metric = 0, Metric()
        # total_start_time=time.time()
        # forward_time=0
        # loss_time=0
        # decode_time=0
        # punct_time=0
        lines = []
        for index, batch in enumerate(loader):
            forward_start = time.time()
            sentences, lengths, mask = batch.BERT
            gold_rels, lengths = batch.rel
            gold_arcs, lengths = batch.head
            s_arc, s_res, _, _ = self.forward(sentences, lengths, mask)
            # forward_end=time.time()
            mask[:, 0] = False
            # self.mask
            # if not prediction_mode:
            #     loss += self._calculate_loss(arc_scores, rel_scores, batch, mask)
            # loss_end=time.time()
            # forward_time+=forward_end-forward_start
            # loss_time+=loss_end-forward_end
            # mask = mask.bool()
            # decode_start=time.time()
            arc_preds, rel_preds = self.decode(s_arc, s_res, mask)
            # decode_end=time.time()
            # decode_time+=decode_end-decode_start
            # ignore all punctuation if not specified

            # if not self.punct:
            #     for sent_id, sentence in enumerate(batch):
            #         for token_id, token in enumerate(sentence):
            #             upos = token.get_tag('upos').value
            #             xpos = token.get_tag('pos').value
            #             word = token.text
            #             if is_punctuation(word, upos, self.punct_list) or is_punctuation(word, upos, self.punct_list):
            #                 mask[sent_id][token_id] = 0
            # mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            if out_path is not None:
                for (sent_idx, sentence) in enumerate(batch):
                    for token_idx, token in enumerate(sentence):
                        if token_idx == 0:
                            continue

                        # append both to file for evaluation
                        eval_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                            token_idx,
                            token.text,
                            'X',
                            'X',
                            'X',
                            'X',
                            arc_preds[sent_idx, token_idx],
                            self.tag_dictionary.get_item_for_index(rel_preds[sent_idx, token_idx]),
                            'X',
                            'X',
                        )
                        lines.append(eval_line)
                    lines.append("\n")

            if not prediction_mode:
                # punct_end=time.time()
                # punct_time+=punct_end-decode_end
                metric(arc_preds, rel_preds, gold_arcs.cuda(), gold_rels.cuda(), mask)
        if out_path is not None:
            with open(out_path, "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))
        if prediction_mode:
            return None, None
        # total_end_time=time.time()
        # print(total_start_time-total_end_time)
        # print(forward_time)
        # print(punct_time)
        # print(decode_time)

        loss /= len(loader)

        return loss, metric

    def decode(self, arc_scores, rel_scores, mask):
        arc_preds = arc_scores.argmax(-1)
        bad = [not alg.istree(sequence, not self.is_mst)
               for sequence in arc_preds.tolist()]
        if self.tree and any(bad):
            arc_preds[bad] = alg.mst(arc_scores[bad], mask[bad])# eisner(arc_scores[bad], mask[bad])
        # if not hasattr(self,'dist') or self.is_mst:
        #   dist = generate_tree(arc_scores,mask,is_mst=False)
        # else:
        #   dist = self.dist
        # arc_preds=get_struct_predictions(dist)

        # deal with masking
        # if not (arc_preds*mask == result*mask).all():
        #   pdb.set_trace()

        rel_preds = rel_scores.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds
