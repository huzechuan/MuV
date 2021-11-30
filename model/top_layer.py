import torch
import torch.nn as nn
import data_preprocess.utils as utils
from data_preprocess.embedding import TransfomerFeatures
import copy
import math
from collections import Counter
import torch.nn.functional as F


class StackedLayer(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Lample et al. 2016, has less parameters than CRF_L.

    args:
        hidden_dim: input dim size
        tagset_size: target_set_size
        if_biase: whether allow bias in linear trans

    """

    def __init__(self, model, num_labels, if_bias=True, dropout=0.1, device='cpu', fintune=True, **kwargs):
        super(StackedLayer, self).__init__()
        self.model = TransfomerFeatures(model=model, device=device, fine_tune=fintune)
        self.hidden_dim = self.model.embedding_length

        self.num_labels = num_labels
        self.linear_layer = nn.Linear(self.hidden_dim, self.num_labels, bias=if_bias)
        self.dropout = nn.Dropout(p=dropout)
        self._init_weight()

    def _init_weight(self):
        self.linear_layer.weight.data.normal_(mean=0.0, std=0.02)  # 0.02 for xlm-roberta
        self.linear_layer.bias.data.zero_()

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp1, tmp2 = sentence
        self.seq_length = tmp2
        self.batch_size = tmp1

    def crit(self, scores, labels, masks):
        return NotImplemented

    def forward(self, sents, lengths=None, mask=None):
        """
        args:
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer ( (batch_size * seq_len), tag_size, tag_size)
        """
        feats, sent_feats = self.model(sents)
        feats = self.dropout(feats)
        scores = self.linear_layer(feats)
        return scores, feats, sent_feats


class LangClassifier(StackedLayer):
    def __init__(self, model, num_labels, if_bias=True, dropout=0.1, device='cpu', num_source=None, **kwargs):
        super(LangClassifier, self).__init__(model, num_labels, if_bias, dropout, device, **kwargs)
        self.rank = 64
        self.U = nn.Parameter(torch.Tensor(self.hidden_dim, self.rank))
        self.V = nn.Parameter(torch.Tensor(self.hidden_dim, self.rank))
        self.lang_embed = nn.Parameter(torch.FloatTensor(num_source, self.hidden_dim))
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

        self._rand_init()
        self.correct = 0.
        self.total = 1e-32
        self.probs = Counter()
        self.softmax = nn.Softmax(dim=-1)
        self.tao = 4
        for i in range(self.num_labels):
            self.probs[i] = 0

    def _rand_init(self):
        utils.init_embeddings(self.lang_embed)
        utils.init_tensor(self.U)
        utils.init_tensor(self.V)

    def forward(self, sents, lengths=None):
        feats, sent_feats = self.model(sents)

        sent_feats = self.dropout(sent_feats)
        scores = self.d_bilinear(sent_feats)
        return scores

    def d_bilinear(self, feats):
        g0 = torch.matmul(feats, self.U)
        g1 = torch.matmul(self.lang_embed, self.V)
        scores = torch.einsum('br, kr->bk', [g0, g1])
        return scores

    def crit(self, scores, labels):
        labels = labels.view(-1, )
        scores = scores.contiguous().view(-1, self.num_labels)
        loss = self.criterion(scores, labels)
        # loss = loss / self.batch_size
        return loss

    def eval_test(self, data_iter, outpath=None, itos=None):
        self.eval()
        self.total = 0.0
        for i in range(self.num_labels):
            self.probs[i] = 0
        with open(outpath, 'w', encoding='utf-8') as fout:
            fout.write(' '.join(itos) + '\n')
            with torch.no_grad():
                for index, batch in enumerate(data_iter):
                    sentences, lengths, mask = batch.BERT
                    scores = self.forward(sentences, lengths)
                    probs = self.softmax(scores / self.tao)
                    scores = torch.unbind(scores, dim=0)
                    probs = torch.sum(probs, dim=0).tolist()
                    for i in range(self.num_labels): self.probs[i] += probs[i]
                    self.total += len(sentences)
                    for score, text in zip(scores, sentences):
                        text = '#sent=' + ' '.join(text) + '\n'
                        line = ' '.join([str(s) for s in score.tolist()]) + '\n'
                        fout.write(text + line + '\n')
        alpha = [self.probs[i] / self.total for i in range(self.num_labels)]
        return alpha

    def eval_dev(self, data_iter):
        self.eval()
        self.total = 0.0
        self.correct = 0.0
        with torch.no_grad():
            for index, batch in enumerate(data_iter):
                sentences, lengths, mask = batch.BERT
                labels = batch.label
                scores = self.forward(sentences, lengths)
                tags = self.decode(scores).cpu()
                correct = torch.sum(torch.eq(labels, tags))
                self.correct += int(correct)
                self.total += int(labels.size(0))

        return round((self.correct / self.total) * 100, 2)


    def decode(self, scores, confidence=False):
        tags = torch.argmax(scores, dim=-1)
        return tags


class Softmax(StackedLayer):
    def __init__(self, model, num_labels, if_bias=True, dropout=0.1, device='cpu', **kwargs):
        super(Softmax, self).__init__(model, num_labels, if_bias, dropout, device, **kwargs)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def crit(self, scores, labels, masks):
        labels = labels.view(-1, )
        scores = scores.contiguous().view(-1, self.num_labels)
        mask_score = masks.contiguous().view(-1, 1)
        scores = scores.masked_select(mask_score.expand(-1, self.num_labels)).view(-1, self.num_labels)
        masks = masks.contiguous().view(-1, )
        labels = labels.masked_select(masks)
        loss = self.criterion(scores, labels)
        # loss = loss / self.batch_size
        return loss

    def decode(self, scores, masks=None, view=None, confidence=False):
        if isinstance(scores, tuple):
            probs, tags = torch.max(scores[0], 2)
        else:
            probs, tags = torch.max(scores, 2)
        if confidence:
            return probs, tags
        return tags


class Tri_Softmax(StackedLayer):
    """ For multi-task Tri-training

    """
    def __init__(self, model, num_labels, if_bias=True, dropout=0.1, device='cpu', **kwargs):
        super(Tri_Softmax, self).__init__(model, num_labels, if_bias, dropout, device, **kwargs)
        self.linear_layers = clone(self.linear_layer, 3)
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.NLLLoss(reduction='sum')
        self._init_weight()

    def _init_weight(self):
        if not hasattr(self, 'linear_layers'):
            return
        if self.use_bilstm:
            # init_lstm(self.bilstm)
            pass
        else:
            for linear_layer in self.linear_layers:
                linear_layer.weight.data.normal_(mean=0.0, std=0.02)  # 0.02 for xlm-roberta
                linear_layer.bias.data.zero_()

    def crit(self, scores, labels, masks):
        scores = [self.softmax(score) for score in scores]
        labels = labels.view(-1, )
        scores = [score.contiguous().view(-1, self.num_labels) for score in scores]
        mask_score = masks.contiguous().view(-1, 1)
        scores = [score.masked_select(mask_score.expand(-1, self.num_labels)).view(-1, self.num_labels) for score in scores]

        W1, W2 = scores[0].unsqueeze(-1), scores[1].unsqueeze(1)
        val = torch.bmm(W1, W2) # TODO: fix problem
        # constraint = torch.norm(val, dim=(1, 2))
        # constraint = val.norm(dim=(1, 2))
        constraint = torch.tensor(0.).cuda()
        for v in val:
            constraint += v.norm()

        masks = masks.contiguous().view(-1, )
        labels = labels.masked_select(masks)
        loss = [self.criterion(torch.log(score), labels) for score in scores]

        loss = sum(loss) + 0.01 * constraint#torch.sum(constraint)
        # loss = loss / self.batch_size
        return loss

    def forward(self, sents, lengths=None, two_model=False):
        """
        args:
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer ( (batch_size * seq_len), tag_size, tag_size)
        """
        feats, sent_feats = self.model(sents)
        feats = self.dropout(feats)
        if two_model:
            scores = [linear_layer(feats) for linear_layer in self.linear_layers[:-1]]
        else:
            scores = [linear_layer(feats) for linear_layer in self.linear_layers]
        return scores

    def decode(self, scores, lengths=None, view=None, confidence=False):
        tags = [torch.max(score, 2)[-1] for score in scores]
        # lengths = torch.unbind(lengths, dim=0)
        decoded = []
        if len(tags) == 2:
            tags = [torch.unbind(tag.cpu(), dim=0) for tag in tags]
            for tag1, tag2, lens in zip(tags[0], tags[1], lengths.cpu()):
                if torch.equal(tag1[:lens], tag2[:lens]):
                    decoded.append(tag1[:lens].tolist())
                else:
                    decoded.append(None)
        else:
            from collections import Counter
            from random import random
            import numpy as np
            batch, seq = tags[0].size()
            decoded = np.zeros((batch, seq), dtype=np.int)
            tags = torch.stack(tags, dim=-1).cpu().numpy()
            for b in range(batch):
                for s in range(seq):
                    candidate = tags[b][s]
                    tmp = Counter(candidate)
                    top_one = tmp.most_common(1)
                    if top_one[0][1] == 1:  # randomly select
                        tmp_label = np.random.choice(candidate)
                    else:
                        tmp_label = top_one[0][0]
                    decoded[b][s] = tmp_label

        return decoded


def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Direct_Model(nn.Module):
    def __init__(self, model, num_source, reduction='avg', itos=None):
        super(Direct_Model, self).__init__()
        self.num_source = num_source
        self.source_models = clone(model, self.num_source)
        self.hidden_dim = model.hidden_dim
        self.num_labels = model.num_labels
        self.reduction = reduction
        self.itos = itos
        self.criterion = nn.Softmax(dim=-1)

    def load_source_models(self, source_models):
        for i in range(self.num_source):
            self.source_models[i].load_state_dict(source_models[i])
            self.source_models[i].model.fine_tune = False

    def forward(self, sents, lengths=None):
        """
        args:
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer ( (batch_size * seq_len), tag_size, tag_size)
        """
        # input_ids, mask, first_subtokens = feats
        # feats = self.embeds.embed(input_ids, mask, first_subtokens)
        scores = [model(sents)[0] for model in self.source_models]
        scores = [self.criterion(score) for score in scores]
        return scores

    def assign_prediction_cat(self, data_iter):
        train_set = data_iter.dataset
        for idx, ex in enumerate(train_set.examples):
            # setattr(ex.BERT, 'index', idx)
            ex.BERT.sent_id = idx

        if self.reduction == 'cat':
            n_examples = []
            for i in range(self.num_source):
                examples = copy.deepcopy(train_set.examples)
                n_examples.append(examples)

        with torch.no_grad():
            self.eval()
            for index, batch in enumerate(data_iter):
                sentences, lengths, mask = batch.BERT
                labels, _ = batch.label
                if isinstance(sentences, tuple):
                    sentences_obj, sentences = sentences
                scores = self.forward(sentences, lengths)
                n_decoded = self.decode(scores)
                for decoded, examples in zip(n_decoded, n_examples):
                    batch_decoded = torch.unbind(decoded.cpu(), 0)
                    for predict, sent_obj, lens in zip(batch_decoded, sentences_obj, lengths.cpu()):
                        sent_id = sent_obj.sent_id
                        predict = predict[:lens].numpy()
                        predict = [self.itos[p] for p in predict]
                        assert examples[sent_id].BERT.sent_id == sent_id, 'wrong index'
                        examples[sent_id].label = predict
        tmp = []
        for examples in n_examples:
            tmp.extend(examples)
        train_set.examples = tmp
        return train_set

    def assign_prediction(self, data_iter):
        train_set = data_iter.dataset
        for idx, ex in enumerate(train_set.examples):
            # setattr(ex.BERT, 'index', idx)
            ex.BERT.sent_id = idx
        with torch.no_grad():
            self.eval()
            for index, batch in enumerate(data_iter):
                sentences, lengths, mask = batch.BERT
                labels, _ = batch.label
                if isinstance(sentences, tuple):
                    sentences_obj, sentences = sentences
                scores = self.forward(sentences, lengths)
                decoded = self.decode(scores)
                batch_decoded = torch.unbind(decoded.cpu(), 0)
                for predict, sent_obj, lens in zip(batch_decoded, sentences_obj, lengths.cpu()):
                    sent_id = sent_obj.sent_id
                    predict = predict[:lens].numpy()
                    predict = [self.itos[p] for p in predict]
                    assert train_set.examples[sent_id].BERT.sent_id == sent_id, 'wrong index'
                    train_set.examples[sent_id].label = predict
        return train_set

    def decode(self, scores, gold=None, **kwargs):
        if self.reduction not in ['vote', 'cat', 'upper_bound']:
            score_temp = torch.stack(scores, dim=-1)
        if self.reduction == 'avg':
            scores = torch.mean(score_temp, dim=-1)
            tags = torch.argmax(scores, -1)
        elif self.reduction == 'max':
            scores = torch.max(score_temp, dim=-1)[0]
            tags = torch.argmax(scores, -1)
        elif self.reduction == 'vote':
            n_tags = [torch.argmax(score, -1) for score in scores]
            from collections import Counter
            from random import random
            import numpy as np
            batch, seq = n_tags[0].size()
            decoded = np.zeros((batch, seq), dtype=np.int)
            n_tags = torch.stack(n_tags, dim=-1).cpu().numpy()
            for b in range(batch):
                for s in range(seq):
                    candidate = n_tags[b][s]
                    tmp = Counter(candidate)
                    top_one = tmp.most_common(1)
                    if top_one[0][1] == 1:  # randomly select
                        tmp_label = np.random.choice(candidate)
                    else:
                        tmp_label = top_one[0][0]
                    decoded[b][s] = tmp_label
            tags = torch.tensor(decoded, dtype=torch.long).cuda()
        elif self.reduction == 'upper_bound':
            golds = gold
            n_tags = [torch.argmax(score, -1) for score in scores]
            from collections import Counter
            from random import random
            import numpy as np
            batch, seq = n_tags[0].size()
            decoded = np.zeros((batch, seq), dtype=np.int)
            golds = golds.numpy()
            n_tags = torch.stack(n_tags, dim=-1).cpu().numpy()
            for b in range(batch):
                for s in range(seq):
                    candidate = n_tags[b][s]
                    gold = golds[b][s]
                    if gold in candidate:
                        tmp_label = gold
                    else:
                        tmp = Counter(candidate)
                        top_one = tmp.most_common(1)
                        if top_one[0][1] == 1:  # randomly select
                            tmp_label = np.random.choice(candidate)
                        else:
                            tmp_label = top_one[0][0]
                    decoded[b][s] = tmp_label
            tags = torch.tensor(decoded, dtype=torch.long).cuda()
        elif self.reduction == 'cat':
            n_tags = [torch.argmax(score, -1) for score in scores]
            tags = n_tags
        return tags



class sum_(nn.Module):
    def __init__(self):
        super(sum_, self).__init__()

    def forward(self, value, query, key=None, **kwargs):
        # return torch.matmul(value, query)
        return torch.mean(value, dim=-1)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, dropout=None, num_source=3, reduction='att', use_bilinear=True):
        super(MultiheadAttention, self).__init__()
        self.reduction = reduction
        # Value is logits, and its size is not equal to key and query.
        self.d_k = d_model

        # self.linears = clone(nn.Linear(d_model, d_model), 2)
        self.use_bilinear = use_bilinear
        if self.use_bilinear:
            self.bilinears = clone(nn.Bilinear(d_model, d_model, 1), num_source)
        else:
            self.linear_query = clone(nn.Linear(d_model, d_model), 1)
            self.linears_key = clone(nn.Linear(d_model, d_model), num_source)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

        self._init_weight()
        self.flag = True

        self.total = 0.0
        self.summation = torch.zeros(3, dtype=torch.float32)

    def _init_weight(self):
        if self.use_bilinear:
            pass
        else:
            for linear_q in self.linear_query:
                nn.init.xavier_normal_(linear_q.weight.data)
                linear_q.bias.data.zero_()

            for linear_key in self.linears_key:
                nn.init.xavier_normal_(linear_key.weight.data)
                linear_key.bias.data.zero_()

    def reset_analysis(self):
        weights = (self.summation / self.total).numpy()
        self.total = 0.0
        self.summation = torch.zeros(3, dtype=torch.float32)
        return weights

    def forward(self, value, query, key, mask=None):
        n_batches = query.size(0)

        # query, key = [linear(x.transpose(-2, -1)).transpose(-2, -1) for linear, x in zip(self.linears, (query, key))]
        # TODO: use bilinear function
        if self.use_bilinear:
            scores = [bilinear(query, k) for bilinear, k in zip(self.bilinears, key)]
            scores = torch.stack(scores, dim=-2)
        else:
            query = self.linear_query[0](query).unsqueeze(-1)
            key = [linear(x) for linear, x in zip(self.linears_key, key)]
            key = torch.stack(key, dim=-1)

            scores = torch.matmul(key.transpose(-2, -1), query)

        scores /= math.sqrt(self.d_k)

        p_attn = F.softmax(scores, dim=-2)
        # if self.dropout is not None:
        #     if self.flag: utils.log_info(f'using dropout for aggregated distribution...')
        #     drop_p_attn = self.dropout(p_attn) #
        #     if self.reduction == 'att_sent':
        #         if self.flag: utils.log_info(f'using sentence dropout for aggregated distribution...')
        #         drop_p_attn = drop_p_attn.unsqueeze(1).repeat(1, value.size(1), 1, 1)
        #     self.flag = False
        if self.reduction == 'att_sent':
            p_attn = p_attn.unsqueeze(1).repeat(1, value.size(1), 1, 1)
        if mask is not None:
            mask = mask.unsqueeze(-1).cpu().int()
            self.total += torch.sum(mask).int()
            temp = torch.sum(p_attn.detach().cpu(), 3)
            temp *= mask
            self.summation += torch.sum(temp, (0, 1))

        return torch.matmul(value, p_attn)

class Assembled_MSource(nn.Module):
    def __init__(self, model, num_source, att_dropout=0.1, reduction='avg', lang=None):
        super(Assembled_MSource, self).__init__()

        self.num_source = num_source
        self.att_dropout = att_dropout
        self.source_models = clone(model, self.num_source)
        self.hidden_dim = model.hidden_dim

        self.num_labels = model.num_labels
        self.reduction = reduction
        if reduction.startswith('avg'):
            self.combine_layer = sum_()
            if reduction.endswith('equal'):
                print(f'Using model level average ({reduction}, notrainable) to aggregate multiple distribution...')
                self.weight_vector = torch.tensor([0.] * self.num_source, dtype=torch.float32, requires_grad=False).cuda()
            else:
                print(f'Using model level average ({reduction}, trainable) to aggregate multiple distribution...')
                self.weight_vector = nn.Parameter(torch.tensor([0.] * self.num_source, dtype=torch.float32), requires_grad=True)
        elif 'att' in reduction:
            if 'sent' in reduction:
                print('Using sentence ([CLS]) level attention to aggregate multiple distribution...')
            else:
                print('Using token level attention to aggregate multiple distribution...')
            # utils.log_info(f'Using bilinear attention? {use_bilinear}')
            self.combine_layer = MultiheadAttention(self.hidden_dim, dropout=att_dropout, num_source=num_source,
                                                    reduction=self.reduction)
            # self.combine_layer = nn.MultiheadAttention(self.num_labels, 1)

        self.nll_loss = nn.NLLLoss(reduction='sum')
        self.criterion = nn.Softmax(dim=-1)

    def load_source_models(self, source_models, device):
        for i in range(self.num_source):
            model_path = source_models[i]
            print(f'loading model {model_path}')
            model_config = torch.load(model_path, map_location=device)
            self.source_models[i].load_state_dict(model_config['model_state_dict'])
            self.source_models[i].model.fine_tune = False

    def crit(self, scores, labels, masks):
        # scores_mask = torch.eq(scores, float('0'))
        # scores = scores.masked_fill(scores_mask, 1e-32)

        scores = torch.log(scores)
        labels = labels.view(-1, )
        scores = scores.contiguous().view(-1, self.num_labels)
        mask_score = masks.contiguous().view(-1, 1)
        scores = scores.masked_select(mask_score.expand(-1, self.num_labels)).view(-1, self.num_labels)
        masks = masks.contiguous().view(-1, )
        labels = labels.masked_select(masks)
        loss = self.nll_loss(scores, labels)
        # loss = loss / self.batch_size
        return loss

    def forward(self, feats, query=None, sent_query=None, mask=None):
        b, l = query.size(0), query.size(1)
        source_outs, hiddens, sent_hiddens = [], [], []
        with torch.no_grad():
            for i in range(self.num_source):
                self.source_models[i].eval()
                score, hidden, sent_hidden = self.source_models[i](feats)
                hiddens.append(hidden)
                sent_hiddens.append(sent_hidden)
                source_outs.append(self.criterion(score))

        source_outs = torch.stack(source_outs, dim=-1)  # .view(-1, self.num_labels, self.num_source)
        if self.reduction.startswith('avg'):
            query = self.criterion(self.weight_vector)
            out = self.combine_layer(value=source_outs, query=query, key=hiddens).view(b, l, self.num_labels)
        elif 'sent' in self.reduction:
            out = self.combine_layer(value=source_outs, query=sent_query, key=sent_hiddens, mask=mask)
            out = out.view(b, l, self.num_labels)
        else:
            out = self.combine_layer(value=source_outs, query=query, key=hiddens, mask=mask)
            out = out.view(b, l, self.num_labels)
        return out

def KLDiv(p, q, mask, kl_func, **kwargs):
    nums = p.size(-1)
    mask = mask.contiguous().view(-1, 1).expand_as(p)
    p = p.masked_select(mask).view(-1, nums)

    q = q.masked_select(mask).view(-1, nums)
    return kl_func(torch.log(p), q)


def SymKLDiv(p, q, mask, kl_func, mu=0.5):
    nums = p.size(-1)
    mask = mask.contiguous().view(-1, 1).expand_as(p)
    p = p.masked_select(mask).view(-1, nums)
    q = q.masked_select(mask).view(-1, nums)

    return mu * kl_func(torch.log(p), q) + (1 - mu) * kl_func(torch.log(q), p)

class Assembled_MView(nn.Module):
    def __init__(self, model, pretrain_models, consensus='KL', reduce='att',
                 interpolation=0.5, view_interpolation=0.5, mu=0.5, att_dropout=0.1, lang=None, device='cpu'):
        super(Assembled_MView, self).__init__()
        self.device = device
        self.counts = Counter()
        if consensus == 'KL':
            print('Using KL divergence loss...')
            self.consensus = KLDiv
        elif consensus == 'SymKL':
            print(f'Using Symmetrical KL Div, setting mu: {mu}...')
            self.consensus = SymKLDiv

        self.source_models = Assembled_MSource(model, len(pretrain_models),
                                               att_dropout=att_dropout, reduction=reduce, lang=lang)
        self.load_source_models(pretrain_models, self.device)

        self.num_labels = model.num_labels
        self.interpolation = interpolation
        self.view_interpolation = view_interpolation
        self.mu = mu
        self.classification_model = copy.deepcopy(model)
        self.dropout = nn.Dropout(p=0.1)

        self.kld = nn.KLDivLoss(reduction='sum')  # torch.distributions.kl.kl_divergence#
        self.criterion = nn.Softmax(dim=-1)

    def load_source_models(self, pretrain_models, device):
        self.source_models.load_source_models(pretrain_models, device)

    def crit(self, out, source_out, labels, masks, if_labeled=False):
        out_ = self.criterion(out)
        loss_1, loss_2 = 0, 0
        interpolation = self.interpolation
        source_interpolation = self.view_interpolation

        loss_2 += self.consensus(out_.view(-1, self.num_labels),
                                 source_out.view(-1, self.num_labels), masks, self.kld, mu=self.mu)
        # if_labeled = False
        if if_labeled:
            loss_1 += interpolation * self.classification_model.crit(out, labels, masks)
            loss_1 += source_interpolation * self.source_models.crit(source_out, labels, masks)

        return loss_1 + loss_2

    def forward(self, feats, lengths, decode=True, warm_up=False, mask=None):
        out, hidden, sent_hidden = self.classification_model(feats, lengths)
        combined_source_out = self.source_models(feats, query=hidden.detach(),
                                                 sent_query=sent_hidden.detach(), mask=mask)
        return out, combined_source_out

    def decode(self, scores, masks=None, view='x', labels=None):
        out, combined_source_out = scores
        if view == 'x':
            return self.classification_model.decode(out, masks)
        elif view == 'source_model':
            return self.classification_model.decode(combined_source_out, masks)
