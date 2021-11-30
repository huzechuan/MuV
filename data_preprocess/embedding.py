import re
from transformers import (XLNetTokenizer, T5Tokenizer,
                          GPT2Tokenizer, AutoTokenizer, AutoConfig,
                          AutoModel, AutoModelForTokenClassification, XLMRobertaModel)
from torch.nn.utils.rnn import pad_sequence
import torch


class TransfomerFeatures(torch.nn.Module):
    def __init__(self,
                 model: str = "bert-base--multilingual-cased",
                 fine_tune: bool = True,
                 device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        config = AutoConfig.from_pretrained(model)
        self.config = config
        self.model = AutoModel.from_pretrained(model, config=config)
        self.fine_tune = fine_tune
        # model name
        self.name = str(model)

        # when initializing, embeddings are in eval mode by default
        self.model.eval()
        self.model.to(device)

    @property
    # @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""

        length = self.model.config.hidden_size
        return length

    def _remove_special_token(self, sub):
        text = re.sub('^##', '', sub)  # BERT models
        return text

    def __first_subword(self, subword, sentence):
        # for mBERT
        first_subword_pos = []
        w_id = 1
        if self.name.startswith('bert'):
            for s_id, word in enumerate(sentence):
                pieces = self.tokenizer.encode(word, add_special_tokens=False)
                sub_l = len(pieces)
                first_subword_pos.append(w_id)
                w_id += sub_l

        # for XLM-R
        else:
            for idx, sub_w in enumerate(subword):
                if sub_w.startswith('▁'):
                    first_subword_pos.append(idx)
        return first_subword_pos

    def forward(self, sentences):
        if isinstance(sentences, tuple):
            # return sentences
            sentences_obj, sentences = sentences
        elif isinstance(sentences, list):
            sentences_obj, sentences = None, sentences
        else:
            return sentences, None
        ids = []
        subwords = []
        for sent in sentences:
            sentence = ' '.join(sent)
            input_id = self.tokenizer.encode(sentence)
            ids.append(input_id)
            subword = self.tokenizer.convert_ids_to_tokens(input_id)
            subwords.append(self.__first_subword(subword, sent))

        max_len = len(max(ids, key=len))
        mask = torch.zeros(
            [len(ids), max_len],
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )
        input_ids = torch.zeros(
            [len(ids), max_len],
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )
        for idx, sent in enumerate(ids):
            length = len(sent)
            input_ids[idx][:length] = torch.tensor(sent, dtype=torch.long)
            mask[idx][:length] = torch.ones(length)

        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()

        with gradient_context:
            scores = self.model(input_ids, attention_mask=mask)[0]
            features = []
            sent_features = scores[:, 0, :]

            for sentence_idx, first_id in enumerate(subwords):
                select_index = torch.tensor(first_id, dtype=torch.long, device=self.device)
                # get states from the last selected layers, choose the first subword representation.
                sent_states = torch.index_select(scores[sentence_idx], 0, select_index)
                features.append(sent_states)
            scores = pad_sequence(features, padding_value=0.0, batch_first=True)

        return scores, sent_features#.transpose(0, 1)#, mask_t.t_()
