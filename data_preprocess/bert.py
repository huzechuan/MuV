from pathlib import Path
from typing import List, Union
import os
import re
from transformers import XLNetTokenizer, T5Tokenizer, GPT2Tokenizer, AutoTokenizer, AutoConfig, AutoModel
import torch
from .utils import log_info


class TransformerWordEmbeddings:
    def __init__(
        self,
        model: str = "bert-base-uncased",
        layers: str = "-1,-2,-3,-4",
        pooling_operation: str = "first",
        batch_size: int = 1,
        use_scalar_mix: bool = False,
        fine_tune: bool = False,
        device: str = 'cpu'
    ):
        """
        Bidirectional transformer embeddings of words from various transformer architectures.
        :param model: name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for
        options)
        :param layers: string indicating which layers to take for embedding (-1 is topmost layer)
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either take the first
        subtoken ('first'), the last subtoken ('last'), both first and last ('first_last') or a mean over all ('mean')
        :param batch_size: How many sentence to push through transformer at once. Set to 1 by default since transformer
        models tend to be huge.
        :param use_scalar_mix: If True, uses a scalar mix of layers as embedding
        :param fine_tune: If True, allows transformers to be fine-tuned during training
        """
        super().__init__()
        self.device = device

        # load tokenizer and transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model, config=config)

        # model name
        self.name = 'transformer-word-' + str(model)

        # when initializing, embeddings are in eval mode by default
        self.model.eval()
        self.model.to(device)

        # embedding parameters
        if layers == 'all':
            # send mini-token through to check how many layers the model has
            hidden_states = self.model(torch.tensor([1], device=device).unsqueeze(0))[-1]
            self.layer_indexes = [int(x) for x in range(len(hidden_states))]
        else:
            self.layer_indexes = [int(x) for x in layers.split(",")]
        # self.mix = ScalarMix(mixture_size=len(self.layer_indexes), trainable=False)
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.fine_tune = fine_tune
        self.static_embeddings = not self.fine_tune
        self.batch_size = batch_size

        self.special_tokens = []
        # check if special tokens exist to circumvent error message
        if self.tokenizer._bos_token:
            self.special_tokens.append(self.tokenizer.bos_token)
        if self.tokenizer._cls_token:
            self.special_tokens.append(self.tokenizer.cls_token)

        # most models have an intial BOS token, except for XLNet, T5 and GPT2
        self.begin_offset = 1
        if type(self.tokenizer) == XLNetTokenizer:
            self.begin_offset = 0
        if type(self.tokenizer) == T5Tokenizer:
            self.begin_offset = 0
        if type(self.tokenizer) == GPT2Tokenizer:
            self.begin_offset = 0

    def _add_embeddings_internal(self, sentences: List) -> List:
        """Add embeddings to all words in a list of sentences."""

        # split into micro batches of size self.batch_size before pushing through transformer
        sentence_batches = [sentences[i * self.batch_size:(i + 1) * self.batch_size]
                            for i in range((len(sentences) + self.batch_size - 1) // self.batch_size)]

        # embed each micro-batch
        for batch in sentence_batches:
            self._add_embeddings_to_sentences(batch)

        return sentences

    @staticmethod
    def _remove_special_markup(text: str):
        # remove special markup
        text = re.sub('^Ġ', '', text)  # RoBERTa models
        text = re.sub('^##', '', text)  # BERT models
        text = re.sub('^▁', '', text)  # XLNet models
        text = re.sub('</w>$', '', text)  # XLM models
        return text

    def _get_processed_token_text(self, token) -> str:
        pieces = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(token.text, add_special_tokens=False))
        token_text = ''
        for piece in pieces:
            token_text += self._remove_special_markup(piece)
        token_text = token_text.lower()
        return token_text

    def _add_embeddings_to_sentences(self, sentences: List):
        """Match subtokenization to Flair tokenization and extract embeddings from transformers for each token."""

        # first, subtokenize each sentence and find out into how many subtokens each token was divided
        subtokenized_sentences = []
        subtokenized_sentences_token_lengths = []

        for sentence in sentences:

            tokenized_string = sentence.to_tokenized_string()

            # method 1: subtokenize sentence
            # subtokenized_sentence = self.tokenizer.encode(tokenized_string, add_special_tokens=True)

            # method 2:
            ids = self.tokenizer.encode(tokenized_string, add_special_tokens=False)
            subtokenized_sentence = self.tokenizer.build_inputs_with_special_tokens(ids)

            subtokenized_sentences.append(torch.tensor(subtokenized_sentence, dtype=torch.long))
            subtokens = self.tokenizer.convert_ids_to_tokens(subtokenized_sentence)
            # print(subtokens)

            word_iterator = iter(sentence)
            token = next(word_iterator)
            token_text = self._get_processed_token_text(token)

            token_subtoken_lengths = []
            reconstructed_token = ''
            subtoken_count = 0

            # iterate over subtokens and reconstruct tokens
            for subtoken_id, subtoken in enumerate(subtokens):

                subtoken_count += 1

                # remove special markup
                subtoken = self._remove_special_markup(subtoken)

                # append subtoken to reconstruct token
                reconstructed_token = reconstructed_token + subtoken

                # check if reconstructed token is special begin token ([CLS] or similar)
                if reconstructed_token in self.special_tokens and subtoken_id == 0:
                    reconstructed_token = ''
                    subtoken_count = 0

                # check if reconstructed token is the same as current token
                if reconstructed_token.lower() == token_text:

                    # if so, add subtoken count
                    token_subtoken_lengths.append(subtoken_count)

                    # reset subtoken count and reconstructed token
                    reconstructed_token = ''
                    subtoken_count = 0

                    # break from loop if all tokens are accounted for
                    if len(token_subtoken_lengths) < len(sentence):
                        token = next(word_iterator)
                        token_text = self._get_processed_token_text(token)
                    else:
                        break

            # check if all tokens were matched to subtokens
            if token != sentence[-1]:
                log_info(f"Tokenization MISMATCH in sentence '{sentence.to_tokenized_string()}'")
                log_info(f"Last matched: '{token}'")
                log_info(f"Last sentence: '{sentence[-1]}'")
                log_info(f"subtokenized: '{subtokens}'")

            subtokenized_sentences_token_lengths.append(token_subtoken_lengths)

        # find longest sentence in batch
        longest_sequence_in_batch: int = len(max(subtokenized_sentences, key=len))

        # initialize batch tensors and mask
        input_ids = torch.zeros(
            [len(sentences), longest_sequence_in_batch],
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.zeros(
            [len(sentences), longest_sequence_in_batch],
            dtype=torch.long,
            device=self.device,
        )
        for s_id, sentence in enumerate(subtokenized_sentences):
            sequence_length = len(sentence)
            input_ids[s_id][:sequence_length] = sentence
            mask[s_id][:sequence_length] = torch.ones(sequence_length)

        # put encoded batch through transformer model to get all hidden states of all encoder layers
        hidden_states = self.model(input_ids, attention_mask=mask)[-1]

        # gradients are enabled if fine-tuning is enabled
        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()

        with gradient_context:

            # iterate over all subtokenized sentences
            for sentence_idx, (sentence, subtoken_lengths) in enumerate(zip(sentences, subtokenized_sentences_token_lengths)):

                subword_start_idx = self.begin_offset

                # for each token, get embedding
                for token_idx, (token, number_of_subtokens) in enumerate(zip(sentence, subtoken_lengths)):

                    subword_end_idx = subword_start_idx + number_of_subtokens

                    subtoken_embeddings: List[torch.FloatTensor] = []

                    # get states from all selected layers, aggregate with pooling operation
                    for layer in self.layer_indexes:
                        current_embeddings = hidden_states[layer][sentence_idx][subword_start_idx:subword_end_idx]

                        if self.pooling_operation == "first":
                            final_embedding: torch.FloatTensor = current_embeddings[0]

                        if self.pooling_operation == "last":
                            final_embedding: torch.FloatTensor = current_embeddings[-1]

                        if self.pooling_operation == "first_last":
                            final_embedding: torch.Tensor = torch.cat([current_embeddings[0], current_embeddings[-1]])

                        if self.pooling_operation == "mean":
                            all_embeddings: List[torch.FloatTensor] = [
                                embedding.unsqueeze(0) for embedding in current_embeddings
                            ]
                            final_embedding: torch.Tensor = torch.mean(torch.cat(all_embeddings, dim=0), dim=0)

                        subtoken_embeddings.append(final_embedding)

                    # use scalar mix of embeddings if so selected
                    if self.use_scalar_mix:
                        sm_embeddings = torch.mean(torch.stack(subtoken_embeddings, dim=1), dim=1)
                        # sm_embeddings = self.mix(subtoken_embeddings)

                        subtoken_embeddings = [sm_embeddings]

                    # set the extracted embedding for the token
                    token.set_embedding(self.name, torch.cat(subtoken_embeddings))

                    subword_start_idx += number_of_subtokens

    def train(self, mode=True):
        # if fine-tuning is not enabled (i.e. a "feature-based approach" used), this
        # module should never be in training mode
        if not self.fine_tune:
            pass
        else:
            super().train(mode)

    @property
    # @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""

        if not self.use_scalar_mix:
            length = len(self.layer_indexes) * self.model.config.hidden_size
        else:
            length = self.model.config.hidden_size

        if self.pooling_operation == 'first_last': length *= 2

        return length

    def __setstate__(self, d):
        self.__dict__ = d

        # reload tokenizer to get around serialization issues
        model_name = self.name.split('transformer-word-')[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)