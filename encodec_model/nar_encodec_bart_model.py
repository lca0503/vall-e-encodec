import math
from typing import Optional

import torch
from torch import nn
from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import (BartDecoder,
                                                    BartDecoderLayer,
                                                    BartEncoder,
                                                    BartLearnedPositionalEmbedding,
                                                    BartModel, _expand_mask)

from .encodec_bart_model import BartEncodecEncoder
from .nar_bart_model import NARBartDecoder


class NARBartEncodecModel(BartModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # Modified: BartEncoder() to BartEncodecEncoder()
        self.encoder = BartEncodecEncoder(config, self.shared)
        # Modified: BartDecoder() to NARBartDecoder()
        self.decoder = NARBartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()


class NARBartEncodecForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        # Modified: BartModel() to NARBartEncodecModel()
        self.model = NARBartEncodecModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
