import logging
from typing import cast

import torch
from torch import FloatTensor, Tensor
from torch.nn.functional import cross_entropy
from torch.nn.parameter import Parameter
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)


class PTBertConfig(PretrainedConfig):
    model_type = "pt-bert"

    def __init__(
        self,
        pretrained_model: str = "bert",
        num_virtual_tokens: int = 1,
        num_labels: int = 1,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_model = pretrained_model
        self.num_virtual_tokens = num_virtual_tokens
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id


class PTBertSequenceClassification(PreTrainedModel):
    config_class = PTBertConfig

    def __init__(self, config: PTBertConfig) -> None:
        super().__init__(config)

        base_config = AutoConfig.from_pretrained(
            config.pretrained_model,
            num_labels=config.num_labels,
            id2label=config.id2label,
            label2id=config.label2id,
        )

        bert = AutoModelForSequenceClassification.from_config(base_config)
        hidden_size = bert.get_input_embeddings().embedding_dim

        # weights initialized from_pretrained or manually on first run
        self.bert = bert
        self.prefix = Parameter(torch.empty(1, config.num_virtual_tokens, hidden_size))

        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> SequenceClassifierOutput:
        batch_size = input_ids.size(0)

        prefix_emb = self.prefix.expand(batch_size, -1, -1)
        prefix_attn = torch.ones(
            input_ids.size(0),
            self.prefix.size(1),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        bert_emb = self.bert.get_input_embeddings().forward(input_ids)

        inputs = torch.cat([prefix_emb, bert_emb], dim=1)
        attn = torch.cat([prefix_attn, attention_mask], dim=1)

        out = self.bert(inputs_embeds=inputs, attention_mask=attn)
        out = cast(SequenceClassifierOutput, out)

        out_logits = cast(Tensor, out.logits)
        out["loss"] = cast(FloatTensor, cross_entropy(out_logits, labels))

        return out
