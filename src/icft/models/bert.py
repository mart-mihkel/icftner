import logging
from typing import Annotated, cast

import torch
from torch import FloatTensor, Tensor
from torch.nn.functional import cross_entropy
from torch.nn.parameter import Parameter
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertForSequenceClassification

logger = logging.getLogger(__name__)


class PTBertConfig(PretrainedConfig):
    def __init__(
        self,
        bert: BertForSequenceClassification,
        prefix_embeds: Annotated[Tensor, "1 virtual emb"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bert = bert
        self.prefix_embeds = prefix_embeds


class PTBert(PreTrainedModel):
    def __init__(self, config: PTBertConfig) -> None:
        super().__init__(config)
        self.bert = config.bert
        self.prefix_embeds = Parameter(config.prefix_embeds)
        freeze_bert(self.bert, freeze_head=False)

    def forward(
        self,
        input_ids: Annotated[Tensor, "batch seq"],
        attention_mask: Annotated[Tensor, "batch seq"],
        labels: Annotated[Tensor, "batch"],
    ) -> SequenceClassifierOutput:
        batch_size = input_ids.size(0)
        prefix_embeds: Annotated[Tensor, "batch virtual emb"] = (
            self.prefix_embeds.expand(batch_size, -1, -1)
        )
        prefix_attention: Annotated[Tensor, "batch virtual"] = torch.ones(
            input_ids.size(0),
            self.prefix_embeds.size(1),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        bert_embeds: Annotated[Tensor, "batch seq emb"] = (
            self.bert.get_input_embeddings().forward(input_ids)
        )

        inputs_embeds = torch.cat([prefix_embeds, bert_embeds], dim=1)
        attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        out = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        out = cast(SequenceClassifierOutput, out)

        out_logits = cast(Tensor, out.logits)
        out["loss"] = cast(FloatTensor, cross_entropy(out_logits, labels))

        return out


def freeze_bert(bert: BertForSequenceClassification, freeze_head: bool):
    logger.info("freeze bert parameters")

    head_params = [
        "classifier.bias",
        "classifier.weight",
        "pre_classifier.bias",
        "pre_classifier.weight",
    ]

    for name, param in bert.named_parameters():
        if not freeze_head and name in head_params:
            logger.info("skip %s", name)
            continue

        param.requires_grad = False
