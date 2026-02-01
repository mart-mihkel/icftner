import logging
import os
from typing import cast

import torch
from torch.nn import Parameter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from icft.datasets.multinerd import Multinerd
from icft.models.bert import PTBertConfig, PTBertSequenceClassification

logger = logging.getLogger(__name__)


def main(
    prefix_random_init: bool,
    pretrained_model: str,
    out_dir: str,
    epochs: int,
):
    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    multinerd = Multinerd(tokenizer=tokenizer, system_prompt="none")
    logger.info("train samples: %d", len(multinerd.train))
    logger.info("eval samples:  %d", len(multinerd.eval))
    logger.info("test samples:  %d", len(multinerd.test))

    logger.info("load %s", pretrained_model)
    if "checkpoint" in pretrained_model:
        config = PTBertConfig.from_pretrained(pretrained_model)
        pt_bert = PTBertSequenceClassification.from_pretrained(
            pretrained_model,
            config=config,
        )
    else:
        bert, info = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=len(Multinerd.ID2LABEL),
            id2label=cast(dict[int, str], Multinerd.ID2LABEL),
            label2id=cast(dict[str, int], Multinerd.LABEL2ID),
            output_loading_info=True,
        )

        system_tokens = tokenizer(Multinerd.SYSTEM_PROMPT, return_tensors="pt")
        system_ids = system_tokens["input_ids"]

        logger.info("init pt-bert")
        config = PTBertConfig(
            pretrained_model=pretrained_model,
            num_virtual_tokens=system_ids.size(1),
            num_labels=len(Multinerd.ID2LABEL),
            id2label=cast(dict[int, str], Multinerd.ID2LABEL),
            label2id=cast(dict[str, int], Multinerd.LABEL2ID),
        )

        pt_bert = PTBertSequenceClassification(config=config)

        logger.info("freeze base bert")
        logger.info("skip %s", info["missing_keys"])
        for name, param in pt_bert.bert.named_parameters():
            param.requires_grad = name in info["missing_keys"]

        logger.info("init prefix")
        bert_emb = bert.get_input_embeddings()
        if prefix_random_init:
            prefix = torch.randn(1, system_ids.size(1), bert_emb.embedding_dim)
        else:
            prefix = bert_emb(system_ids).detach()

        pt_bert.bert.load_state_dict(bert.state_dict(), strict=False)
        pt_bert.prefix = Parameter(prefix)

    total = sum(p.numel() for p in pt_bert.parameters())
    trainable = sum(p.numel() for p in pt_bert.parameters() if p.requires_grad)
    logger.info("total params:     %d", total)
    logger.info("trainable params: %d", trainable)

    logger.info("init trainer")
    os.environ["TENSORBOARD_LOGGING_DIR"] = f"{out_dir}/tensorboard"
    args = TrainingArguments(
        output_dir=out_dir,
        report_to="tensorboard",
        num_train_epochs=epochs,
        logging_steps=5000,
        logging_first_step=True,
        eval_steps=50000,
        eval_strategy="steps",
        save_strategy="epoch",
        auto_find_batch_size=True,
        remove_unused_columns=False,
        fp16=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        pt_bert,
        args=args,
        train_dataset=multinerd.train,
        eval_dataset=multinerd.eval,
        data_collator=data_collator,
        compute_metrics=Multinerd.compute_metrics,
    )

    trainer.evaluate()
    trainer.train()
    trainer.evaluate(eval_dataset=multinerd.test, metric_key_prefix="test")
