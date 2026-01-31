import logging
import os

import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from icft.datasets.multinerd import Multinerd
from icft.models.bert import PTBert, PTBertConfig

logger = logging.getLogger(__name__)


def _log_params(model: PreTrainedModel):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("total params:     %d", total_params)
    logger.info("trainable params: %d", trainable_params)


def _setup_ptbert(
    base_model: str,
    prefix_random_init: bool,
    tokenizer: PreTrainedTokenizer,
) -> PTBert:
    bert = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(Multinerd.ID2TAG),
        id2label=Multinerd.ID2TAG,
        label2id=Multinerd.TAG2ID,
    )

    logger.info("init pt-bert")
    system_prompt = tokenizer(Multinerd.SYSTEM_PROMPT, return_tensors="pt")
    bert_emb = bert.get_input_embeddings()
    num_virtual_tokens = len(system_prompt["input_ids"])

    if prefix_random_init:
        prefix_embeds = torch.randn(1, num_virtual_tokens, bert_emb.embedding_dim)
    else:
        prefix_embeds = bert_emb.forward(system_prompt["input_ids"])

    conf = PTBertConfig(bert=bert, prefix_embeds=prefix_embeds)
    return PTBert(conf)


def main(
    prefix_random_init: bool,
    pretrained_model: str,
    out_dir: str,
    epochs: int,
    train_split: str,
    eval_split: str,
    test_split: str,
):
    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    multinerd = Multinerd(
        tokenizer=tokenizer,
        system_prompt="none",
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
    )

    logger.info("train samples: %d", len(multinerd.train))
    logger.info("eval samples:  %d", len(multinerd.eval))
    logger.info("test samples:  %d", len(multinerd.test))

    logger.info("load %s", pretrained_model)
    if "checkpoint" in pretrained_model:
        pt_bert = PTBert.from_pretrained(pretrained_model)
    else:
        pt_bert = _setup_ptbert(
            base_model=pretrained_model,
            prefix_random_init=prefix_random_init,
            tokenizer=tokenizer,
        )

    _log_params(model=pt_bert)

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
