import logging
import os
from typing import Literal

from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from icft.datasets.multinerd import Multinerd

logger = logging.getLogger(__name__)


def _freeze_model(model: PreTrainedModel, skip_layers: set[str]):
    logger.info("freeze base bert")
    logger.info("skip %s", skip_layers)
    for name, param in model.named_parameters():
        param.requires_grad = name in skip_layers


def _log_params(model: PreTrainedModel):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("total params:     %d", total_params)
    logger.info("trainable params: %d", trainable_params)


def main(
    system_prompt: Literal["ner", "random", "none"],
    head_only: bool,
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
        system_prompt=system_prompt,
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
    )

    logger.info("train samples: %d", len(multinerd.train))
    logger.info("eval samples:  %d", len(multinerd.eval))
    logger.info("test samples:  %d", len(multinerd.test))

    logger.info("load %s", pretrained_model)
    bert, info = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=len(Multinerd.ID2TAG),
        id2label=Multinerd.ID2TAG,
        label2id=Multinerd.TAG2ID,
        output_loading_info=True,
    )

    if head_only:
        _freeze_model(model=bert, skip_layers=info["unexpected_keys"])

    _log_params(model=bert)

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
        bert,
        args=args,
        train_dataset=multinerd.train,
        eval_dataset=multinerd.eval,
        data_collator=data_collator,
        compute_metrics=Multinerd.compute_metrics,
    )

    trainer.evaluate()
    trainer.train()
    trainer.evaluate(eval_dataset=multinerd.test, metric_key_prefix="test")
