import logging
import os
from typing import Literal

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from icft.datasets.multinerd import Multinerd

logger = logging.getLogger(__name__)


def main(
    system_prompt: Literal["ner", "random", "none"],
    head_only: bool,
    pretrained_model: str,
    out_dir: str,
    epochs: int,
):
    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    multinerd = Multinerd(tokenizer=tokenizer, system_prompt=system_prompt)
    logger.info("train samples: %d", len(multinerd.train))
    logger.info("eval samples:  %d", len(multinerd.eval))
    logger.info("test samples:  %d", len(multinerd.test))

    logger.info("load %s", pretrained_model)
    bert, info = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=len(Multinerd.ID2LABEL),
        id2label=Multinerd.ID2LABEL,
        label2id=Multinerd.LABEL2ID,
        output_loading_info=True,
    )

    if head_only:
        logger.info("freeze base bert")
        logger.info("skip %s", info["missing_keys"])
        for name, param in bert.named_parameters():
            param.requires_grad = name in info["missing_keys"]

    total = sum(p.numel() for p in bert.parameters())
    trainable = sum(p.numel() for p in bert.parameters() if p.requires_grad)
    logger.info("total params:     %d", total)
    logger.info("trainable params: %d", trainable)

    logger.info("init trainer")
    os.environ["TENSORBOARD_LOGGING_DIR"] = out_dir
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
