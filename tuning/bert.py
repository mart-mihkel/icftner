from typing import cast

import torch
from datasets import DatasetDict
from peft import PeftModel, PromptTuningConfig, TaskType, get_peft_model
from transformers import (
    BertForMultipleChoice,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForMultipleChoice,
)

from .util.datasets import prep_swag


def main(
    pretrained_model_name: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    prompt_tune: bool,
    num_virtual_tokens: int,
):
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
    tokenizer = cast(BertTokenizerFast, tokenizer)

    dataset = prep_swag(tokenizer)

    model = BertForMultipleChoice.from_pretrained(pretrained_model_name)

    if prompt_tune:
        model = _get_peft_bert(model, pretrained_model_name, num_virtual_tokens)

    _tune(
        model,
        tokenizer,
        dataset,
        output_dir,
        num_epochs,
        batch_size,
    )


def _get_peft_bert(
    bert: BertForMultipleChoice,
    pretrained_tokenizer_name: str,
    num_virtual_tokens: int,
) -> PeftModel:
    cfg = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=num_virtual_tokens,
        tokenizer_name_or_path=pretrained_tokenizer_name,
    )

    peft_bert = get_peft_model(bert, cfg)
    peft_bert = cast(PeftModel, peft_bert)

    return peft_bert


def _tune(
    model: torch.nn.Module,
    tokenizer: BertTokenizerFast,
    dataset: DatasetDict,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
):
    collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir="logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
