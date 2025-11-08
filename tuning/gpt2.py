from typing import cast
from datasets import load_dataset, DatasetDict
from peft import PrefixTuningConfig, TaskType, get_peft_model
from transformers.models.gpt2 import GPT2TokenizerFast, GPT2LMHeadModel
from transformers import (
    TrainingArguments,
    Trainer,
)


def main(pretrained_model: str, epochs: int):
    tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = cast(
        DatasetDict,
        load_dataset("wikitext", "wikitext-2-raw-v1")
        .map(
            lambda x: tokenizer(
                x["text"],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            ),
            remove_columns=["text"],
            batched=True,
        )
        .map(
            lambda x: {"input_ids": x["input_ids"], "labels": x["input_ids"]},
            batched=True,
        ),
    )

    peft_cfg = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=64,
    )

    model = GPT2LMHeadModel.from_pretrained(pretrained_model)
    model.resize_token_embeddings(len(tokenizer))
    model = get_peft_model(model, peft_cfg)

    train_args = TrainingArguments(
        output_dir="./trainer-out/gpt2-prefix-tuning",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        eval_dataset=dataset["test"].select(range(2500)),
        train_dataset=dataset["train"],
    )

    trainer.train()
