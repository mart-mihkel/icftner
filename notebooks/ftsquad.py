import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")

with app.setup:
    import logging
    from pathlib import Path

    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    from cptlms.squad import Squad
    from cptlms.trainer import Trainer

    logging.basicConfig(level="INFO")
    torch.set_float32_matmul_precision("high")


@app.cell
def _():
    pretrained_model = "jhu-clsp/mmBERT-small"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    bert = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)
    squad = Squad(tokenizer)
    return bert, squad


@app.cell
def _(bert, squad):
    trainer = Trainer(
        model=bert,
        epochs=5,
        batch_size=32,
        qa_dataset=squad,
        collate_fn=squad.default_collate_fn,
        out_dir=Path("out/mmbert-small-ft"),
    )
    return (trainer,)


@app.cell
def _(trainer):
    trainer.train()
    return


if __name__ == "__main__":
    app.run()
