import marimo

__generated_with = "0.19.4"
app = marimo.App()

with app.setup:
    from typing import cast, Annotated

    from torch import Tensor
    from transformers import AutoTokenizer
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
    from transformers.models.bert.modeling_bert import (
        BertForSequenceClassification,
    )


@app.function
def encode_prefix(
    prefix: str,
    model: BertForSequenceClassification,
    tokenizer: PreTrainedTokenizerFast,
) -> Annotated[Tensor, "prefix emb"]:
    prefix_tokenized = tokenizer(prefix, return_tensors="pt")
    token_ids = prefix_tokenized["input_ids"][0]
    return model.get_input_embeddings().forward(token_ids)


@app.function
def decode_prefix(
    prefix_embeddings: Annotated[Tensor, "prefix emb"],
    model: BertForSequenceClassification,
    tokenizer: PreTrainedTokenizerFast,
) -> str:
    voc_embeddings = cast(Tensor, model.get_input_embeddings().weight)
    similarity = prefix_embeddings @ voc_embeddings.T
    token_ids = similarity.argmax(dim=1)
    return tokenizer.decode(token_ids=token_ids)


@app.cell
def _():
    _pretrained_model = "boltuix/bert-micro"
    tokenizer = AutoTokenizer.from_pretrained(_pretrained_model)
    model = BertForSequenceClassification.from_pretrained(_pretrained_model)
    return model, tokenizer


@app.cell
def _(model, tokenizer):
    prefix_embeddings = encode_prefix(
        prefix="What time is it?",
        model=model,
        tokenizer=tokenizer,
    )

    decode_prefix(
        prefix_embeddings=prefix_embeddings,
        model=model,
        tokenizer=tokenizer,
    )
    return


if __name__ == "__main__":
    app.run()
