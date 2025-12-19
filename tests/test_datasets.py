from cptlms.datasets.multinerd import _prepare_prompt_bert


def test_prepare_prompt_bert():
    target = "[CLS] Find the NER tags . [SEP] fish [SEP] How much is the fish ? [SEP]"
    prompt_tokens = _prepare_prompt_bert(
        target_token="fish",
        tokens=["How", "much", "is", "the", "fish", "?"],
        system_tokens=["Find", "the", "NER", "tags", "."],
    )

    assert target == " ".join(prompt_tokens)
