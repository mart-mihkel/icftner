from icftner.datasets.multinerd import _prepare_prompt_bert


def test_prepare_prompt_bert():
    target1 = "[CLS] Find the NER tags . [SEP] fish [SEP] How much is the fish ? [SEP]"
    prompt_tokens1 = _prepare_prompt_bert(
        target_token="fish",
        tokens=["How", "much", "is", "the", "fish", "?"],
        system_tokens=["Find", "the", "NER", "tags", "."],
    )

    target2 = "[CLS] fish [SEP] How much is the fish ? [SEP]"
    prompt_tokens2 = _prepare_prompt_bert(
        target_token="fish",
        tokens=["How", "much", "is", "the", "fish", "?"],
        system_tokens=[],
    )

    assert target1 == " ".join(prompt_tokens1)
    assert target2 == " ".join(prompt_tokens2)
