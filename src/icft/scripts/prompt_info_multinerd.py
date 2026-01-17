import logging

from transformers import AutoTokenizer

from icft.datasets.multinerd import Multinerd, _random_tokens

logger = logging.getLogger(__name__)


def main(pretrained_model: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    logger.info("model            %s", pretrained_model)

    system_tokenized = tokenizer(Multinerd._SYSTEM_TOKENS, is_split_into_words=True)
    system_ids = system_tokenized["input_ids"]
    logger.info("system length    %d", len(Multinerd._SYSTEM_TOKENS))
    logger.info("system tokens    %d", len(system_ids))

    gibberish = _random_tokens(len(Multinerd._SYSTEM_TOKENS))
    gibberish_tokenized = tokenizer(gibberish, is_split_into_words=True)
    gibberish_ids = gibberish_tokenized["input_ids"]
    logger.info("gibberish length %d", len(gibberish))
    logger.info("gibberish tokens %d", len(gibberish_ids))
