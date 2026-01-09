import logging

from transformers import AutoTokenizer

from icftner.datasets.multinerd import MULTINERD_SYSTEM_PROMPT, random_gibberish

logger = logging.getLogger(__name__)


def main(pretrained_model: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    logger.info("model            %s", pretrained_model)

    system_tokenized = tokenizer(MULTINERD_SYSTEM_PROMPT, is_split_into_words=True)
    system_ids = system_tokenized["input_ids"]
    logger.info("system length    %d", len(MULTINERD_SYSTEM_PROMPT))
    logger.info("system tokens    %d", len(system_ids))

    gibberish = random_gibberish(len(MULTINERD_SYSTEM_PROMPT))
    gibberish_tokenized = tokenizer(gibberish, is_split_into_words=True)
    gibberish_ids = gibberish_tokenized["input_ids"]
    logger.info("gibberish length %d", len(gibberish))
    logger.info("gibberish tokens %d", len(gibberish_ids))
