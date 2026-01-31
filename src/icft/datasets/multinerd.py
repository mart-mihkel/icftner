import logging
from typing import Literal, TypedDict, cast

import numpy as np
from datasets.load import load_dataset
from datasets.utils.info_utils import VerificationMode
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BatchEncoding, EvalPrediction, PreTrainedTokenizer

logger = logging.getLogger(__name__)

type MultinerdLang = Literal[
    "zh",
    "nl",
    "en",
    "fr",
    "de",
    "it",
    "pl",
    "pt",
    "ru",
    "es",
]

type MultinerdTag = Literal[
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-ANIM",
    "I-ANIM",
    "B-BIO",
    "I-BIO",
    "B-CEL",
    "I-CEL",
    "B-DIS",
    "I-DIS",
    "B-EVE",
    "I-EVE",
    "B-FOOD",
    "I-FOOD",
    "B-INST",
    "I-INST",
    "B-MEDIA",
    "I-MEDIA",
    "B-MYTH",
    "I-MYTH",
    "B-PLANT",
    "I-PLANT",
    "B-TIME",
    "I-TIME",
    "B-VEHI",
    "I-VEHI",
]


class MultinerdMetrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1: float


class MultinerdBatch(TypedDict):
    tokens: list[list[str]]
    ner_tags: list[list[MultinerdTag]]
    lang: list[MultinerdLang]


class Multinerd:
    TAG2ID: dict[MultinerdTag, int] = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-ANIM": 7,
        "I-ANIM": 8,
        "B-BIO": 9,
        "I-BIO": 10,
        "B-CEL": 11,
        "I-CEL": 12,
        "B-DIS": 13,
        "I-DIS": 14,
        "B-EVE": 15,
        "I-EVE": 16,
        "B-FOOD": 17,
        "I-FOOD": 18,
        "B-INST": 19,
        "I-INST": 20,
        "B-MEDIA": 21,
        "I-MEDIA": 22,
        "B-MYTH": 23,
        "I-MYTH": 24,
        "B-PLANT": 25,
        "I-PLANT": 26,
        "B-TIME": 27,
        "I-TIME": 28,
        "B-VEHI": 29,
        "I-VEHI": 30,
    }

    ID2TAG: dict[int, MultinerdTag] = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-ANIM",
        8: "I-ANIM",
        9: "B-BIO",
        10: "I-BIO",
        11: "B-CEL",
        12: "I-CEL",
        13: "B-DIS",
        14: "I-DIS",
        15: "B-EVE",
        16: "I-EVE",
        17: "B-FOOD",
        18: "I-FOOD",
        19: "B-INST",
        20: "I-INST",
        21: "B-MEDIA",
        22: "I-MEDIA",
        23: "B-MYTH",
        24: "I-MYTH",
        25: "B-PLANT",
        26: "I-PLANT",
        27: "B-TIME",
        28: "I-TIME",
        29: "B-VEHI",
        30: "I-VEHI",
    }

    SYSTEM_PROMPT = """You are a named entity recognition model.
Given a target word and a sentence containing that word, predict the NER tag of the target word based on its context.

Example
Word: Paris
Sentence: Paris is the capital of France.
Output: B-LOC
"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        system_prompt: Literal["ner", "random", "none"],
        train_split: str,
        eval_split: str,
        test_split: str,
    ) -> None:
        logger.info("load multinerd")
        train, eval, test = load_dataset(
            "Babelscape/multinerd",
            split=[train_split, eval_split, test_split],
            verification_mode=VerificationMode.NO_CHECKS,
        )

        logger.info("filter english")
        train = train.filter(self._filter_english, batched=True)
        eval = eval.filter(self._filter_english, batched=True)
        test = test.filter(self._filter_english, batched=True)

        logger.info("prepare system prompt")
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.special_tokens_map["sep_token"]
        self.cls_token = tokenizer.special_tokens_map["cls_token"]

        if system_prompt == "ner":
            self.system_tokens = self.tokenizer(
                f"{self.cls_token} {self.SYSTEM_PROMPT} {self.sep_token}",
                add_special_tokens=False,
            )
        elif system_prompt == "random":
            self.system_tokens = self._randomize_system_prompt()
        else:
            self.system_tokens = self.tokenizer(
                self.cls_token,
                add_special_tokens=False,
            )

        logger.info("tokenize multinerd")
        self.train = train.map(
            self._tokenize,
            batched=True,
            remove_columns=train.column_names,
        )

        self.eval = eval.map(
            self._tokenize,
            batched=True,
            remove_columns=eval.column_names,
        )

        self.test = test.map(
            self._tokenize,
            batched=True,
            remove_columns=test.column_names,
        )

    def _randomize_system_prompt(self) -> BatchEncoding:
        tokens = self.tokenizer(self.SYSTEM_PROMPT, add_special_tokens=False)

        ids = cast(list[int], tokens["input_ids"])
        attn = cast(list[int], tokens["attention_mask"])
        vocab_size = self.tokenizer.vocab_size
        random_ids = np.random.randint(0, vocab_size - 1, size=len(ids))

        cls_id = self.tokenizer.convert_tokens_to_ids(self.cls_token)
        sep_id = self.tokenizer.convert_tokens_to_ids(self.sep_token)

        input_ids = [cls_id] + random_ids.tolist() + [sep_id]
        attention_mask = [1] + attn + [1]

        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})

    def _tokenize(self, batch: MultinerdBatch) -> BatchEncoding:
        labels: list[MultinerdTag] = []
        prompts: list[list[str]] = []
        for tokens, tags in zip(batch["tokens"], batch["ner_tags"]):
            for token, tag in zip(tokens, tags):
                prompt = [token, self.sep_token] + tokens
                prompts.append(prompt)
                labels.append(tag)

        tokens = self.tokenizer(
            prompts,
            is_split_into_words=True,
            add_special_tokens=False,
            return_token_type_ids=False,
        )

        sys_ids = cast(list[int], self.system_tokens["input_ids"])
        sys_attn = cast(list[int], self.system_tokens["attention_mask"])

        ids = cast(list[list[int]], tokens["input_ids"])
        attn = cast(list[list[int]], tokens["attention_mask"])

        tokens["labels"] = labels
        tokens["input_ids"] = [sys_ids + prompt_ids for prompt_ids in ids]
        tokens["attention_mask"] = [sys_attn + prompt_attn for prompt_attn in attn]

        return tokens

    @staticmethod
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = MultinerdMetrics(
            accuracy=accuracy_score(labels, preds),
            precision=precision_score(labels, preds, average="macro"),
            recall=recall_score(labels, preds, average="macro"),
            f1=f1_score(labels, preds, average="macro"),
        )

        return cast(dict[str, float], metrics)

    @staticmethod
    def _filter_english(batch: MultinerdBatch) -> list[bool]:
        return [lang == "en" for lang in batch["lang"]]
