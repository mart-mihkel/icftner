#!/usr/bin/env bash
#SBATCH --output=out/slurm/icft-%j.out

set -euo pipefail
mkdir -p out

BASE_MODEL=distilbert/distilbert-base-uncased
# BASE_MODEL=jhu-clsp/mmBERT-base

MODEL_NAME=$(echo $BASE_MODEL | tr / -)

nix-shell --argstr run "make install"

# full fine tune
nix-shell --argstr run "uv run cli.py fine-tune-bert-multinerd \
    --pretrained-model $BASE_MODEL \
    --out-dir out/ft-full-multinerd-eng-$MODEL_NAME"

nix-shell --argstr run "uv run cli.py fine-tune-bert-multinerd \
    --pretrained-model $BASE_MODEL \
    --out-dir out/ft-full-prompted-multinerd-eng-$MODEL_NAME \
    --system-prompt ner"

nix-shell --argstr run "uv run cli.py fine-tune-bert-multinerd \
    --pretrained-model $BASE_MODEL \
    --out-dir out/ft-full-gibberish-multinerd-eng-$MODEL_NAME \
    --system-prompt random"

# head fine tune
nix-shell --argstr run "uv run cli.py fine-tune-bert-multinerd \
    --pretrained-model $BASE_MODEL \
    --out-dir out/ft-head-multinerd-eng-$MODEL_NAME \
    --head-only"

nix-shell --argstr run "uv run cli.py fine-tune-bert-multinerd \
    --pretrained-model $BASE_MODEL \
    --out-dir out/ft-head-prompted-multinerd-eng-$MODEL_NAME \
    --system-prompt ner \
    --head-only"

nix-shell --argstr run "uv run cli.py fine-tune-bert-multinerd \
    --pretrained-model $BASE_MODEL \
    --out-dir out/ft-head-gibberish-multinerd-eng-$MODEL_NAME \
    --system-prompt random \
    --head-only"

# prompt tune
nix-shell --argstr run "uv run cli.py prompt-tune-bert-multinerd \
    --pretrained-model $BASE_MODEL \
    --out-dir out/pt-multinerd-eng-$MODEL_NAME"

nix-shell --argstr run "uv run cli.py prompt-tune-bert-multinerd \
    --pretrained-model $BASE_MODEL \
    --out-dir out/pt-random-multinerd-eng-$MODEL_NAME \
    --prefix-random-init"
