#!/usr/bin/env bash

rsync -rv \
    --exclude 'checkpoint-*' \
    --exclude 'slurm' \
    $1:git/icft/out .
