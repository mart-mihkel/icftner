#!/usr/bin/env bash

rsync -rv --exclude-from '.gitignore' . $1:git/icft
