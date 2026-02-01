REMOTE ?=
LOGDIR = out
NOTEBOOKS = notebooks
PDF_VIEWER = zathura

install:
	uv sync
	uv pip install --editable .

pre-commit:
	uv run ruff check --fix
	uv run ruff format
	uv run ty check

marimo:
	uv run marimo edit $(NOTEBOOKS)

tensorboard:
	uv run tensorboard --logdir $(LOGDIR)

typst:
	typst watch typesetting/main.typ --open $(PDF_VIEWER)

upload:
	rsync -rv --exclude-from '.gitignore' . $(REMOTE)

download-out:
	rsync -rv $(REMOTE)/$(LOGDIR) .

download-tensorboard:
	rsync -rv --exclude 'checkpoint-*' --exclude 'slurm' $(REMOTE)/$(LOGDIR) .
