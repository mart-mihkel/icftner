DOCUMENT_VIEWER = zathura

.PHONY: sync
sync:
	uv sync
	uv pip install -e .

.PHONY: marimo
marimo:
	uv run marimo edit

.PHONY: test
test:
	uv run pytest

.PHONY: types
types:
	uv run ty check

.PHONY: lint
lint:
	uv run ruff check

.PHONY: format
format:
	uv run ruff format

.PHONY: check
check: format lint types test

.PHONY: typst
typst:
	typst watch typesetting/main.typ --open $(DOCUMENT_VIEWER)

.PHONY: rsync
rsync:
	rsync -r --exclude='.venv' . $(REMOTE):git/cptlms
