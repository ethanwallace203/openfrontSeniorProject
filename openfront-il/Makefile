.PHONY: fetch replay build-dataset train eval test lint

# Stage 1: fetch raw finished games from the public API -> data/raw/
fetch:
	bash scripts/01_fetch_games.sh

# Stage 2: replay games on the real engine, deterministically -> data/snapshots/
replay:
	bash scripts/02_replay_games.sh

# Stage 3: featurize snapshots into training tensors -> data/processed/
build-dataset:
	python scripts/03_build_dataset.py

# Stage 4: behavior-clone a policy from data/processed/ -> checkpoint
train:
	python scripts/04_train_bc.py

# Stage 5: evaluate a checkpoint against held-out games -> docs/ report
eval:
	python scripts/05_evaluate.py

test:
	pytest tests/unit
	npm test

lint:
	ruff check src scripts tests
	mypy src
	npm run lint
