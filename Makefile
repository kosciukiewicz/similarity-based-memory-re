help:
	@echo "  setup >>> set up environment and installing dependencies"
	@echo "  setup-cuda >>> set up environment and installing dependencies using CUDA version of pytorch library"
	@echo "  fetch-datasets >>> fetch datasets in ready-to-use format"
	@echo "  fetch-models >>> fetch pretrained models for example inference"
	@echo "  check >>> check the source for style errors"
	@echo "  fix >>> fix style errors"
	@echo "  fix-check >>> fix style errors and check"
	@echo "  clean >>> remove temporary files"

.PHONY: setup
setup:
	@echo ">>> Setting up environment and installing dependencies <<<"
	poetry install --no-interaction

.PHONY: setup-cuda
setup-cuda:
	@echo ">>> Setting up environment and installing dependencies forcing the CUDA version of pytorch library <<<"
	poetry install --no-interaction && poetry run poe force_cuda

.PHONY: fetch-datasets
fetch-datasets:
	@echo ">>> Downloading datasets into ./storage/data/dataset <<<"
	./scripts/datasets/fetch_datasets.sh

.PHONY: fetch-models
fetch-models:
	@echo ">>> Downloading pretrained models into ./storage/models <<<"
	./scripts/fetch_models.sh

.PHONY: check
check:
	@echo ">>> Checking the source code for style errors <<<"
	poetry run poe check

.PHONY: fix
fix:
	@echo ">>> Fixing the source code from style errors <<<"
	poetry run poe fix

.PHONY: fix-check
fix-check:
	@echo ">>> Fixing the source code from style errors and run check <<<"
	poetry run poe fix_check

.PHONY: clean
clean:
	@echo ">>> Removing temporary files <<<"
	rm -rf .artifacts .mypy_cache .pytest_cache .coverage
