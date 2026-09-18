.PHONY: help install test check train evaluate predict compare gui clean

CONFIG ?= configs/default.yaml

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run tests
	pytest

check: ## Verify the dataset layout
	python -m medsam_seg check --config $(CONFIG)

train: ## Train a model
	python -m medsam_seg train --config $(CONFIG)

evaluate: ## Evaluate saved weights
	python -m medsam_seg evaluate --config $(CONFIG)

predict: ## Segment one image: make predict IMG=scan.png
	python -m medsam_seg predict $(IMG) --config $(CONFIG)

compare: ## Compare architectures: make compare IMG=scan.png
	python -m medsam_seg compare $(IMG) --config $(CONFIG)

gui: ## Open the desktop app
	python -m medsam_seg gui --config $(CONFIG)

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
