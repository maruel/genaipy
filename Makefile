# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Build, test, lint, and format the repository.

.DEFAULT_GOAL := help
.PHONY: help tools custom-gcl fix verify build test test-race smoke git-hooks

GOLANGCI_VERSION := 2.14.0
RUFF_VERSION := 0.16.8
SHFMT_VERSION := 3.12.0
export PATH := $(shell go env GOPATH)/bin:$(PATH)
export PYTHONDONTWRITEBYTECODE := 1

tools:
	@golangci-lint version 2>/dev/null | grep -Fq 'version $(GOLANGCI_VERSION) ' || go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v$(GOLANGCI_VERSION)
	@ruff --version 2>/dev/null | grep -Fxq 'ruff $(RUFF_VERSION)' || { command -v uv >/dev/null && uv tool install --force ruff==$(RUFF_VERSION); }
	@shfmt --version 2>/dev/null | grep -Fxq 'v$(SHFMT_VERSION)' || go install mvdan.cc/sh/v3/cmd/shfmt@v$(SHFMT_VERSION)
	@command -v shellcheck >/dev/null || { echo 'Install shellcheck to run shell checks.' >&2; exit 1; }

custom-gcl: tools
	@want=$$({ sha256sum .custom-gcl.yml; echo '$(GOLANGCI_VERSION)'; go env GOVERSION; } | sha256sum | cut -d' ' -f1); \
	if [ -x custom-gcl ] && [ "$$want" = "$$(cat .custom-gcl.sha 2>/dev/null)" ]; then exit 0; fi; \
	golangci-lint custom --version v$(GOLANGCI_VERSION) && echo "$$want" > .custom-gcl.sha

fix: custom-gcl
	@./custom-gcl run --show-stats=false --fix ./...
	@golangci-lint fmt
	@ruff check --fix .
	@ruff format .
	@python3 scripts/update_agents_file_index.py
	@shfmt -w setup.sh setup-test.sh scripts/install-git-hooks.sh scripts/hooks/pre-commit scripts/hooks/commit-msg

verify: custom-gcl
	@./custom-gcl run --show-stats=false ./...
	@ruff check .
	@ruff format --check .
	@test -z "$$(shfmt -l setup.sh setup-test.sh scripts/install-git-hooks.sh scripts/hooks/pre-commit scripts/hooks/commit-msg)"
	@shellcheck setup.sh setup-test.sh scripts/install-git-hooks.sh scripts/hooks/pre-commit scripts/hooks/commit-msg
	@go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12
	# addlicense does not honor .gitignore; exclude model caches and Python environments.
	@go run github.com/google/addlicense@v1.2.0 -check -ignore 'cache/**' -ignore 'venv/**' -ignore 'venv-test/**' .
	@python3 scripts/lint_binaries.py
	@python3 scripts/update_agents_file_index.py --check

build:
	@go build ./...

# Default tests are offline; models and Python model packages are not required.
test:
	@go test ./...
	@python3 -m unittest discover -s tests

test-race:
	@go test -race ./...

smoke:
	@GENAIPY_MODEL_SMOKE=1 go test -run '^TestModelSmoke$$' -timeout 20m -v .

git-hooks:
	@./scripts/install-git-hooks.sh

help:
	@echo 'make fix, verify, build, test, test-race, git-hooks'
	@echo 'make smoke installs dependencies and downloads a real model (explicit opt-in).'
