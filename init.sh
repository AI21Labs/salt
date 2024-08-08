#!/bin/bash

cd "$(dirname "$0")" || exit

# install pre-commit if not already installed
if ! pre-commit --version; then
  brew install pre-commit
fi

# install pre-commit hooks
pre-commit install --install-hooks -t pre-commit -t commit-msg
