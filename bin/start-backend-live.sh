#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR/..

source .venv/bin/activate
export WATCHFILES_FORCE_POLLING=true
uvicorn llm_rankings.api:app --reload --reload-include 'backend/**'
