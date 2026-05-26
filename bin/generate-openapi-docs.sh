#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR/..

source .venv/bin/activate
cd backend/llm_rankings
python -c "import api; import json; print(json.dumps(api.app.openapi(), indent=4))" > ../../data/openapi.json
