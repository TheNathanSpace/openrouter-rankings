#!/usr/bin/env bash

if [ ! -d "backend" ]; then
  echo "Error!"
  echo "Expected directory 'backend' does not exist. Are you running this from the correct location?"
  echo "You should be executing: ./bin/generate-openapi-docs.sh"
  exit 1
fi

source .venv/bin/activate
cd backend/llm_rankings
python -c "import api; import json; print(json.dumps(api.app.openapi(), indent=4))" > ../../data/openapi.json
