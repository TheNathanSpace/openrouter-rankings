# Back-end Docs

## API Details

- Artificial Analysis:
  - API reference: <https://artificialanalysis.ai/api-reference#models-endpoint>
  - Key endpoint: `GET /data/llms/models`
  - Authentication: `x-api-key: <api-key>` header
- OpenRouter:
  - API reference: <https://openrouter.ai/docs/api/api-reference/models/get-models>
  - Key endpoint: `GET /models`
  - Authentication: `Authorization: Bearer <api-key>` header

## Cross-Referencing OpenRouter and Artificial Analysis

1. Split OpenRouter model IDs into provider and model names. Extract providers.
2. Extract providers from each Artificial Analysis model.
3. De-duplicate each list of providers.
4. Pair up OpenRouter and Artificial Analysis providers.
5. For each provider, check Levenshtein distance between each model combination.
