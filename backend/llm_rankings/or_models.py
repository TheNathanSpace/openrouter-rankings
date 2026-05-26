import datetime
import logging
import string
from typing import Any

from pydantic import BaseModel, ConfigDict

from llm_rankings import util


class ORBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def optional_field(key: str, value: Any) -> dict[str, Any]:
        return {key: value} if value else {}


# https://openrouter.ai/docs/api/api-reference/models/get-models


class ORPricing(ORBaseModel):
    """Prices are in $/token, NOT $/1M tokens."""

    prompt: float | None = None
    completion: float | None = None
    image: float | None = None
    audio: float | None = None
    request: float | None = None
    web_search: float | None = None
    internal_reasoning: float | None = None
    input_cache_read: float | None = None
    input_cache_write: float | None = None
    audio_output: float | None = None
    image_output: float | None = None
    image_token: float | None = None
    input_audio_cache: float | None = None
    discount: float | None = None

    @staticmethod
    def get_per_million_tokens(value: float | None) -> float | None:
        if value is None:
            return None
        return round(float(value) * 1_000_000, 4)

    def get_minimal(self) -> dict[str, float]:
        return {
            **self.optional_field("input", self.get_per_million_tokens(self.prompt)),
            **self.optional_field("output", self.get_per_million_tokens(self.completion)),
        }


class ORArchitecture(ORBaseModel):
    modality: str | None = None
    input_modalities: list[str] | None = None
    output_modalities: list[str] | None = None
    tokenizer: str | None = None
    instruct_type: str | None = None

    def is_text(self) -> bool:
        if "text" not in self.input_modalities or "text" not in self.output_modalities:
            return False
        return True


class ORTopProvider(ORBaseModel):
    context_length: int | None = None
    max_completion_tokens: int | None = None
    is_moderated: bool | None = None


class ORLinks(ORBaseModel):
    details: str | None = None


class ORPerRequestLimits(ORBaseModel):
    completion_tokens: float | None = None
    prompt_tokens: float | None = None


class OpenRouterModel(ORBaseModel):
    id: str
    name: str | None = None
    created: int | None = None  # Raw data has unix timestamp (int)
    description: str | None = None
    context_length: int | None = None
    architecture: ORArchitecture | None = None
    pricing: ORPricing | None = None
    top_provider: ORTopProvider | None = None
    per_request_limits: ORPerRequestLimits | None = None
    supported_parameters: list[str] | None = None
    default_parameters: dict[str, Any] | None = None
    supported_voices: list[Any] | None = None
    knowledge_cutoff: str | None = None
    expiration_date: str | None = None
    links: ORLinks | None = None
    hugging_face_id: str | None = None
    canonical_slug: str | None = None

    class Config:
        populate_by_name = True

    def get_url(self) -> str | None:
        if self.links and self.links.details and self.canonical_slug:
            return "https://openrouter.ai/" + self.canonical_slug.lstrip("/")
        return None

    def get_minimal_pricing(self) -> dict[str, float]:
        if self.pricing is None:
            return {}
        return self.pricing.get_minimal()

    def get_minimal_created(self) -> str | None:
        if self.created is None:
            return None
        return util.unix_epoch_to_utc(self.created)

    def get_provider(self) -> str:
        return self.id.split("/")[0]

    def get_clean_name(self) -> str:
        """google/gemini-3.5-flash -> gemini 3 5 flash"""
        name = "/".join(self.id.split("/")[1:])
        name = (
            name.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
            .strip()
            .lower()
        )
        name = name.replace("thinking", "reasoning")
        # Collapse extra spaces
        name = " ".join(name.split())
        return name

    def get_created_date(self) -> datetime.datetime | None:
        if self.created:
            return datetime.datetime.fromtimestamp(self.created, datetime.UTC)
        else:
            return None

    def __hash__(self) -> int:
        return self.id.__hash__()


class OpenRouterAPIResponse(ORBaseModel):
    data: list[OpenRouterModel]

    def get_minimal_models(self) -> list[dict]:
        logging.debug("Cleaning OpenRouter models")
        all_minimal: list[dict] = []
        for model in self.data:
            minimal = model.get_minimal()
            if minimal is not None:
                all_minimal.append(minimal)
        return all_minimal

    def get_providers(self) -> list[str]:
        return list({model.get_provider() for model in self.data})

    def get_models_for_provider(self, provider: str) -> list[OpenRouterModel]:
        return [model for model in self.data if model.get_provider() == provider]
