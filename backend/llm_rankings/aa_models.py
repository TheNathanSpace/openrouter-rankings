import datetime
import string
from typing import Any

from pydantic import BaseModel, ConfigDict

# https://artificialanalysis.ai/api-reference/#models-endpoint


class AABaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def optional_field(key: str, value: Any) -> dict[str, Any]:
        return {key: value} if value else {}


class AACreator(AABaseModel):
    id: str
    name: str
    slug: str


class AAEvaluations(AABaseModel):
    artificial_analysis_intelligence_index: float | None = None
    artificial_analysis_coding_index: float | None = None
    artificial_analysis_math_index: float | None = None
    mmlu_pro: float | None = None
    gpqa: float | None = None
    hle: float | None = None
    livecodebench: float | None = None
    scicode: float | None = None
    math_500: float | None = None
    aime: float | None = None
    aime_25: float | None = None
    ifbench: float | None = None
    lcr: float | None = None
    terminalbench_hard: float | None = None
    tau2: float | None = None


class AAPricing(AABaseModel):
    price_1m_blended_3_to_1: float | None = None
    price_1m_input_tokens: float | None = None
    price_1m_output_tokens: float | None = None

    def get_minimal(self) -> dict[str, float]:
        return {
            **self.optional_field("input", self.price_1m_input_tokens),
            **self.optional_field("output", self.price_1m_output_tokens),
        }


class AAPromptOptions(AABaseModel):
    parallel_queries: int | None = None
    prompt_length: str | int | None = None


class AAModel(AABaseModel):
    id: str
    name: str
    slug: str
    release_date: str | None = None
    model_creator: AACreator
    evaluations: AAEvaluations | None = None
    pricing: AAPricing | None = None
    median_output_tokens_per_second: float | None = None
    median_time_to_first_token_seconds: float | None = None
    median_time_to_first_answer_token: float | None = None  # Accounts for reasoning

    def get_url(self):
        return f"https://artificialanalysis.ai/models/{self.slug}"

    def get_provider(self) -> str:
        return self.model_creator.name

    def get_clean_name(self) -> str:
        """gemini-3-5-flash -> gemini 3 5 flash"""
        name = self.slug
        # Remove punctuation
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
        if self.release_date:
            return datetime.datetime.strptime(self.release_date, "%Y-%m-%d").replace(
                tzinfo=datetime.UTC
            )
        else:
            return None

    def get_minimal_pricing(self) -> dict[str, float]:
        if self.pricing is None:
            return {}
        return self.pricing.get_minimal()

    def __hash__(self) -> int:
        return self.id.__hash__()


class ArtificialAnalysisAPIResponse(AABaseModel):
    status: int
    prompt_options: AAPromptOptions | None = None
    data: list[AAModel]

    def get_providers(self) -> list[str]:
        return list({model.get_provider() for model in self.data})

    def get_models_for_provider(self, provider: str) -> list[AAModel]:
        return [model for model in self.data if model.get_provider() == provider]
