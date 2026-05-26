import json
import logging
from itertools import product

from rapidfuzz.distance.metrics_cpp import levenshtein_distance

from llm_rankings.aa_models import AAModel, ArtificialAnalysisAPIResponse
from llm_rankings.combined_models import CombinedModel
from llm_rankings.or_models import OpenRouterAPIResponse, OpenRouterModel
from llm_rankings.retrieve_data import get_all_model_data
from llm_rankings.util import (
    erase_data_dir,
    get_data_dir,
    get_intermediate_data_dir,
    get_provider_dir,
    setup_logging,
    sort_dict,
    split_to_words,
    string_is_only_in_one,
)


def write_providers(
    or_aa_providers: dict[str, str],
    remaining_or_providers: list[str],
    remaining_aa_providers: list[str],
):
    logging.debug("Writing providers data to files")
    providers = {
        "matched": sort_dict(or_aa_providers),
        "unmatched_openrouter_providers": sorted(remaining_or_providers),
        "unmatched_artificialanalysis_providers": sorted(remaining_aa_providers),
    }
    (get_intermediate_data_dir() / "providers.json").write_text(json.dumps(providers, indent=4))


def filter_models_to_shared_providers(
    or_models: OpenRouterAPIResponse,
    aa_models: ArtificialAnalysisAPIResponse,
    matched_providers: dict[str, str],
) -> tuple[list[OpenRouterModel], list[AAModel]]:
    or_providers = matched_providers.keys()
    aa_providers = matched_providers.values()

    or_models_filtered = []
    for model in or_models.data:
        for provider in or_providers:
            if provider == model.get_provider():
                or_models_filtered.append(model)
                break

    aa_models_filtered = []
    for model in aa_models.data:
        model_provider = model.model_creator.name
        for filtered_provider in aa_providers:
            if model_provider == filtered_provider:
                aa_models_filtered.append(model)
                break

    return or_models_filtered, aa_models_filtered


def remove_used_providers(
    matched: dict[str, str], or_providers: list[str], aa_providers: list[str]
) -> tuple[list[str], list[str]]:
    logging.debug("Removing used providers")
    new_or_providers = set(or_providers)
    new_aa_providers = set(aa_providers)
    new_or_providers -= set(matched.keys())
    new_aa_providers -= set(matched.values())
    return list(new_or_providers), list(new_aa_providers)


def match_providers(
    or_models: OpenRouterAPIResponse, aa_models: ArtificialAnalysisAPIResponse
) -> tuple[dict[str, str], list[str], list[str]]:
    """Returns OpenRouter -> Artificial Analysis provider mapping."""
    logging.debug("Matching providers")
    manual_mappings = {
        "ai21": "AI21 Labs",
        "allenai": "Allen Institute for AI",
        "ibm-granite": "IBM",
        "liquid": "Liquid AI",
        "meta-llama": "Meta",
        "mistralai": "Mistral",
        "moonshotai": "Kimi",
        "kwaipilot": "KwaiKAT",
        "qwen": "Alibaba",
    }

    or_providers = or_models.get_providers()
    aa_providers = aa_models.get_providers()

    or_to_aa_map: dict[str, str] = {}

    # First, try simply matching by lowercase name
    for or_provider in or_models.get_providers():
        for aa_provider in aa_models.get_providers():
            if or_provider.lower() == aa_provider.lower():
                or_to_aa_map[or_provider] = aa_provider
                break
    or_providers, aa_providers = remove_used_providers(or_to_aa_map, or_providers, aa_providers)

    # Next, try removing dashes and spaces
    for or_provider in or_providers:
        for aa_provider in aa_providers:
            if or_provider.lower().replace("-", "").replace(" ", "") == aa_provider.lower().replace(
                "-", ""
            ).replace(" ", ""):
                or_to_aa_map[or_provider] = aa_provider
                break
    or_providers, aa_providers = remove_used_providers(or_to_aa_map, or_providers, aa_providers)

    # Next, use manual mappings
    for or_provider, aa_provider in manual_mappings.items():
        or_to_aa_map[or_provider] = aa_provider
        if or_provider in or_providers:
            or_providers.remove(or_provider)
        if aa_provider in aa_providers:
            aa_providers.remove(aa_provider)

    return or_to_aa_map, list(or_providers), list(aa_providers)


def parse_providers(
    or_models: OpenRouterAPIResponse, aa_models: ArtificialAnalysisAPIResponse
) -> dict[str, str]:
    matched_providers, remaining_or_providers, remaining_aa_providers = match_providers(
        or_models, aa_models
    )
    write_providers(matched_providers, remaining_or_providers, remaining_aa_providers)
    return matched_providers


def validate_match(model_a: OpenRouterModel, model_b: AAModel) -> bool:
    if string_is_only_in_one("non-instruct", model_a.get_clean_name(), model_b.get_clean_name()):
        return False
    if string_is_only_in_one("non-reasoning", model_a.get_clean_name(), model_b.get_clean_name()):
        return False
    if string_is_only_in_one("instruct", model_a.get_clean_name(), model_b.get_clean_name()):
        return False
    if string_is_only_in_one("reasoning", model_a.get_clean_name(), model_b.get_clean_name()):
        return False

    # Ensure same version
    a_words = split_to_words(model_a.get_clean_name())
    b_words = split_to_words(model_b.get_clean_name())
    a_ints = [int(w) for w in a_words if len(w) == 1 and w.isdigit()]
    b_ints = [int(w) for w in b_words if len(w) == 1 and w.isdigit()]
    versions_match = a_ints == b_ints
    if not versions_match:
        return False

    return True


def alphabetical_compare(a: OpenRouterModel, b: AAModel) -> int:
    a_words = split_to_words(a.get_clean_name())
    b_words = split_to_words(b.get_clean_name())

    a_alpha = " ".join(sorted(a_words))
    b_alpha = " ".join(sorted(b_words))

    return levenshtein_distance(a_alpha, b_alpha)


def get_match_score(or_model: OpenRouterModel, aa_model: AAModel) -> float:
    if not validate_match(or_model, aa_model):
        return float("inf")
    distance = alphabetical_compare(or_model, aa_model)
    if or_model.get_clean_name() == "qwen3 6 flash":
        logging.debug(f"{or_model} / {aa_model} : {distance}")
    if distance <= 4:
        return distance
    return float("inf")


def match_models(
    or_models: list[OpenRouterModel], aa_models: list[AAModel]
) -> tuple[list[tuple[OpenRouterModel, AAModel]], list[OpenRouterModel], list[AAModel]]:
    """The list of models should already be filtered down to the intended provider."""
    remaining_or_models = or_models.copy()
    remaining_aa_models = aa_models.copy()
    matched_models: list[tuple[OpenRouterModel, AAModel]] = []

    while len(remaining_or_models) > 0 and len(remaining_aa_models) > 0:
        combinations: list[tuple[OpenRouterModel, AAModel]] = list(
            product(remaining_or_models, remaining_aa_models)
        )
        scores: dict[tuple[OpenRouterModel, AAModel], float] = {
            (or_model, aa_model): get_match_score(or_model, aa_model)
            for or_model, aa_model in combinations
        }
        min_combination: tuple[OpenRouterModel, AAModel]
        min_score: float
        min_combination, min_score = min(scores.items(), key=lambda x: x[1])
        if min_score <= 4:
            matched_models.append(min_combination)
            remaining_or_models.remove(min_combination[0])
            remaining_aa_models.remove(min_combination[1])
        else:
            break

    return (
        matched_models,
        remaining_or_models,
        remaining_aa_models,
    )


def write_matched_models(
    provider_name: str,
    matched_models: list[tuple[OpenRouterModel, AAModel]],
    remaining_or_models: list[OpenRouterModel],
    remaining_aa_models: list[AAModel],
):
    logging.debug(f"Writing matched models for provider {provider_name} to file")
    provider_dict = {
        "openrouter_to_artificialanalysis_model_names": [
            f"{m1.get_clean_name()} -> {m2.get_clean_name()}" for m1, m2 in matched_models
        ],
        "unmatched_model_names": {
            "openrouter": [m.get_clean_name() for m in remaining_or_models],
            "artificialanalysis": [m.get_clean_name() for m in remaining_aa_models],
        },
        "openrouter_to_artificialanalysis_models": [
            [m1.model_dump(), m2.model_dump_json()] for m1, m2 in matched_models
        ],
        "unmatched_models": {
            "unmatched_openrouter_models": [m.model_dump() for m in remaining_or_models],
            "unmatched_artificialanalysis_models": [m.model_dump() for m in remaining_aa_models],
        },
    }
    provider_dir = get_provider_dir()
    (provider_dir / f"{provider_name}.json").write_text(json.dumps(provider_dict, indent=4))


def match_models_for_provider(
    or_provider: str,
    aa_provider: str,
    or_models: OpenRouterAPIResponse,
    aa_models: ArtificialAnalysisAPIResponse,
) -> list[tuple[OpenRouterModel, AAModel]]:
    or_models_for_provider = or_models.get_models_for_provider(or_provider)
    aa_models_for_provider = aa_models.get_models_for_provider(aa_provider)
    matched_models, remaining_or_models, remaining_aa_models = match_models(
        or_models_for_provider, aa_models_for_provider
    )
    write_matched_models(or_provider, matched_models, remaining_or_models, remaining_aa_models)
    return matched_models


def combine_or_aa_models(
    matched_models: list[tuple[OpenRouterModel, AAModel]],
) -> list[CombinedModel]:
    combined_models: list[CombinedModel] = []
    for or_model, aa_model in matched_models:
        # Use earliest release date
        release_date: str | None = None
        or_created = or_model.get_created_date()
        aa_created = aa_model.get_created_date()
        if or_created and aa_created:
            earliest = min(or_created, aa_created)
            release_date = earliest.strftime("%Y-%m-%d")
        elif or_created:
            release_date = or_created.strftime("%Y-%m-%d")
        elif aa_created:
            release_date = aa_created.strftime("%Y-%m-%d")

        # Use OR then AA pricing
        pricing = or_model.get_minimal_pricing()
        if len(pricing) == 0:
            pricing = aa_model.get_minimal_pricing()
        prefix_pricing: dict[str, float] = {}
        for key, value in pricing.items():
            prefix_pricing[f"pricing_{key}"] = value

        prefix_evaluations: dict[str, float] = {}
        evaluations = (
            aa_model.evaluations.model_dump(exclude_none=True, by_alias=False)
            if aa_model.evaluations
            else {}
        )
        for key, value in evaluations.items():
            prefix_evaluations[f"benchmark_{key}"] = round(float(value), 4)

        combined_model = CombinedModel(
            name=aa_model.name,
            creator=aa_model.get_provider(),
            description=or_model.description,
            created=release_date,
            url_openrouter=or_model.get_url(),
            url_artificialanalysis=aa_model.get_url(),
            knowledge_cutoff=or_model.knowledge_cutoff,
            context_length=or_model.context_length,
            **prefix_pricing,
            speed_tokens_per_second=aa_model.median_output_tokens_per_second,
            speed_time_to_first_token=aa_model.median_time_to_first_token_seconds,
            speed_time_to_first_answer_token=aa_model.median_time_to_first_answer_token,
            **prefix_evaluations,
        )
        combined_models.append(combined_model)
    return combined_models


def write_combined_models(combined_models: list[CombinedModel]):
    combined_models_path = get_data_dir() / "combined_models.json"
    serialized = [m.model_dump() for m in combined_models]
    combined_models_path.write_text(json.dumps(serialized, indent=4))
    logging.info(f"Combined models written to {combined_models_path}")


def get_and_clean_data() -> list[CombinedModel]:
    logging.debug("Retrieving and cleaning data")

    or_models: OpenRouterAPIResponse
    aa_models: ArtificialAnalysisAPIResponse
    or_models, aa_models = get_all_model_data()

    matched_providers = parse_providers(or_models, aa_models)

    all_matched_models: list[tuple[OpenRouterModel, AAModel]] = []
    for or_provider, aa_provider in matched_providers.items():
        matched_models = match_models_for_provider(or_provider, aa_provider, or_models, aa_models)
        all_matched_models.extend(matched_models)

    combined_models: list[CombinedModel] = combine_or_aa_models(all_matched_models)
    write_combined_models(combined_models)

    return combined_models


if __name__ == "__main__":
    setup_logging("DEBUG")
    erase_data_dir()
    combined_models = get_and_clean_data()
