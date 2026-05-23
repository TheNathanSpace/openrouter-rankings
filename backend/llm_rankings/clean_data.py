import json
import logging
import string
from pathlib import Path

from rapidfuzz.distance.metrics_cpp import levenshtein_distance
from slugify import slugify

from llm_rankings import util
from llm_rankings.retrieve_data import get_all_model_data
from llm_rankings.util import (
    erase_data_dir,
    get_data_dir,
    invert_dict,
    setup_logging,
    sort_dict,
    split_to_words,
    string_is_only_in_one,
)


def write_models_data(or_models: list[dict], aa_models: list[dict], prefix: str = ""):
    logging.debug("Writing models data to files")
    data_dir = get_data_dir()
    (data_dir / f"{prefix}_or_models.json").write_text(json.dumps(or_models, indent=4))
    (data_dir / f"{prefix}_aa_models.json").write_text(json.dumps(aa_models, indent=4))


def write_unmatched_providers(or_providers: list[str], aa_providers: list[str]):
    logging.debug("Writing providers data to files")
    data_dir = get_data_dir()
    (data_dir / "unmatched_or_providers.json").write_text(
        json.dumps(sorted(or_providers), indent=4)
    )
    (data_dir / "unmatched_aa_providers.json").write_text(
        json.dumps(sorted(aa_providers), indent=4)
    )


def write_matched_providers(or_aa_providers: dict[str, str]):
    logging.debug("Writing matched providers data to files")
    data_dir = get_data_dir()
    (data_dir / "or_to_aa_providers.json").write_text(
        json.dumps(sort_dict(or_aa_providers), indent=4)
    )


def safe_delete_key(d: dict, key: str):
    if key in d:
        del d[key]


def clean_or_models(or_models: dict) -> list[dict]:
    logging.debug("Cleaning OpenRouter models")
    cleaned_models = [model for model in or_models["data"] if not model["id"].endswith(":free")]
    cleaned_models = [model for model in cleaned_models if not model["id"].startswith("~")]
    cleaned_models = [model for model in cleaned_models if not model["id"].startswith("openrouter")]
    for model in cleaned_models:
        model["created"] = util.unix_epoch_to_utc(model["created"])
        safe_delete_key(model, "per_request_limit")
        safe_delete_key(model, "default_parameters")
        safe_delete_key(model, "supported_voices")
        safe_delete_key(model, "expiration_date")
        safe_delete_key(model, "top_provider")
        safe_delete_key(model, "supported_parameters")
        safe_delete_key(model, "hugging_face_id")

        if "architecture" in model:
            if (
                "text" not in model["architecture"]["input_modalities"]
                or "text" not in model["architecture"]["output_modalities"]
            ):
                safe_delete_key(model, "architecture")

    return cleaned_models


def clean_aa_models(aa_models: dict) -> list[dict]:
    logging.debug("Cleaning Artificial Analysis models")
    cleaned_models = aa_models["data"]
    return cleaned_models


def filter_models_to_shared_providers(
    or_models: list[dict], aa_models: list[dict], matched_providers: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    or_models = or_models.copy()
    aa_models = aa_models.copy()

    or_providers = matched_providers.keys()
    aa_providers = matched_providers.values()

    or_models_filtered = []
    for model in or_models:
        model_id = model["id"]
        for provider in or_providers:
            if provider in model_id:
                or_models_filtered.append(model)
                break
    aa_models_filtered = []
    for model in aa_models:
        model_provider = model["model_creator"]["name"]
        for filtered_provider in aa_providers:
            if model_provider == filtered_provider:
                aa_models_filtered.append(model)
                break

    return or_models_filtered, aa_models_filtered


def get_model_data() -> tuple[list[dict], list[dict]]:
    logging.debug("Retrieving and cleaning models data")
    or_models, aa_models = get_all_model_data()
    or_models = clean_or_models(or_models)
    aa_models = clean_aa_models(aa_models)
    return or_models, aa_models


def extract_or_providers(or_models: list[dict]) -> set[str]:
    logging.debug("Extracting OpenRouter providers")
    return {model["id"].split("/")[0] for model in or_models}


def extract_aa_providers(aa_models: list[dict]) -> set[str]:
    logging.debug("Extracting Artificial Analysis providers")
    return {model["model_creator"]["name"] for model in aa_models}


def remove_used_providers(
    matched: dict[str, str], or_providers: set[str], aa_providers: set[str]
) -> tuple[set[str], set[str]]:
    logging.debug("Removing used providers")
    new_or_providers = or_providers.copy()
    new_aa_providers = aa_providers.copy()
    new_or_providers -= set(matched.keys())
    new_aa_providers -= set(matched.values())
    return new_or_providers, new_aa_providers


def match_providers(
    or_providers: set[str], aa_providers: set[str]
) -> tuple[dict[str, str], list[str], list[str]]:
    """Returns OpenRouter -> Artificial Analysis provider mapping."""
    logging.debug("Matching providers")
    manual_mappings = {
        "meta-llama": "Meta",
        "mistralai": "Mistral",
        "kwaipilot": "KwaiKAT",
        "qwen": "Alibaba",
    }

    or_to_aa_map: dict[str, str] = {}

    # First, try simply matching by lowercase name
    for or_provider in or_providers:
        for aa_provider in aa_providers:
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


def get_models_for_provider(
    provider: str, providers_map: dict[str, str], or_models: list[dict], aa_models: list[dict]
) -> tuple[list[dict], list[dict]]:
    logging.debug(f"Getting models for provider: {provider}")
    # Get both providers
    or_provider: str = None
    aa_provider: str = None
    if provider in providers_map:
        or_provider = provider
        aa_provider = providers_map[or_provider]
    elif provider in providers_map.values():
        aa_provider = provider
        or_provider = invert_dict(providers_map)[aa_provider]
    else:
        raise ValueError(f"Provider not found in list of OR or AA providers: {provider}")

    logging.debug(f"AA provider: {aa_provider}")
    logging.debug(f"OR provider: {or_provider}")

    or_models_filtered = []
    aa_models_filtered = []
    for model in or_models:
        if or_provider in model["id"]:
            or_models_filtered.append(model)
    for model in aa_models:
        if aa_provider == model["model_creator"]["name"]:
            aa_models_filtered.append(model)
    return or_models_filtered, aa_models_filtered


def parse_providers(or_models: list[dict], aa_models: list[dict]) -> dict[str, str]:
    or_providers = extract_or_providers(or_models)
    aa_providers = extract_aa_providers(aa_models)
    matched_providers, remaining_or_providers, remaining_aa_providers = match_providers(
        or_providers, aa_providers
    )
    write_providers_data(matched_providers, remaining_or_providers, remaining_aa_providers)
    return matched_providers


def write_providers_data(
    matched_providers: dict[str, str],
    remaining_or_providers: list[str],
    remaining_aa_providers: list[str],
):
    write_matched_providers(matched_providers)
    write_unmatched_providers(remaining_or_providers, remaining_aa_providers)


def validate_match(model_a: str, model_b: str) -> bool:
    if string_is_only_in_one("non-instruct", model_a, model_b):
        return False
    if string_is_only_in_one("non-reasoning", model_a, model_b):
        return False
    if string_is_only_in_one("instruct", model_a, model_b):
        return False
    if string_is_only_in_one("reasoning", model_a, model_b):
        return False
    # Instruction tuned
    # if string_is_only_in_one("it", model_a, model_b):
    #     return False

    # Ensure same version
    a_words = split_to_words(model_a)
    b_words = split_to_words(model_b)
    a_ints = [int(w) for w in a_words if len(w) == 1 and w.isdigit()]
    b_ints = [int(w) for w in b_words if len(w) == 1 and w.isdigit()]
    versions_match = a_ints == b_ints
    if not versions_match:
        return False

    return True


def alphabetical_compare(a: str, b: str) -> int:
    a_words = split_to_words(a)
    b_words = split_to_words(b)

    a_alpha = " ".join(sorted(a_words))
    b_alpha = " ".join(sorted(b_words))

    return levenshtein_distance(a_alpha, b_alpha)


def match_provider_models(
    or_provider: str, aa_provider: str, or_models: list[dict], aa_models: list[dict]
) -> tuple[dict[str, str], list[str], list[str]]:
    # OR models: id google/gemini-3.5-flash -> gemini 3 5 flash
    or_models_clean_to_dirty_map: dict[str, str] = {}
    for or_model in or_models:
        or_dirty = or_model["id"]
        # Remove provider name
        or_clean = "/".join(or_dirty.split("/")[1:])
        # Remove punctuation
        or_clean = (
            or_clean.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
            .strip()
            .lower()
        )
        or_clean = or_clean.replace("thinking", "reasoning")
        # Collapse extra spaces
        or_clean = " ".join(or_clean.split())
        or_models_clean_to_dirty_map[or_clean] = or_dirty

    # AA models: slug gemini-3-5-flash -> gemini 3 5 flash
    aa_models_clean_to_dirty_map: dict[str, str] = {}
    for aa_model in aa_models:
        aa_dirty = aa_model["slug"]
        # Remove punctuation
        aa_clean = (
            aa_dirty.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
            .strip()
            .lower()
        )
        aa_clean = aa_clean.replace("thinking", "reasoning")
        # Collapse extra spaces
        aa_clean = " ".join(aa_clean.split())
        aa_models_clean_to_dirty_map[aa_clean] = aa_dirty

    remaining_or_models = list(or_models_clean_to_dirty_map.keys())
    remaining_aa_models = list(aa_models_clean_to_dirty_map.keys())

    # Pair up using Levenshtein distance
    clean_matched_models: dict[str, str] = {}
    for clean_or in remaining_or_models.copy():
        closest_match = None
        closest_distance = float("inf")
        for clean_aa in remaining_aa_models.copy():
            if not validate_match(clean_or, clean_aa):
                continue
            distance = alphabetical_compare(clean_or, clean_aa)
            if distance < closest_distance and distance <= 4:
                closest_distance = distance
                closest_match = clean_aa

        if closest_match:
            clean_matched_models[clean_or] = closest_match
            remaining_or_models.remove(clean_or)
            remaining_aa_models.remove(closest_match)

    dirty_matched_models: dict[str, str] = {}
    for clean_or, clean_aa in clean_matched_models.items():
        dirty_matched_models[or_models_clean_to_dirty_map[clean_or]] = aa_models_clean_to_dirty_map[
            clean_aa
        ]

    dirty_remaining_or_models = [or_models_clean_to_dirty_map[m] for m in remaining_or_models]
    dirty_remaining_aa_models = [aa_models_clean_to_dirty_map[m] for m in remaining_aa_models]

    return (
        sort_dict(dirty_matched_models),
        sorted(dirty_remaining_or_models),
        sorted(dirty_remaining_aa_models),
    )


def create_provider_dir(or_provider: str) -> Path:
    data_dir = get_data_dir()
    provider_slug = slugify(or_provider)
    provider_dir = data_dir / provider_slug
    provider_dir.mkdir(parents=True, exist_ok=True)
    return provider_dir


def write_matched_models(or_provider: str, matched_models: dict):
    logging.debug(f"Writing matched models for provider {or_provider} to file")
    provider_dir = create_provider_dir(or_provider)
    (provider_dir / "matched_models.json").write_text(json.dumps(matched_models, indent=4))


def write_unmatched_models(or_provider: str, provider_models: list[dict], is_aa: bool = False):
    logging.debug(f"Writing unmatched models for provider {or_provider} to file")
    provider_dir = create_provider_dir(or_provider)
    (provider_dir / f"unmatched_models_{('aa' if is_aa else 'or')}.json").write_text(
        json.dumps(provider_models, indent=4)
    )


def process_provider(
    or_provider: str,
    aa_provider: str,
    matched_providers: dict[str, str],
    or_models_filtered: list[dict],
    aa_models_filtered: list[dict],
):
    or_models_for_provider, aa_models_for_provider = get_models_for_provider(
        or_provider, matched_providers, or_models_filtered, aa_models_filtered
    )
    write_unmatched_models(or_provider, or_models_for_provider, is_aa=False)
    write_unmatched_models(or_provider, aa_models_for_provider, is_aa=True)

    matched_models, remaining_or_models, remaining_aa_models = match_provider_models(
        or_provider, aa_provider, or_models_for_provider, aa_models_for_provider
    )
    provider_dict = {
        "or_to_aa_models": matched_models,
        "unmatched_or_models": remaining_or_models,
        "unmatched_aa_models": remaining_aa_models,
    }
    write_matched_models(or_provider, provider_dict)


def get_and_clean_data() -> tuple[list[dict], list[dict]]:
    logging.debug("Retrieving and cleaning data")

    or_models, aa_models = get_model_data()
    write_models_data(or_models, aa_models, "cleaned")

    matched_providers = parse_providers(or_models, aa_models)

    or_models_filtered, aa_models_filtered = filter_models_to_shared_providers(
        or_models, aa_models, matched_providers
    )
    write_models_data(or_models_filtered, aa_models_filtered, "filtered")

    for or_provider, aa_provider in matched_providers.items():
        process_provider(
            or_provider, aa_provider, matched_providers, or_models_filtered, aa_models_filtered
        )

    return or_models_filtered, aa_models_filtered


if __name__ == "__main__":
    setup_logging("INFO")
    erase_data_dir()
    or_models, aa_models = get_and_clean_data()
