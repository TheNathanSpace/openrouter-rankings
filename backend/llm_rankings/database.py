import logging
from pathlib import Path

from pydantic_sqlite import DataBase

from llm_rankings.clean_data import get_and_clean_data
from llm_rankings.combined_models import CombinedModel
from llm_rankings.util import get_data_dir, setup_logging


def get_database_path() -> Path:
    return get_data_dir() / "database.db"


def initialize_database() -> DataBase:
    logging.debug(f"Initializing database at {get_database_path().as_posix()}")
    db = DataBase(filename_or_conn=get_database_path())
    return db


def wipe_database():
    logging.debug(f"Wiping database at {get_database_path().as_posix()}")
    Path.unlink(get_database_path(), missing_ok=True)


def populate_with_models():
    models: list[CombinedModel] = get_and_clean_data()
    db: DataBase = initialize_database()
    logging.debug(f"Writing {len(models)} models to database")
    for model in models:
        model.add_to_database(db)
    logging.debug(f"Wrote {len(models)} models to database")


def get_all_models() -> list[CombinedModel]:
    db: DataBase = initialize_database()
    models: list[CombinedModel] = [model for model in db("models")]
    return models


if __name__ == "__main__":
    setup_logging("DEBUG")
    wipe_database()
    populate_with_models()
