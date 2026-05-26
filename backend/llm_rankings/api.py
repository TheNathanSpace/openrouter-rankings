import logging
from fastapi import FastAPI, HTTPException
import uvicorn

from llm_rankings.database import populate_with_models, wipe_database, get_all_models
from llm_rankings.retrieve_data import get_all_model_data
from llm_rankings.combined_models import CombinedModel
from llm_rankings.util import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Rankings API")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/refresh")
def refresh_data():
    try:
        logger.info("Refreshing data from APIs...")
        # Step 1: Retrieve data from APIs
        get_all_model_data()
        
        # Step 2: Wipe and repopulate database
        wipe_database()
        populate_with_models()
        
        logger.info("Data refreshed successfully.")
        return {"message": "Data refreshed successfully"}
    except Exception as e:
        logger.exception("Failed to refresh data")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=list[CombinedModel])
def get_models():
    try:
        return get_all_models()
    except Exception as e:
        logger.exception("Failed to retrieve models")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
