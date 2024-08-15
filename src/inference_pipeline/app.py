from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from utils.startup import load_model
from utils.validate_input import validate_input
from utils.preprocess import preprocess
from utils.postprocess import postprocess
from utils.format_output import format_output
import numpy as np
import json

app = FastAPI()

# Global variables dictionary
global_vars = {
    'predict': None,
    'min_max_values': None
}

# Load the model from S3 at startup
@app.on_event("startup")
async def startup_event():
    await load_model(global_vars)

@app.post("/predict")
async def predict_endpoint(request: Request):
    input = await request.json()
    if (not validate_input(input)):
        raise HTTPException(status_code=400, detail="Invalid input data")
    try:
        preprocessed_normalised_data, user_ids = preprocess(input, global_vars['min_max_values'])
        results = global_vars['predict'](preprocessed_normalised_data)['output_0'].numpy()
        postprocessed_results = postprocess(results, global_vars['min_max_values'])
        formatted_results = format_output(postprocessed_results, user_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return formatted_results

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "Healthy"}