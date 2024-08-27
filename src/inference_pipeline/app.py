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
import traceback
import sys

app = FastAPI()

global_vars = {
    'predict': None,
    'min_max_values': None
}

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
        print("shape of preprocessed_normalised_data", preprocessed_normalised_data.shape)
        results = global_vars['predict'](preprocessed_normalised_data)['output_0'].numpy()
        print("inference made, shape of results", results.shape, " now postprocessing")    
        postprocessed_results = postprocess(results, global_vars['min_max_values'])
        formatted_results = format_output(postprocessed_results, user_ids)
    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        lineno = tb.tb_lineno
        print(f"Error!! {str(e)} on line {lineno}")
        
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)} on line {lineno}")
    return formatted_results

@app.get("/health")
async def health_check():
    return {"status": "Healthy"}