from __future__ import annotations
import logging
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.params import Query
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from asyncio import Lock, create_task
from typing import Dict
from mappers import Mapper
from neural_net_model import NeuralNetworkModel


app = FastAPI(
    title="Neural Network Model API v2",
    description="API to create, serialize, output, evaluate, generate, train and diagnose of neural network models.",
    version="0.2.0"
)

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

log = logging.getLogger(__name__)

# Constants for examples
EXAMPLES = [
    {
        "input":  [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "target": 4
    },
    {
        "input":  [0, 0, 0, 0, 2, 0, 0, 0, 0],
        "target": 0
    },
    {
        "input":  [1, 0, 0, 0, 2, 0, 0, 0, 0],
        "target": 1
    },
    {
        "input":  [1, 2, 0, 0, 2, 0, 0, 0, 0],
        "target": 7
    },
    {
        "input":  [1, 2, 0, 0, 2, 0, 0, 1, 0],
        "target": 2
    },
    {
        "input":  [1, 2, 2, 0, 2, 0, 0, 1, 0],
        "target": 6
    },
    {
        "input":  [1, 2, 2, 0, 2, 0, 1, 1, 0],
        "target": 8
    },
    {
        "input":  [1, 2, 2, 0, 2, 0, 1, 1, 2],
        "target": 3
    },
    {
        "input":  [1, 2, 2, 1, 2, 0, 1, 1, 2],
        "target": 5
    },
]


class ModelRequest(BaseModel):
    model_id: str = Field(
        ...,
        examples=["test"],
        description="The unique identifier for the model."
    )


class CreateModelRequest(ModelRequest):
    layers: list[dict] = Field(
        ...,
        examples=[
            [{"embedding": {"num_embeddings": 9, "embedding_dim": 2}}, {"flatten": {}},
             {"linear": {"in_features": 18, "out_features": 9, "bias": False},
              "xavier_uniform": {"gain": 0.7}},
             {"batchnorm1d": {"num_features": 9}},
             {"tanh": {}},
             {"linear": {"in_features": 9, "out_features": 9},
              "kaiming_uniform": {"nonlinearity": "tanh"},
              "confidence": 0.5},
             {"dropout": {"p": 0.5}},
             {"tanh": {}},
             {"linear": {"in_features": 9, "out_features": 9, "bias": False}},
             {"softmax": {"dim": -1}},
            ],
        ],
        description="List of dictionaries where the key is the PyTorch nn or init algorithm and the value is its args."
    )
    optimizer: dict = Field(
        ...,
        examples=[
            {"adam": {"lr": 0.01, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.01}},
            {"adamw": {"lr": 0.01, "weight_decay": 0.1}},
            {"sgd": {"lr": 0.1, "weight_decay": 0.0}}
        ],
        description="A dictionary where the key is the PyTorch optimizer name and the value is its args."
    )


class OutputRequest(ModelRequest):
    input: list = Field(
        ...,
        examples=[[example["input"] for example in EXAMPLES] * 20,],
        description="The input data"
    )
    target: list | int | None = Field(
        None,
        description="The expected target data (Optional)"
    )


class EvaluateRequest(OutputRequest):
    target: list = Field(
        ...,
        examples=[[example["target"] for example in EXAMPLES] * 20,],
        description="The expected target data"
    )
    epochs: int = Field(
        1,
        examples=[10],
        description="The number of evaluation epochs"
    )
    batch_size: int | None = Field(
        None,
        examples=[32],
        description="The batch size to sample each epoch (Optional)"
    )


class GenerateRequest(ModelRequest):
    input: list = Field(
        ...,
        examples=[[[0] * 9]],
        description="The initial input context"
    )
    block_size: int = Field(
        ...,
        examples=[9],
        description="The block size of context"
    )
    max_new_tokens: int = Field(
        ...,
        examples=[10],
        description="The maximum number of tokens to generate"
    )


class TrainingRequest(EvaluateRequest):
    epochs: int = Field(
        10,
        examples=[10],
        description="The number of training epochs"
    )


class ModelIdQuery(Query):
    description="The unique identifier for the model"


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, e: Exception):
    log.error(f"An error occurred: {str(e)}")
    return JSONResponse(status_code=500, content={"detail": "Please refer to server logs"})

@app.exception_handler(KeyError)
async def key_error_handler(_: Request, e: KeyError):
    raise HTTPException(status_code=404, detail=f"Not found error occurred: {str(e)}")

@app.exception_handler(ValueError)
async def value_error_handler(_: Request, e: ValueError):
    raise HTTPException(status_code=400, detail=f"Value error occurred: {str(e)}")

@app.get("/", include_in_schema=False)
def redirect_to_dashboard():
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")

@app.post("/model/")
def create_model(body: CreateModelRequest = Body(...)):
    model_id = body.model_id
    log.info(f"Requesting creation of model {model_id}")
    model = NeuralNetworkModel(model_id, Mapper(body.layers, body.optimizer))
    model.serialize()
    return {"message": f"Model {model_id} created and saved successfully"}

@app.post("/output/")
def compute_model_output(body: OutputRequest =
                         Body(...,
                              openapi_examples={f"example_{idx}": {
                                 "summary": f"Example {idx + 1}",
                                 "description": f"Example input and training data for case {idx + 1}",
                                 "value": {
                                     "model_id": "test",
                                     "input": [example["input"]] * 2,
                                     "target": [example["target"]] * 2,
                                 }
                             } for idx, example in enumerate(EXAMPLES)} )):
    model_id = body.model_id
    log.info(f"Requesting output for model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)
    output, cost = model.compute_output(body.input, body.target)
    return {"output": output,
            "cost": cost,
            }

@app.post("/evaluate/")
def evaluate_model(body: EvaluateRequest =
                         Body(...,
                              openapi_examples={f"example_{idx}": {
                                 "summary": f"Example {idx + 1}",
                                 "description": f"Example input and training data for case {idx + 1}",
                                 "value": {
                                     "model_id": "test",
                                     "input": [example["input"]] * 6,
                                     "target": [example["target"]] * 6,
                                     "epochs": 2,
                                     "batch_size": 3,
                                 }
                             } for idx, example in enumerate(EXAMPLES)} )):
    model_id = body.model_id
    log.info(f"Requesting evaluation of model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)
    cost = model.evaluate_model(body.input, body.target, body.epochs, body.batch_size)
    return {"cost": cost}


@app.post("/generate/")
def model_generate(body: GenerateRequest = Body(...)):
    model_id = body.model_id
    log.info(f"Generating tokens using model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)
    generated_tokens = model.generate_tokens(body.input, body.block_size, body.max_new_tokens)
    return {"tokens": generated_tokens}


# This will track active training sessions by model_id
model_locks: Dict[str, Lock] = {}

@app.put("/train/")
async def train_model(body: TrainingRequest = Body(...)):
    model_id = body.model_id
    log.info(f"Requesting training for model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)

    # Get or create a lock for this model
    if model_id not in model_locks:
        model_locks[model_id] = Lock()
    lock = model_locks[model_id]

    # If the model is already locked (training), return 409 Conflict
    if lock.locked():
        raise HTTPException(status_code=409, detail=f"Training already in progress for model {model_id}.")

    async def train():
        async with lock:
            await run_in_threadpool(
                model.train_model,body.input, body.target, body.epochs, body.batch_size
            )

    # Start training in the background
    create_task(train())

    # Respond with request accepted
    return JSONResponse(content={"message": f"Training for model {model_id} started asynchronously."}, status_code=202)

@app.get("/progress/")
def model_progress(model_id: str = ModelIdQuery(...)):
    log.info(f"Requesting progress for model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)
    return {
        "progress": model.progress,
        "average_cost": model.avg_cost,
        "average_cost_history": model.avg_cost_history,
        "status": model.status,
    }

@app.get("/stats/")
def model_stats(model_id: str = ModelIdQuery(...)):
    log.info(f"Requesting stats for model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)
    return model.stats

@app.delete("/model/")
def delete_model(model_id: str = ModelIdQuery(...)):
    log.info(f"Requesting deletion of model {model_id}")
    NeuralNetworkModel.delete(model_id)
    return Response(status_code=204)


if __name__ == "__main__": # pragma: no cover
    import uvicorn
    import json

    with open("log_config.json", "r") as f:
        log_config = json.load(f)

    uvicorn.run(app, host="127.0.0.1", port=8000, log_config=log_config)
