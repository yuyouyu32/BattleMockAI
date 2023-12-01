import argparse
import json
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse, PlainTextResponse
import uvicorn
from .inference_batch import Predictor

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
bot = Predictor(process_num=10)

app = FastAPI()
@app.post("/predict")
async def predict(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    #try:
    #    frames_json = await request.json()
    #    action_json = bot.predict(frames_json)
    #    return JSONResponse(action_json)
    #except Exception as e:
    #    err_msg = "server error! {}".format(str(e))
    #    print(err_msg)
    #    return PlainTextResponse(err_msg)
    frames_json = await request.json()
    action_json = bot.predict(frames_json)
    return JSONResponse(action_json)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("python server.py IP PORT")
    IP = sys.argv[1]
    Port = int(sys.argv[2])
    uvicorn.run(app,
                host=IP,
                port=Port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
