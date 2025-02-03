from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Generator, get_args

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic_core import PydanticUndefined

from model import Model
from schema import Query

app = FastAPI(docs_url="/", redoc_url=None)
app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
    allow_origins=["*"],
)


@app.get("/settings")
def get_settings() -> dict[str, Any]:
    settings = {
        k: v.default
        for k, v in Query.model_fields.items()
        if v.default is not PydanticUndefined
    }

    settings["audio"] = list(model.audio)
    settings["formats"] = list(get_args(Query.model_fields.get("format").annotation))
    return settings


@app.post("/generate")
def generate_audio(query: Query) -> Response:
    response = model.generate_audio(query)
    return Response(response, media_type=f"audio/{query.format}")


@app.post("/stream")
def stream_audio(query: Query) -> StreamingResponse:
    def generator() -> Generator[bytes, None, None]:
        for response in model.stream_audio(query):
            yield response

    return StreamingResponse(generator(), media_type=f"audio/{query.format}")


parser = ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8020)
parser.add_argument("-m", "--model", type=Path, required=True)
parser.add_argument("-c", "--codec", type=Path, required=True)
parser.add_argument("-a", "--audio", type=Path, default="audio")
parser.add_argument("--cache-bits", type=int, choices=[4, 6, 8, 16], default=16)
parser.add_argument("--device", default="cuda")
parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--sample-rate", type=int, default=16000)
args = parser.parse_args()

model = Model(
    model=args.model,
    codec=args.codec,
    audio=args.audio,
    cache_bits=args.cache_bits,
    device=args.device,
    dtype=args.dtype,
    max_seq_len=args.max_seq_len,
    sample_rate=args.sample_rate,
)

uvicorn.run(app, host=args.host, port=args.port)
