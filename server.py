from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Generator, get_args

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic_core import PydanticUndefined

from model import Model
from schema import Query

parser = ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8020)
parser.add_argument("-m", "--model_dir", type=Path, required=True)
parser.add_argument("-c", "--codec_dir", type=Path, required=True)
parser.add_argument("-v", "--voice_dir", type=Path, default="voices")
parser.add_argument("--cache-bits", type=int, choices=[4, 6, 8, 16], default=16)
parser.add_argument("--device", default="cuda")
parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--sample-rate", type=int, default=16000)
args = parser.parse_args()

model = Model(
    model_dir=args.model_dir,
    codec_dir=args.codec_dir,
    voice_dir=args.voice_dir,
    cache_bits=args.cache_bits,
    device=args.device,
    dtype=args.dtype,
    max_seq_len=args.max_seq_len,
    sample_rate=args.sample_rate,
)

directory = Path(__file__).parent / "assets"
template = Jinja2Templates(directory)

app = FastAPI()
app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
    allow_origins=["*"],
)


@app.get("/")
def index(request: Request):
    return template.TemplateResponse(
        name="index.html",
        context={"request": request, "settings": settings()},
    )


@app.get("/settings")
def settings() -> dict[str, Any]:
    settings = {
        k: v.default
        for k, v in Query.model_fields.items()
        if v.default is not PydanticUndefined
    }

    settings["formats"] = list(get_args(Query.model_fields["format"].annotation))
    settings["voices"] = list(model.voices)
    return settings


@app.post("/generate")
def generate(query: Query) -> Response:
    response = model.generate(query)
    return Response(response, media_type=f"audio/{query.format}")


@app.post("/stream")
def stream(query: Query) -> StreamingResponse:
    def generator() -> Generator[bytes, None, None]:
        for response in model.stream(query):
            yield response

    return StreamingResponse(generator(), media_type=f"audio/{query.format}")


app.mount("/", StaticFiles(directory=directory))
uvicorn.run(app, host=args.host, port=args.port)
