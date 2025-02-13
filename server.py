from argparse import ArgumentParser
from pathlib import Path
from threading import Event, Lock
from typing import Any, Generator

import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic_core import PydanticUndefined
from starlette.templating import _TemplateResponse

from model import Model
from schema import Query

parser = ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8020)
parser.add_argument("-m", "--model-path", type=Path, required=True)
parser.add_argument("-c", "--codec-path", default="hkustaudio/xcodec2")
parser.add_argument("-w", "--whisper-path", default="openai/whisper-large-v3-turbo")
parser.add_argument("-v", "--voice-path", type=Path, default="voices")
parser.add_argument("--cache", choices=["q4", "q6", "q8", "fp16"], default="fp16")
parser.add_argument("--device", default="cuda")
parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--sample-rate", type=int, default=16000)
parser.add_argument("--voice_loudness", type=float, default=-20.0)
parser.add_argument("--voice_max_len", type=int, default=15)
parser.add_argument("--rebuild", action="store_true")
args = parser.parse_args()

model = Model(
    model_path=args.model_path,
    codec_path=args.codec_path,
    whisper_path=args.whisper_path,
    voice_path=args.voice_path,
    cache=args.cache,
    device=args.device,
    dtype=args.dtype,
    max_seq_len=args.max_seq_len,
    sample_rate=args.sample_rate,
    voice_loudness=args.voice_loudness,
    voice_max_len=args.voice_max_len,
    rebuild=args.rebuild,
)

directory = Path(__file__).parent / "assets"
template = Jinja2Templates(directory)
abort_event = None
lock = Lock()

app = FastAPI()
app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
    allow_origins=["*"],
)


@app.get("/")
def index(request: Request) -> _TemplateResponse:
    return template.TemplateResponse(
        name="index.html",
        context={"request": request, "settings": settings()},
    )


@app.post("/abort")
def abort() -> None:
    global abort_event
    abort_event.set()


@app.post("/cache")
async def cache(file: UploadFile) -> list[str]:
    await model.cache(file)
    return sorted(model.voices)


@app.get("/settings")
def settings() -> dict[str, Any]:
    settings = {
        k: v.default
        for k, v in Query.model_fields.items()
        if v.default is not PydanticUndefined
    }

    settings["formats"] = Query.formats()
    settings["voices"] = sorted(model.voices)
    return settings


@app.post("/generate")
def generate(query: Query) -> Response:
    global abort_event

    with lock:
        abort_event = Event()
        response = model.generate(query, abort_event)
        return Response(response, media_type=f"audio/{query.format}")


@app.post("/stream")
def stream(query: Query) -> StreamingResponse:
    global abort_event

    def generator() -> Generator[bytes, None, None]:
        for response in model.stream(query, abort_event):
            yield response

    with lock:
        abort_event = Event()
        return StreamingResponse(generator(), media_type=f"audio/{query.format}")


app.mount("/", StaticFiles(directory=directory))
uvicorn.run(app, host=args.host, port=args.port, access_log=False)
