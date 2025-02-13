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

from model import Model
from schema import Query

parser = ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8020)
parser.add_argument("--model", required=True)
parser.add_argument("--codec", default="hkustaudio/xcodec2")
parser.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
parser.add_argument("--voices", type=Path, default="voices")
parser.add_argument("--cache-mode", choices=["q4", "q6", "q8", "fp16"], default="fp16")
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--device", default="cuda")
parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
parser.add_argument("--loudness", type=float, default=-20.0)
parser.add_argument("--max-len", type=int, default=10)
parser.add_argument("--rebuild-cache", action="store_true")
parser.add_argument("--sample-rate", type=int, default=16000)
args = parser.parse_args()

model = Model(
    model=args.model,
    codec=args.codec,
    whisper=args.whisper,
    voices=args.voices,
    cache_mode=args.cache_mode,
    max_seq_len=args.max_seq_len,
    device=args.device,
    dtype=args.dtype,
    loudness=args.loudness,
    max_len=args.max_len,
    rebuild_cache=args.rebuild_cache,
    sample_rate=args.sample_rate,
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
def index(request: Request) -> Response:
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
    return await model.cache(file)


@app.get("/settings")
def settings() -> dict[str, Any]:
    settings = Query.defaults()
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
