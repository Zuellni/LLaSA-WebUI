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
server_group = parser.add_argument_group("Server")
server_group.add_argument("-H", "--host", default="127.0.0.1")
server_group.add_argument("-P", "--port", default=8020, type=int)

path_group = parser.add_argument_group("Paths")
path_group.add_argument("-m", "--model", required=True)
path_group.add_argument("-c", "--codec", default="annuvin/xcodec2-fp32")
path_group.add_argument("-w", "--whisper", default="openai/whisper-large-v3-turbo")
path_group.add_argument("-v", "--voices", default="voices", type=Path)

model_group = parser.add_argument_group("Models")
model_group.add_argument("--cache", choices=["q4", "q6", "q8", "fp16"], default="fp16")
model_group.add_argument("--max-seq-len", default=2048, type=int)
model_group.add_argument("--device", default="cuda")
model_group.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")

voice_group = parser.add_argument_group("Voices")
voice_group.add_argument("--max-voice-len", default=15, type=int)
voice_group.add_argument("--rebuild-cache", action="store_true")
voice_group.add_argument("--sample-rate", default=16000, type=int)
args = parser.parse_args()

model = Model(
    model=args.model,
    codec=args.codec,
    whisper=args.whisper,
    voices=args.voices,
    cache=args.cache,
    max_seq_len=args.max_seq_len,
    device=args.device,
    dtype=args.dtype,
    max_voice_len=args.max_voice_len,
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
    return await model.cache_voice(file)


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
