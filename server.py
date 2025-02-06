from argparse import ArgumentParser
from pathlib import Path
from threading import Event, Lock
from typing import Annotated, Any, Generator
from warnings import simplefilter

simplefilter("ignore")

import uvicorn
from fastapi import FastAPI, Form, Request, UploadFile
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
parser.add_argument("-m", "--model-dir", type=Path, required=True)
parser.add_argument("-c", "--codec-dir", type=Path, required=True)
parser.add_argument("-v", "--voice-dir", type=Path, default="voices")
parser.add_argument("--cache", choices=["q4", "q6", "q8", "fp16"], default="fp16")
parser.add_argument("--device", default="cuda")
parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--sample-rate", type=int, default=16000)
args = parser.parse_args()

model = Model(
    model_dir=args.model_dir,
    codec_dir=args.codec_dir,
    voice_dir=args.voice_dir,
    cache=args.cache,
    device=args.device,
    dtype=args.dtype,
    max_seq_len=args.max_seq_len,
    sample_rate=args.sample_rate,
)

directory = Path(__file__).parent / "assets"
template = Jinja2Templates(directory)
lock = Lock()
event = None

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


@app.post("/interrupt")
def interrupt() -> None:
    global event
    event.set()


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


@app.post("/upload")
async def upload(
    audio: UploadFile, text: Annotated[str, Form(min_length=1)]
) -> list[str]:
    await model.cache_audio(audio, text)
    return sorted(model.voices)


@app.post("/generate")
def generate(query: Query) -> Response:
    global event

    with lock:
        event = Event()
        response = model.generate(query, event)
        return Response(response, media_type=f"audio/{query.format}")


@app.post("/stream")
def stream(query: Query) -> StreamingResponse:
    global event

    def generator() -> Generator[bytes, None, None]:
        for response in model.stream(query, event):
            yield response

    with lock:
        event = Event()
        return StreamingResponse(generator(), media_type=f"audio/{query.format}")


app.mount("/", StaticFiles(directory=directory))
uvicorn.run(app, host=args.host, port=args.port, access_log=False)
