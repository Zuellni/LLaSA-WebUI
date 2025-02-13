import contextlib
import json
from io import BytesIO
from pathlib import Path
from threading import Event
from typing import Any, Callable, Generator
from warnings import simplefilter

simplefilter("ignore")

import torch
import torchaudio
import transformers

transformers.logging.set_verbosity_error()

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer, attn
from exllamav2.generator import (
    ExLlamaV2DynamicGenerator,
    ExLlamaV2DynamicJob,
    ExLlamaV2Sampler,
)
from fastapi import UploadFile
from huggingface_hub import hf_hub_download
from jinja2 import Template
from transformers import Pipeline, pipeline

with contextlib.redirect_stdout(None):
    from xcodec2.modeling_xcodec2 import XCodec2Model

import utils
from schema import Query
from utils import Progress, Timer


class Model:
    @staticmethod
    def autocast(func) -> Callable[..., Any]:
        def wrapper(self, *args, **kwargs) -> Any:
            with torch.autocast(self.device, self.dtype), torch.inference_mode():
                return func(self, *args, **kwargs)

        return wrapper

    def __init__(
        self,
        model: str,
        codec: str,
        whisper: str,
        voices: Path,
        cache_mode: str = "fp16",
        max_seq_len: int = 2048,
        device: str = "cuda",
        dtype: str = "fp32",
        loudness: float = -20.0,
        max_len: int = 10,
        rebuild_cache: bool = False,
        sample_rate: int = 16000,
    ) -> None:
        self.cache_mode = utils.cache(cache_mode)
        self.max_seq_len = max_seq_len

        self.device = device
        self.dtype = utils.dtype(dtype)

        self.loudness = loudness
        self.max_len = max_len
        self.rebuild_cache = rebuild_cache
        self.sample_rate = sample_rate

        self.model = self.load_model(model)
        self.codec = self.load_codec(codec)
        self.whisper = self.load_whisper(whisper)

        self.voice_dir = voices
        self.voice_cache = voices / ".cache"
        self.voice_cache.mkdir(parents=True, exist_ok=True)
        self.voices = self.load_voices()

        self.template = Template(
            self.model.tokenizer.tokenizer_config_dict.get("chat_template", "")
        )

        eos = self.model.tokenizer.single_id("<|SPEECH_GENERATION_END|>")
        first = self.model.tokenizer.single_id("<|s_0|>")
        last = self.model.tokenizer.single_id("<|s_65535|>")

        self.stop_conditions = [eos]
        self.gen_settings = ExLlamaV2Sampler.Settings()
        self.gen_settings.allow_tokens(
            tokenizer=self.model.tokenizer,
            tokens=[eos] + list(range(first, last + 1)),
        )

    def load_model(self, path: str) -> ExLlamaV2DynamicGenerator:
        if not Path(path).is_dir():
            path = hf_hub_download(path)

        config = ExLlamaV2Config(path)
        config.max_seq_len = self.max_seq_len

        model = ExLlamaV2(config, lazy_load=True)
        cache = self.cache_mode(model, lazy=True)
        paged = attn.has_flash_attn

        with Progress("Loading model", len(model.modules) + 1) as progress:
            model.load_autosplit(cache, callback=progress)

        with Progress("Loading tokenizer"):
            tokenizer = ExLlamaV2Tokenizer(config, lazy_init=True)

        return ExLlamaV2DynamicGenerator(model, cache, tokenizer, paged=paged)

    def load_codec(self, path: str) -> XCodec2Model:
        with Progress("Loading codec"):
            codec = XCodec2Model.from_pretrained(path)
            return codec.eval().to(self.device, self.dtype)

    def load_whisper(self, path: str) -> Pipeline:
        with Progress("Loading whisper"):
            return pipeline(
                task="automatic-speech-recognition",
                model=path,
                device=self.device,
                torch_dtype=self.dtype,
            )

    def load_voices(self) -> dict[str, dict[str, Any]]:
        suffixes = [f".{s}" for s in Query.formats()]
        files = [f for f in self.voice_dir.glob("*.*") if f.suffix in suffixes]
        voices = {}

        with Progress("Caching voices", len(files)) as progress:
            for file in files:
                self.encode(file, rebuild_cache=self.rebuild_cache)
                progress()

        for file in self.voice_cache.glob("*.json"):
            name = file.stem.lower()
            voices[name] = json.loads(file.read_text(encoding="utf-8"))

        return voices

    async def cache(self, file: UploadFile) -> list[str]:
        buffer = await file.read()
        buffer = BytesIO(buffer)
        name, data = self.encode(buffer, file.filename, rebuild_cache=True)
        self.voices[name] = data
        return sorted(self.voices)

    @autocast
    def encode(
        self,
        audio: BytesIO | Path,
        name: str = "",
        text: str = "",
        rebuild_cache: bool = False,
    ) -> tuple[str, dict[str, Any] | None]:
        name = Path(name if name else audio).stem.lower()
        file = self.voice_cache / f"{name}.json"

        if file.exists() and not rebuild_cache:
            return name, None

        audio, input_rate = torchaudio.load(audio)

        audio = utils.process_audio(
            audio=audio,
            input_rate=input_rate,
            output_rate=self.sample_rate,
            max_len=self.max_len,
            output_loudness=self.loudness,
        )

        if not text:
            text = self.voice_dir / f"{name}.txt"

            if not text.exists():
                text = self.whisper(audio[0].numpy())["text"]

        text = utils.process_text(text)
        audio = self.codec.encode_code(audio, self.sample_rate)
        audio = audio[0, 0, :].tolist()

        file.write_text(
            json.dumps({"audio": audio, "text": text}, ensure_ascii=False),
            encoding="utf-8",
        )

        torch.cuda.empty_cache()
        return name, {"audio": audio, "text": text}

    @autocast
    def decode(self, input: list[str], output_rate: int, format: str) -> bytes:
        input = [int(i[4:-2]) for i in input if i]
        input = torch.tensor([[input]]).to(self.device)

        output = self.codec.decode_code(input)
        output = output[0, 0, :].unsqueeze(0).float().cpu()

        output = utils.process_audio(
            audio=output,
            input_rate=self.sample_rate,
            output_rate=output_rate,
            output_loudness=self.loudness,
        )

        buffer = BytesIO()
        torchaudio.save(buffer, output, output_rate, format=format)
        return buffer.getvalue()

    def __call__(
        self, query: Query, abort_event: Event
    ) -> Generator[list[str], None, None]:
        self.gen_settings.temperature = query.temperature
        self.gen_settings.token_repetition_penalty = query.penalty
        self.gen_settings.top_k = query.top_k
        self.gen_settings.top_p = query.top_p

        voice = self.voices.get(query.voice.lower(), {})
        audio = voice.get("audio", [])
        audio = "".join([f"<|s_{a}|>" for a in audio])

        transcript = voice.get("text", "")
        transcript += " " if transcript else ""

        text = utils.clean_text(query.input)
        chunks = utils.split_text(text, query.chunk)
        count = len(chunks)
        digits = len(str(count))
        tokens = 0

        with Timer() as timer:
            for index, chunk in enumerate(chunks):
                messages = [
                    {
                        "role": "user",
                        "content": (
                            "Convert the text to speech:"
                            "<|TEXT_UNDERSTANDING_START|>"
                            f"{transcript}{chunk}"
                            "<|TEXT_UNDERSTANDING_END|>"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": f"<|SPEECH_GENERATION_START|>{audio}",
                    },
                ]

                input = self.template.render(messages=messages)
                input_ids = self.model.tokenizer.encode(input, add_bos=True)[:, :-1]
                max_new_tokens = self.max_seq_len - input_ids.shape[-1]

                if max_new_tokens < 2:
                    continue

                job = ExLlamaV2DynamicJob(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=2,
                    gen_settings=self.gen_settings,
                    seed=query.seed,
                    stop_conditions=self.stop_conditions,
                )

                self.model.enqueue(job)
                output = []

                with Progress(
                    f"Generating chunk [bright_cyan]{index + 1:0{digits}}/"
                    f"{count}[/bright_cyan]",
                    max_new_tokens,
                ) as progress:
                    while self.model.num_remaining_jobs():
                        for result in self.model.iterate():
                            text = result.get("text")
                            progress()

                            if text:
                                output.append(text)
                                tokens += 1

                        if abort_event and abort_event.is_set():
                            self.model.clear_queue()
                            return

                if not output:
                    continue

                if query.reuse:
                    audio = "".join(output)
                    transcript = chunk

                yield output

            self.model.clear_queue()

        timer(f"Generated {tokens / 50:.2f} seconds of audio with seed {query.seed}")

    def generate(self, query: Query, abort_event: Event) -> bytes | None:
        outputs = []

        for output in self(query, abort_event):
            outputs.extend(output)

        if outputs:
            return self.decode(outputs, query.rate, query.format)

    def stream(self, query: Query, abort_event: Event) -> Generator[bytes, None, None]:
        for output in self(query, abort_event):
            yield self.decode(output, query.rate, query.format)
