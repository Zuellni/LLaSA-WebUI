import json
from io import BytesIO
from pathlib import Path
from threading import Event
from typing import Any, BinaryIO, Callable, Generator

import torch
import torchaudio
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer, attn
from exllamav2.generator import (
    ExLlamaV2DynamicGenerator,
    ExLlamaV2DynamicJob,
    ExLlamaV2Sampler,
)
from fastapi import UploadFile
from jinja2 import Template
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
        model_dir: Path,
        codec_dir: Path,
        voice_dir: Path,
        cache: str = "fp16",
        device: str = "cuda",
        dtype: str = "fp32",
        max_seq_len: int = 2048,
        sample_rate: int = 16000,
    ) -> None:
        self.device = device
        self.dtype = utils.dtype(dtype)
        self.max_seq_len = max_seq_len
        self.sample_rate = sample_rate

        self.model = self.load_model(model_dir, cache)
        self.codec = self.load_codec(codec_dir)

        self.voice_dir = voice_dir
        self.voice_cache = voice_dir / ".cache"
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

    def load_model(self, path: Path, cache: str) -> ExLlamaV2DynamicGenerator:
        config = ExLlamaV2Config(str(path))
        config.max_seq_len = self.max_seq_len
        model = ExLlamaV2(config, lazy_load=True)
        cache = utils.cache(cache)(model, lazy=True)
        paged = attn.has_flash_attn

        with Progress(f"Loading model{' ' * 4}", len(model.modules) + 1) as progress:
            model.load_autosplit(cache, callback=progress())

        with Progress("Loading tokenizer"):
            tokenizer = ExLlamaV2Tokenizer(config, lazy_init=True)

        return ExLlamaV2DynamicGenerator(model, cache, tokenizer, paged=paged)

    def load_codec(self, path: Path | str) -> XCodec2Model:
        with Progress(f"Loading codec{' ' * 4}"):
            codec = XCodec2Model.from_pretrained(path)
            return codec.eval().to(self.device, self.dtype)

    def load_voices(self) -> dict[str, dict[str, Any]]:
        suffixes = [f".{s}" for s in Query.formats()]
        files = [f for f in self.voice_dir.glob("*.*") if f.suffix in suffixes]
        voices = {}

        with Progress(f"Caching voices{' ' *  3}", len(files)) as progress:
            for file in files:
                self.cache_path(file)
                progress()

        for file in self.voice_cache.glob("*.json"):
            name = file.stem.lower()
            data = json.loads(file.read_text(encoding="utf-8"))
            voices[name] = {"audio": data["audio"], "text": data["text"]}

        return voices

    async def cache_audio(
        self, audio: UploadFile, text: str, rebuild: bool = True
    ) -> None:
        name = Path(audio.filename).stem.lower()
        file = self.voice_cache / f"{name}.json"

        if not file.exists() or rebuild:
            buffer = await audio.read()
            buffer = BytesIO(buffer)
            audio = self.encode(buffer)
            text = utils.process_text(text)

            file.write_text(
                json.dumps({"audio": audio, "text": text}), encoding="utf-8"
            )

        data = json.loads(file.read_text(encoding="utf-8"))
        self.voices[name] = {"audio": data["audio"], "text": data["text"]}

    def cache_path(self, path: Path, rebuild: bool = False) -> None:
        name = path.stem.lower()
        file = self.voice_cache / f"{name}.json"

        if not file.exists() or rebuild:
            text = self.voice_dir / f"{name}.txt"

            if not text.exists():
                return

            audio = self.encode(path)
            text = utils.process_text(text)

            file.write_text(
                json.dumps({"audio": audio, "text": text}), encoding="utf-8"
            )

    @autocast
    def encode(self, path: BinaryIO | Path) -> list[int]:
        audio, input_rate = torchaudio.load(path)
        audio = utils.process_audio(audio, input_rate, self.sample_rate)
        audio = self.codec.encode_code(audio, self.sample_rate)
        return audio[0, 0, :].tolist()

    @autocast
    def decode(self, input: list[str], output_rate: int, format: str) -> bytes:
        input = [int(i[4:-2]) for i in input if i]
        input = torch.tensor([[input]]).to(self.device)

        output = self.codec.decode_code(input)
        output = output[0, 0, :].unsqueeze(0)
        output = utils.process_audio(output, self.sample_rate, output_rate)

        buffer = BytesIO()
        torchaudio.save(buffer, output.cpu(), output_rate, format=format)
        return buffer.getvalue()

    def __call__(self, query: Query) -> Generator[list[str], None, None]:
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

                input = self.template.render(messages=messages)[:-10]
                input_ids = self.model.tokenizer.encode(input, add_bos=True)
                max_new_tokens = self.max_seq_len - input_ids.shape[-1]

                if max_new_tokens <= 0:
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

                if not output:
                    continue

                if query.reuse:
                    audio = "".join(output)
                    transcript = chunk

                yield output

        timer(f"Finished with seed {query.seed}")

    def generate(self, query: Query, event: Event) -> bytes | None:
        outputs = []

        for output in self(query):
            outputs.extend(output)

            if event.is_set():
                break

        if outputs:
            return self.decode(outputs, query.rate, query.format)

    def stream(self, query: Query, event: Event) -> Generator[bytes, None, None]:
        for output in self(query):
            yield self.decode(output, query.rate, query.format)

            if event.is_set():
                break
