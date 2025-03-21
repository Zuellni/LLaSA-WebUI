import json
from io import BytesIO
from pathlib import Path
from threading import Event
from typing import Any, Generator

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
from huggingface_hub import snapshot_download
from jinja2 import Template
from transformers import Pipeline, pipeline
from xcodec2.modeling_xcodec2 import XCodec2Model

import utils
from schema import Query
from utils import Progress, Timer


class Model:
    def __init__(
        self,
        model: str,
        codec: str,
        whisper: str,
        voices: Path,
        cache: str = "fp16",
        max_seq_len: int = 2048,
        device: str = "cuda",
        dtype: str = "fp32",
        max_voice_len: int = 15,
        rebuild_cache: bool = False,
        sample_rate: int = 16000,
    ) -> None:
        utils.log("Starting [[cyan]LLaSA WebUI[/cyan]]")

        self.cache = utils.get_cache(cache)
        self.max_seq_len = max_seq_len

        self.device = device
        self.dtype = utils.get_dtype(dtype)

        self.max_voice_len = max_voice_len
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
        self.gen_settings = ExLlamaV2Sampler.Settings.greedy()
        self.gen_settings.allow_tokens(
            tokenizer=self.model.tokenizer,
            tokens=[eos] + list(range(first, last + 1)),
        )

    def load_model(self, path: str) -> ExLlamaV2DynamicGenerator:
        if not Path(path).is_dir():
            path = snapshot_download(path)

        config = ExLlamaV2Config(path)
        config.max_seq_len = self.max_seq_len

        model = ExLlamaV2(config, lazy_load=True)
        cache = self.cache(model, lazy=True)
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

    def load_whisper(self, path: str | None = None) -> Pipeline:
        if hasattr(self, "whisper") and self.whisper:
            self.whisper.model.to(self.device)
            return self.whisper

        with Progress("Loading whisper"):
            whisper = pipeline(
                task="automatic-speech-recognition",
                model=path,
                device=self.device,
                torch_dtype=self.dtype,
            )

            whisper.model.cpu()
            torch.cuda.empty_cache()
            return whisper

    def offload_whisper(self) -> None:
        if hasattr(self, "whisper") and self.whisper:
            self.whisper.model.cpu()
            torch.cuda.empty_cache()

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

    async def cache_voice(self, file: UploadFile) -> list[str]:
        buffer = await file.read()
        buffer = BytesIO(buffer)
        name, data = self.encode(buffer, file.filename, rebuild_cache=True)
        self.voices[name] = data
        return sorted(self.voices)

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

        audio, sample_rate = torchaudio.load(audio)
        audio = audio.to(self.device)

        audio = utils.process_audio(
            audio=audio,
            input_rate=sample_rate,
            output_rate=self.sample_rate,
            max_len=self.max_voice_len,
        )

        if not text:
            text = self.voice_dir / f"{name}.txt"

            if not text.exists():
                self.load_whisper()
                text = self.whisper(audio[0].cpu().numpy())["text"]

        text = utils.process_text(text)

        with torch.autocast(self.device, self.dtype), torch.inference_mode():
            audio = self.codec.encode_code(audio, self.sample_rate)
            audio = audio[0, 0, :].tolist()

        file.write_text(
            json.dumps({"audio": audio, "text": text}, ensure_ascii=False),
            encoding="utf-8",
        )

        self.offload_whisper()
        return name, {"audio": audio, "text": text}

    def decode(self, audio: list[str]) -> torch.Tensor:
        audio = [int(a[4:-2]) for a in audio if a]
        audio = torch.tensor([[audio]], device=self.device)

        with torch.autocast(self.device, self.dtype), torch.inference_mode():
            audio = self.codec.decode_code(audio)
            return audio[0, 0, :].unsqueeze(0)

    def get_bytes(self, audio: torch.Tensor, sample_rate: int, format: str) -> bytes:
        audio = utils.process_audio(audio, self.sample_rate, sample_rate)
        buffer = BytesIO()
        torchaudio.save(buffer, audio.cpu(), sample_rate, format=format)
        return buffer.getvalue()

    def get_voice(self, voice: str) -> tuple[str, str]:
        voice = self.voices.get(voice.lower(), {})
        audio = voice.get("audio", [])
        audio = "".join([f"<|s_{a}|>" for a in audio])
        text = voice.get("text", "")
        text += " " if text else ""
        return audio, text

    def __call__(
        self, query: Query, abort_event: Event
    ) -> Generator[list[str], None, None]:
        self.gen_settings.temperature = query.temperature
        self.gen_settings.token_repetition_penalty = query.penalty
        self.gen_settings.top_k = query.top_k
        self.gen_settings.top_p = query.top_p

        text = utils.clean_text(query.input)
        pairs = utils.get_pairs(text, query.voice)
        chunks = []

        for pair in pairs:
            split = utils.split_text(pair["text"], query.max_len)
            chunks.extend([{"voice": pair["voice"], "text": s} for s in split])

        count = len(chunks)
        digits = len(str(count))
        tokens = 0

        with Timer() as timer:
            for index, chunk in enumerate(chunks):
                if not query.reuse or index == 0 or audio != chunk["voice"]:
                    audio, text = self.get_voice(chunk["voice"])

                utils.log(chunk["text"])

                messages = [
                    {
                        "role": "user",
                        "content": (
                            "Convert the text to speech:"
                            "<|TEXT_UNDERSTANDING_START|>"
                            f"{text}{chunk['text']}"
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
                    text = chunk["text"]

                yield output

            self.model.clear_queue()

        timer(f"Generated {tokens / 50:.2f} seconds of audio with seed {query.seed}")

    def generate(self, query: Query, abort_event: Event) -> bytes | None:
        outputs = []

        for output in self(query, abort_event):
            if query.join:
                outputs.extend(output)
            else:
                outputs.append(output)

        if not outputs:
            return

        if query.join:
            outputs = [outputs]

        outputs = [self.decode(o) for o in outputs]
        output = torch.cat(outputs, dim=-1)
        return self.get_bytes(output, query.rate, query.format)

    def stream(self, query: Query, abort_event: Event) -> Generator[bytes, None, None]:
        for output in self(query, abort_event):
            output = self.decode(output)
            yield self.get_bytes(output, query.rate, query.format)
