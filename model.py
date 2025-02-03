import json
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Generator

import torch
import torchaudio
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2DynamicGenerator,
    ExLlamaV2DynamicJob,
    ExLlamaV2Sampler,
)
from jinja2 import Template
from xcodec2.modeling_xcodec2 import XCodec2Model

import utils
from schema import Query
from utils import Timer


class Model:
    @staticmethod
    def autocast(func) -> Callable[..., Any]:
        def wrapper(self, *args, **kwargs) -> Any:
            with torch.autocast(self.device, self.dtype), torch.inference_mode():
                return func(self, *args, **kwargs)

        return wrapper

    def get_cache(self, cache_bits: int) -> ExLlamaV2Cache:
        match (cache_bits):
            case 4:
                return ExLlamaV2Cache_Q4
            case 6:
                return ExLlamaV2Cache_Q6
            case 8:
                return ExLlamaV2Cache_Q8
            case _:
                return ExLlamaV2Cache

    def get_dtype(self, dtype: str) -> torch.dtype:
        match (dtype):
            case "fp16":
                return torch.float16
            case "bf16":
                return torch.bfloat16
            case _:
                return torch.float32

    def __init__(
        self,
        model_dir: Path,
        codec_dir: Path,
        voice_dir: Path,
        cache_bits: int = 16,
        device: str = "cuda",
        dtype: str = "fp32",
        max_seq_len: int = 2048,
        sample_rate: int = 16000,
    ) -> None:
        self.device = device
        self.dtype = self.get_dtype(dtype)
        self.max_seq_len = max_seq_len
        self.sample_rate = sample_rate

        self.model = self.load_model(model_dir, cache_bits)
        self.codec = self.load_codec(codec_dir)
        self.voices = self.load_voices(voice_dir)

        template = self.model.tokenizer.tokenizer_config_dict.get("chat_template", "")
        self.template = Template(template)

        eos = self.model.tokenizer.single_id("<|SPEECH_GENERATION_END|>")
        first = self.model.tokenizer.single_id("<|s_0|>")
        last = self.model.tokenizer.single_id("<|s_65535|>")

        self.stop_conditions = [eos]

        self.gen_settings = ExLlamaV2Sampler.Settings()
        self.gen_settings.allow_tokens(
            tokenizer=self.model.tokenizer,
            tokens=self.stop_conditions + list(range(first, last + 1)),
        )

    def load_model(self, path: Path, cache_bits: int) -> ExLlamaV2DynamicGenerator:
        with Timer() as timer:
            config = ExLlamaV2Config(str(path))
            config.max_seq_len = self.max_seq_len

            model = ExLlamaV2(config, lazy_load=True)
            cache = self.get_cache(cache_bits)(model, lazy=True)
            model.load_autosplit(cache)

            tokenizer = ExLlamaV2Tokenizer(config, lazy_init=True)
            generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)
            generator.warmup()

        timer("Loaded model")
        return generator

    def load_codec(self, path: Path) -> XCodec2Model:
        with Timer() as timer:
            codec = XCodec2Model.from_pretrained(path)
            codec = codec.eval().to(self.device, self.dtype)

        timer("Loaded codec")
        return codec

    def load_voices(
        self, path: Path, suffixes: list[str] = [".flac", ".mp3", ".ogg", ".wav"]
    ) -> dict[str, dict[str, str]]:
        with Timer() as timer:
            files = [f for f in path.glob("*.*") if f.suffix in suffixes]
            voices = dict([self.encode(f) for f in files])

        timer("Cached voices")
        return voices

    @autocast
    def encode(
        self, path: Path, cache_dir: str = ".cache"
    ) -> tuple[str, dict[str, str]] | None:
        name = path.stem.lower()
        file = path.parent / cache_dir / f"{name}.json"
        file.parent.mkdir(parents=True, exist_ok=True)

        if not file.exists():
            text = path.parent / f"{name}.txt"

            if not text.exists():
                return

            text = utils.process_text(text)
            speech, sample_rate = torchaudio.load(path)
            speech = utils.process_audio(speech, sample_rate, self.sample_rate)
            speech = self.codec.encode_code(speech, self.sample_rate)
            speech = speech[0, 0, :]
            speech = [f"<|s_{s}|>" for s in speech]
            speech = "".join(speech)

            file.write_text(
                json.dumps({"speech": speech, "text": text}), encoding="utf-8"
            )

        data = json.loads(file.read_text(encoding="utf-8"))
        return name, {"speech": data["speech"], "text": data["text"]}

    def __call__(self, query: Query) -> Generator[list[str], None, None]:
        voice = self.voices.get(query.voice, {})
        ref_speech = voice.get("speech", "")
        ref_text = voice.get("text", "")
        ref_text = f"{ref_text} " if ref_text else ""
        text = utils.clean_text(query.input)

        self.gen_settings.temperature = query.temperature
        self.gen_settings.token_repetition_penalty = query.repetition_penalty
        self.gen_settings.top_k = query.top_k
        self.gen_settings.top_p = query.top_p

        with Timer() as outer_timer:
            for line in utils.split_text(text, query.max_len):
                with Timer() as inner_timer:
                    messages = [
                        {
                            "role": "user",
                            "content": (
                                "Convert the text to speech:"
                                "<|TEXT_UNDERSTANDING_START|>"
                                f"{ref_text}{line}"
                                "<|TEXT_UNDERSTANDING_END|>"
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": f"<|SPEECH_GENERATION_START|>{ref_speech}",
                        },
                    ]

                    input = self.template.render(messages=messages)[:-10]
                    input_ids = self.model.tokenizer.encode(input, add_bos=True)
                    max_new_tokens = self.max_seq_len - input_ids.shape[-1]
                    output = []

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

                    while self.model.num_remaining_jobs():
                        for result in self.model.iterate():
                            if result.get("stage") == "streaming":
                                text = result.get("text")

                                if text:
                                    output.append(text)

                            if result.get("eos") and output:
                                if query.reuse:
                                    ref_speech = "".join(output)
                                    ref_text = line

                                yield output

                inner_timer(f"Generated {len(output)} tokens")

            self.model.clear_queue()

        outer_timer(f"Finished with seed {query.seed}")

    @autocast
    def decode(self, input: list[str], sample_rate: int, format: str) -> bytes:
        input = [int(i[4:-2]) for i in input]
        input = torch.tensor([[input]]).to(self.device)

        output = self.codec.decode_code(input)
        output = output[0, 0, :].unsqueeze(0)
        output = utils.process_audio(output, self.sample_rate, sample_rate)

        buffer = BytesIO()
        torchaudio.save(buffer, output.cpu(), sample_rate, format=format)
        return buffer.getvalue()

    def generate(self, query: Query) -> bytes | None:
        outputs = []

        for output in self(query):
            outputs.extend(output)

        if outputs:
            return self.decode(outputs, query.sample_rate, query.format)

    def stream(self, query: Query) -> Generator[bytes, None, None]:
        for output in self(query):
            yield self.decode(output, query.sample_rate, query.format)
