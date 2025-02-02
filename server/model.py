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

caches = {
    4: lambda model: ExLlamaV2Cache_Q4(model, lazy=True),
    6: lambda model: ExLlamaV2Cache_Q6(model, lazy=True),
    8: lambda model: ExLlamaV2Cache_Q8(model, lazy=True),
    16: lambda model: ExLlamaV2Cache(model, lazy=True),
}

dtypes = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


class Model:
    @staticmethod
    def autocast(func) -> Callable[..., Any]:
        def wrapper(self, *args, **kwargs) -> Any:
            with torch.autocast(self.device, self.dtype), torch.inference_mode():
                return func(self, *args, **kwargs)

        return wrapper

    def __init__(
        self,
        model: Path,
        codec: Path,
        audio: Path,
        cache_bits: int = 16,
        device: str = "cuda",
        dtype: str = "fp32",
        max_seq_len: int = 2048,
        sample_rate: int = 16000,
    ) -> None:
        self.device = device
        self.dtype = dtypes.get(dtype, "fp32")
        self.max_seq_len = max_seq_len
        self.sample_rate = sample_rate

        self.model = self.load_model(model, cache_bits)
        self.codec = self.load_codec(codec)
        self.audio = self.load_audio(audio)

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
            cache = caches.get(cache_bits, 16)(model)
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

    def load_audio(
        self, path: Path, suffixes: list[str] = [".flac", ".mp3", ".ogg", ".wav"]
    ) -> dict[str, dict[str, str]]:
        with Timer() as timer:
            files = [f for f in path.glob("*.*") if f.suffix in suffixes]
            audio = dict([self.encode_audio(f) for f in files])

        timer("Cached audio")
        return audio

    @autocast
    def encode_audio(
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
            audio, sample_rate = torchaudio.load(path)
            audio = utils.process_audio(audio, sample_rate, self.sample_rate)
            audio = self.codec.encode_code(audio, self.sample_rate)
            audio = audio[0, 0, :]
            audio = [f"<|s_{a}|>" for a in audio]
            audio = "".join(audio)

            file.write_text(
                json.dumps({"audio": audio, "text": text}), encoding="utf-8"
            )

        data = json.loads(file.read_text(encoding="utf-8"))
        return name, {"audio": data.get("audio", ""), "text": data.get("text", "")}

    def __call__(self, query: Query) -> Generator[list[str], None, None]:
        audio = self.audio.get(query.audio, {})
        ref_audio = audio.get("audio", "")
        ref_text = audio.get("text", "")
        ref_text = f"{ref_text} " if ref_text else ""
        text = utils.clean_text(query.text)

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
                            "content": f"<|SPEECH_GENERATION_START|>{ref_audio}",
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
                                    ref_audio = "".join(output)
                                    ref_text = line

                                yield output

                inner_timer(f"Generated {len(output)} tokens")

            self.model.clear_queue()

        outer_timer(f"Finished with seed {query.seed}")

    @autocast
    def decode_audio(self, input: list[str], sample_rate: int, format: str) -> bytes:
        input = [int(i[4:-2]) for i in input]
        input = torch.tensor([[input]]).to(self.device)

        output = self.codec.decode_code(input)
        output = output[0, 0, :].unsqueeze(0).cpu()
        output = utils.process_audio(output, self.sample_rate, sample_rate)

        buffer = BytesIO()
        torchaudio.save(buffer, output, sample_rate, format=format)
        return buffer.getvalue()

    def generate_audio(self, query: Query) -> bytes | None:
        outputs = []

        for output in self(query):
            outputs.extend(output)

        if outputs:
            return self.decode_audio(outputs, query.sample_rate, query.format)

    def stream_audio(self, query: Query) -> Generator[bytes, None, None]:
        for output in self(query):
            yield self.decode_audio(output, query.sample_rate, query.format)
