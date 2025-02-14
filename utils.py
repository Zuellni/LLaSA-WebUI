from pathlib import Path
from time import time
from typing import Self

import torch
from exllamav2 import (
    ExLlamaV2Cache,
    ExLlamaV2CacheBase,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
)
from rich import print
from rich.progress import (
    BarColumn,
    Progress as ProgressBar,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
from semantic_text_splitter import TextSplitter
from torchaudio import functional as F


class Progress:
    def __init__(self, description: str, total: float = 1.0) -> None:
        self.description = description
        self.total = total
        self.task = None

        self.progress = ProgressBar(
            TextColumn(f"[green]INFO[/green]:{' ' * 5}{{task.description}}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

    def __enter__(self) -> Self:
        self.progress.start()
        self.task = self.progress.add_task(self.description, total=self.total)
        return self

    def __exit__(self, *args) -> None:
        self.progress.update(self.task, completed=self.total)
        self.progress.stop()

    def __call__(self, *args, advance: float = 1.0) -> None:
        self.progress.advance(self.task, advance)


class Timer:
    def __init__(self) -> None:
        self.start = 0.0
        self.end = 0.0
        self.total = 0.0

    def __enter__(self) -> Self:
        self.start = time()
        return self

    def __exit__(self, *args) -> None:
        self.end = time()
        self.total = self.end - self.start

    def __call__(self, text: str, precision: int = 2) -> None:
        print(
            f"[green]INFO[/green]:{' ' * 5}{text} "
            f"in {self.total:.{precision}f} seconds."
        )


def cache(cache: str) -> ExLlamaV2CacheBase:
    match cache:
        case "q4":
            return ExLlamaV2Cache_Q4
        case "q6":
            return ExLlamaV2Cache_Q6
        case "q8":
            return ExLlamaV2Cache_Q8
        case _:
            return ExLlamaV2Cache


def dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "fp16":
            return torch.float16
        case "bf16":
            return torch.bfloat16
        case _:
            return torch.float32


def process_audio(
    audio: torch.Tensor, input_rate: int, output_rate: int, max_len: int = 0
) -> torch.Tensor:
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if max_len and audio.shape[1] / input_rate > max_len:
        audio = audio[:, : input_rate * max_len]

    if input_rate != output_rate:
        audio = F.resample(audio, input_rate, output_rate)

    return audio


def clean_text(text: list[str] | str) -> str:
    if isinstance(text, list):
        text = "\n".join(text)

    lines = [" ".join(l.split()) for l in text.splitlines()]
    lines = [l.strip() for l in lines if l.strip()]
    return "\n".join(lines)


def process_text(text: str, suffixes: list[str] = [".txt"]) -> str:
    path = Path(text)

    if path.is_dir():
        files = [f for f in path.glob("*.*") if f.suffix in suffixes]
        text = [f.read_text(encoding="utf-8") for f in files]
    elif path.is_file():
        text = path.read_text(encoding="utf-8")

    return clean_text(text)


def split_text(text: str, max_len: int) -> list[str]:
    splitter = TextSplitter(max_len)
    chunks = []

    for line in text.splitlines():
        chunk = splitter.chunks(line)
        chunks.extend(chunk)

    return chunks
