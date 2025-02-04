from pathlib import Path

import spacy
import torch
from exllamav2 import (
    ExLlamaV2Cache,
    ExLlamaV2CacheBase,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
)
from rich import print
from torchaudio import functional as F

nlp = spacy.load("en_core_web_sm")


def info(text: str) -> None:
    print(f"[green]INFO[/green]:{' ' * 5}{text}")


def get_cache(cache: str) -> type[ExLlamaV2CacheBase]:
    match cache:
        case "q4":
            return ExLlamaV2Cache_Q4
        case "q6":
            return ExLlamaV2Cache_Q6
        case "q8":
            return ExLlamaV2Cache_Q8
        case _:
            return ExLlamaV2Cache


def get_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "fp16":
            return torch.float16
        case "bf16":
            return torch.bfloat16
        case _:
            return torch.float32


def process_audio(
    audio: torch.Tensor, input_rate: int, output_rate: int
) -> torch.Tensor:
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if input_rate != output_rate:
        audio = F.resample(audio, input_rate, output_rate)

    return audio


def clean_text(text: str | list[str]) -> str:
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
    text = clean_text(text)
    chunks = []

    for line in text.splitlines():
        chunk = ""

        for sent in nlp(line).sents:
            sent = sent.text.strip()

            if len(chunk) + len(sent) < max_len:
                chunk = f"{chunk} {sent}" if chunk else sent
            else:
                if chunk:
                    chunks.append(chunk)

                chunk = sent

        if chunk:
            chunks.append(chunk)

    return chunks
