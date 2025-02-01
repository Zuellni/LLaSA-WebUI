from pathlib import Path
from time import time
from typing import Self
from warnings import simplefilter

import spacy
import torch
from rich import print
from torchaudio import functional as F

nlp = spacy.load("en_core_web_sm")
simplefilter("ignore")


class Timer:
    def __init__(self) -> None:
        self.start = 0.0
        self.end = 0.0
        self.total = 0.0

    def __enter__(self) -> Self:
        self.start = time()
        return self

    def __exit__(self, _, __, ___) -> None:
        self.end = time()
        self.total = self.end - self.start

    def __call__(self, text: str, precision: int = 2) -> None:
        print(
            f"[green]INFO[/green]:{' ' * 5}"
            f"{text} in {self.total:.{precision}f} seconds."
        )


def process_audio(
    input: torch.Tensor, input_rate: int, output_rate: int, output_loudness: int = -20.0
) -> torch.Tensor:
    if input.shape[0] > 1:
        input = torch.mean(input, dim=0, keepdim=True)

    if input_rate != output_rate:
        input = F.resample(input, input_rate, output_rate)

    loudness = F.loudness(input, output_rate)
    gain = torch.pow(10.0, (output_loudness - loudness) / 20.0)
    return input * gain


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
