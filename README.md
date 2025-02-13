# LLaSA WebUI
A simple web interface for [LLaSA](https://huggingface.co/collections/HKUSTAudio/llasa-679b87dbd06ac556cc0e0f44) using [ExLlamaV2](https://github.com/turboderp-org/exllamav2) with an [OpenAI](https://platform.openai.com/docs/guides/text-to-speech) compatible [FastAPI](https://github.com/fastapi/fastapi) server.

## Installation
Clone the repo:
```sh
git clone https://github.com/zuellni/llasa-webui
cd llasa-webui
```

Create a conda/mamba/python env:
```sh
conda create -n llasa-webui python=3.12
conda activate llasa-webui
```

Install dependencies, ignore any `xcodec2` errors:
```sh
pip install -r requirements.txt
pip install xcodec2 --no-deps
```
If you want to use `torch+cu126`, keep in mind that you'll need to compile `exllamav2` and (optionally) `flash-attn`, and for `python=3.13` you may need to compile `sentencepiece`.

## Models
LLaSA-1B:
```sh
git clone https://huggingface.co/hkustaudio/llasa-1b              model # bf16
git clone https://huggingface.co/hkustaudio/llasa-1b-multilingual model # bf16
```

LLaSA-3B:
```sh
git clone https://huggingface.co/annuvin/llasa-3b-8.0bpw-h8-exl2  model # 8bpw
git clone https://huggingface.co/hkustaudio/llasa-3b              model # bf16
```

LLaSA-8B:
```sh
git clone https://huggingface.co/annuvin/llasa-8b-4.0bpw-exl2     model # 4bpw
git clone https://huggingface.co/annuvin/llasa-8b-6.0bpw-exl2     model # 6bpw
git clone https://huggingface.co/annuvin/llasa-8b-8.0bpw-h8-exl2  model # 8bpw
git clone https://huggingface.co/hkustaudio/llasa-8b              model # bf16
```

## Usage
```sh
python server.py -m model
```
Add `--cache q4 --dtype bf16` for less [VRAM usage](https://www.canirunthisllm.net).

## Preview
![Preview](assets/preview.png)
