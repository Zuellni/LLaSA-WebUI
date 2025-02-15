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

## Usage
```sh
python server.py --model <path or repo id>
```
You can use the HF [models](https://huggingface.co/collections/HKUSTAudio/llasa-679b87dbd06ac556cc0e0f44) or EXL2 quants from [here](https://huggingface.co/collections/Annuvin/llasa-67aeef30ce5e4da91124027c). Add `--cache q4 --dtype bf16` for less [VRAM usage](https://www.canirunthisllm.net).

## Preview
![Preview](assets/preview.png)
