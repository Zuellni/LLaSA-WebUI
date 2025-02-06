# LLaSA WebUI
WebUI for [LLaSA](https://huggingface.co/collections/HKUSTAudio/llasa-679b87dbd06ac556cc0e0f44) using [ExLlamaV2](https://github.com/turboderp-org/exllamav2) with an [OpenAI](https://platform.openai.com/docs/guides/text-to-speech) compatible [FastAPI](https://github.com/fastapi/fastapi) server.

## Installation
```sh
git clone https://github.com/zuellni/llasa-webui & cd llasa-webui
mamba create -n tts python=3.12 # you can use conda instead, or create a venv
pip install torch torchao torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install exllamav2 # jit, you can install a wheel on windows
pip install flash-attn # optional, needs a wheel on windows
pip install xcodec2 --no-deps # ignore all dependency errors
```

## Downloads
[LLaSA-1B](https://huggingface.co/HKUSTAudio/Llasa-1B):
```sh
git clone https://huggingface.co/hkustaudio/llasa-1b             model # bf16
```

[LLaSA-3B](https://huggingface.co/HKUSTAudio/Llasa-3B):
```sh
git clone https://huggingface.co/annuvin/llasa-3b-8.0bpw-h8-exl2 model # 8bpw
git clone https://huggingface.co/hkustaudio/llasa-3b             model # bf16
```

[LLaSA-8B](https://huggingface.co/HKUSTAudio/Llasa-8B):
```sh
git clone https://huggingface.co/annuvin/llasa-8b-4.0bpw-exl2    model # 4bpw
git clone https://huggingface.co/annuvin/llasa-8b-6.0bpw-exl2    model # 6bpw
git clone https://huggingface.co/annuvin/llasa-8b-8.0bpw-h8-exl2 model # 8bpw
git clone https://huggingface.co/hkustaudio/llasa-8b             model # bf16
```

[X-Codec-2](https://huggingface.co/HKUSTAudio/xcodec2):
```sh
git clone https://huggingface.co/annuvin/xcodec2-bf16            codec # bf16
git clone https://huggingface.co/annuvin/xcodec2-fp32            codec # fp32
```

## Usage
```sh
python server.py -m model -c codec -v voices
```

## Preview
![Preview](assets/preview.png)
