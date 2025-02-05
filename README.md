# LLaSA Server
Server for [LLaSA](https://huggingface.co/collections/HKUSTAudio/llasa-679b87dbd06ac556cc0e0f44) using [ExLlamaV2](https://github.com/turboderp-org/exllamav2) and [FastAPI](https://github.com/fastapi/fastapi).

## Installation
```sh
git clone https://github.com/zuellni/llasa-server
cd llasa-server
pip install -r requirements.txt
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