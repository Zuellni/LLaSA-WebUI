# LLaSA TTS Server
A simple server (and client) for [LLaSA TTS](https://huggingface.co/collections/HKUSTAudio/llasa-679b87dbd06ac556cc0e0f44) using [ExLlamaV2](https://github.com/turboderp-org/exllamav2).

## Installation
```sh
git clone https://github.com/zuellni/llasa-tts-server
cd llasa-tts-server
pip install -r requirements.txt
```

## Download
```sh
git clone https://huggingface.co/hkustaudio/llasa-1b             model # 1b @ bf16

git clone https://huggingface.co/annuvin/llasa-3b-8.0bpw-h8-exl2 model # 3b @ 8.0bpw
git clone https://huggingface.co/hkustaudio/llasa-3b             model # 3b @ bf16

git clone https://huggingface.co/annuvin/llasa-8b-4.0bpw-exl2    model # 8b @ 4.0bpw
git clone https://huggingface.co/annuvin/llasa-8b-6.0bpw-exl2    model # 8b @ 6.0bpw
git clone https://huggingface.co/annuvin/llasa-8b-8.0bpw-h8-exl2 model # 8b @ 8.0bpw
git clone https://huggingface.co/hkustaudio/llasa-8b             model # 8b @ bf16

git clone https://huggingface.co/annuvin/xcodec2-bf16            codec # bf16
git clone https://huggingface.co/srinivasbilla/xcodec2           codec # fp32
```

## Start
```sh
python server -m model -c codec
python -m http.server -b 127.0.0.1 8021 -d client
start http://127.0.0.1:8021
```
