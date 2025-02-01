# LLaSA TTS Server
A simple server (and client) for [LLaSA TTS](https://huggingface.co/collections/HKUSTAudio/llasa-679b87dbd06ac556cc0e0f44) using [ExLlamaV2](https://github.com/turboderp-org/exllamav2).

## Install
```sh
git clone https://github.com/zuellni/llasa-tts-server
cd llasa-tts-server
pip install -r requirements.txt
```

## Download
```sh
git clone https://huggingface.co/hkustaudio/llasa-1b             llasa-1b # 1b @ bf16

git clone https://huggingface.co/hkustaudio/llasa-3b             llasa-3b # 3b @ bf16
git clone https://huggingface.co/annuvin/llasa-3b-8.0bpw-h8-exl2 llasa-3b # 3b @ 8.0bpw

git clone https://huggingface.co/hkustaudio/llasa-8b             llasa-8b # 8b @ bf16
git clone https://huggingface.co/annuvin/llasa-8b-8.0bpw-h8-exl2 llasa-8b # 8b @ 8.0bpw
git clone https://huggingface.co/annuvin/llasa-8b-6.5bpw-h8-exl2 llasa-8b # 8b @ 6.5bpw
git clone https://huggingface.co/annuvin/llasa-8b-6.0bpw-exl2    llasa-8b # 8b @ 6.0bpw

git clone https://huggingface.co/srinivasbilla/xcodec2           xcodec2  # fp32
git clone https://huggingface.co/annuvin/xcodec2-bf16            xcodec2  # bf16
```

## Start
```sh
python server -m llasa-8b -c xcodec2
python -m http.server -b 127.0.0.1 8021 -d client
```
