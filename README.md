# LLaSA Server
Server and client for [LLaSA](https://huggingface.co/collections/HKUSTAudio/llasa-679b87dbd06ac556cc0e0f44) using [ExLlamaV2](https://github.com/turboderp-org/exllamav2) and [FastAPI](https://github.com/fastapi/fastapi).
## Installation
```sh
git clone https://github.com/zuellni/llasa-server
cd llasa-server
pip install -r requirements.txt
spacy download en_core_web_sm
```
## Downloads
### LLaSA 1B
```sh
git clone https://huggingface.co/hkustaudio/llasa-1b             model # bf16
```
### LLaSA 3B
```sh
git clone https://huggingface.co/annuvin/llasa-3b-8.0bpw-h8-exl2 model # 8.0bpw
git clone https://huggingface.co/hkustaudio/llasa-3b             model # bf16
```
### LLaSA 8B
```sh
git clone https://huggingface.co/annuvin/llasa-8b-4.0bpw-exl2    model # 4.0bpw
git clone https://huggingface.co/annuvin/llasa-8b-6.0bpw-exl2    model # 6.0bpw
git clone https://huggingface.co/annuvin/llasa-8b-8.0bpw-h8-exl2 model # 8.0bpw
git clone https://huggingface.co/hkustaudio/llasa-8b             model # bf16
```
### X-Codec-2.0
```sh
git clone https://huggingface.co/annuvin/xcodec2-bf16            codec # bf16
git clone https://huggingface.co/srinivasbilla/xcodec2           codec # fp32
```
## Usage
```sh
python server -m model -c codec -a audio
python -m http.server -b 127.0.0.1 8021 -d client
start http://127.0.0.1:8021
```
