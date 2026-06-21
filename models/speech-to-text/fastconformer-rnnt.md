# NeMo Fast-Conformer (RNN-T) 훈련 & 추론 가이드

> 소스: https://github.com/NVIDIA/NeMo
> 기준 브랜치: `main` (2025년 기준)

---

## 목차

1. [환경 설치](#1-환경-설치)
2. [데이터 준비](#2-데이터-준비)
3. [토크나이저 생성](#3-토크나이저-생성)
4. [훈련 (Training)](#4-훈련-training)
5. [파인튜닝 (Fine-tuning)](#5-파인튜닝-fine-tuning)
6. [배치 추론 (Batch Inference)](#6-배치-추론-batch-inference)
7. [버퍼드 스트리밍 추론](#7-버퍼드-스트리밍-추론)
8. [Python API 추론](#8-python-api-추론)
9. [사전학습 모델 목록](#9-사전학습-모델-목록)

---

## 1. 환경 설치

```bash
# 1. NVIDIA NeMo 설치 (CUDA 12.x 권장)
pip install nemo_toolkit['asr']

# 또는 최신 소스에서 설치
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
pip install '.[asr]'

# 2. 필수 의존성
pip install lightning omegaconf hydra-core sentencepiece
```

**Docker 권장 환경**

```bash
docker pull nvcr.io/nvidia/nemo:24.07
docker run --gpus all -it --rm \
    -v /your/data:/data \
    nvcr.io/nvidia/nemo:24.07
```

---

## 2. 데이터 준비

NeMo는 **JSON Lines 형식의 manifest 파일**을 사용합니다.
각 줄은 하나의 오디오 샘플을 나타냅니다.

### manifest 형식

```jsonl
{"audio_filepath": "/data/audio/sample_001.wav", "duration": 3.45, "text": "안녕하세요 반갑습니다"}
{"audio_filepath": "/data/audio/sample_002.wav", "duration": 5.12, "text": "오늘 날씨가 좋네요"}
{"audio_filepath": "/data/audio/sample_003.wav", "duration": 2.80, "text": "네 감사합니다"}
```

### HuggingFace 데이터셋 → NeMo 변환

```bash
python scripts/speech_recognition/convert_hf_dataset_to_nemo.py \
    dataset_name="mozilla-foundation/common_voice_13_0" \
    language="ko" \
    split="train" \
    output_dir="/data/korean_cv" \
    use_auth_token=True
```

### 커스텀 데이터 manifest 생성 (Python)

```python
import json
import soundfile as sf
from pathlib import Path

def create_manifest(audio_dir: str, output_path: str):
    entries = []
    for wav_file in Path(audio_dir).glob("**/*.wav"):
        txt_file = wav_file.with_suffix(".txt")
        if not txt_file.exists():
            continue
        info = sf.info(str(wav_file))
        entries.append({
            "audio_filepath": str(wav_file),
            "duration": info.duration,
            "text": txt_file.read_text().strip()
        })
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"총 {len(entries)}개 샘플 저장 → {output_path}")

create_manifest("/data/korean_audio", "/data/train_manifest.json")
```

---

## 3. 토크나이저 생성

Fast-Conformer RNN-T (BPE 모델)는 **SentencePiece 토크나이저**를 사용합니다.

```bash
python scripts/tokenizers/process_asr_text_tokenizer.py \
    --manifest="/data/train_manifest.json" \
    --data_root="/data/tokenizer" \
    --vocab_size=1024 \
    --tokenizer="spe" \
    --spe_type="unigram" \
    --spe_character_coverage=1.0 \
    --log
```

> **한국어 팁:** `--spe_character_coverage=1.0` 으로 설정해야 한글 자모가 누락되지 않습니다.
> 영어는 기본 0.9995 사용. vocab_size는 한국어의 경우 2048~4096 권장.

생성 결과:
```
/data/tokenizer/
  tokenizer_spe_unigram_v1024/
    tokenizer.model    ← SentencePiece 모델
    tokenizer.vocab    ← 어휘 목록
    vocab.txt
```

---

## 4. 훈련 (Training)

### 4-1. 훈련 스크립트 위치

```
examples/asr/asr_transducer/speech_to_text_rnnt.py
examples/asr/conf/conformer/conformer_transducer_bpe.yaml  ← 기본 config
```

### 4-2. Fast-Conformer config 핵심 수정사항

`conformer_transducer_bpe.yaml`에서 아래 항목을 수정하여 Fast-Conformer로 전환합니다:

```yaml
model:
  encoder:
    _target_: nemo.collections.asr.modules.ConformerEncoder
    subsampling: dw_striding       # Fast-Conformer: depthwise striding
    subsampling_factor: 8          # 기존 4 → 8 (Fast-Conformer 핵심)
    subsampling_conv_channels: 256
    d_model: 512
    n_layers: 17
    n_heads: 8
    # Local attention 설정 (Fast-Conformer 핵심)
    self_attention_model: rel_pos
    att_context_size: [128, 128]   # [-1,-1] 전역 → [128,128] 로컬
    att_context_style: regular
    conv_kernel_size: 9            # 기존 31 → 9 (dw_striding 최적값)
    conv_norm_type: layer_norm
```

### 4-3. 단일 GPU 훈련

```bash
python examples/asr/asr_transducer/speech_to_text_rnnt.py \
    --config-path="../conf/conformer" \
    --config-name="conformer_transducer_bpe" \
    model.train_ds.manifest_filepath="/data/train_manifest.json" \
    model.validation_ds.manifest_filepath="/data/val_manifest.json" \
    model.tokenizer.dir="/data/tokenizer/tokenizer_spe_unigram_v1024" \
    model.tokenizer.type="bpe" \
    model.encoder.subsampling=dw_striding \
    model.encoder.subsampling_factor=8 \
    model.encoder.att_context_size=[128,128] \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=bf16 \
    trainer.max_epochs=100 \
    trainer.accumulate_grad_batches=4 \
    model.train_ds.batch_size=16 \
    exp_manager.exp_dir="/data/experiments/fastconformer_rnnt"
```

### 4-4. 멀티 GPU 훈련 (DDP)

```bash
python examples/asr/asr_transducer/speech_to_text_rnnt.py \
    --config-path="../conf/conformer" \
    --config-name="conformer_transducer_bpe" \
    model.train_ds.manifest_filepath="/data/train_manifest.json" \
    model.validation_ds.manifest_filepath="/data/val_manifest.json" \
    model.tokenizer.dir="/data/tokenizer/tokenizer_spe_unigram_v1024" \
    model.tokenizer.type="bpe" \
    trainer.devices=4 \
    trainer.num_nodes=1 \
    trainer.strategy=ddp \
    trainer.accelerator=gpu \
    trainer.precision=bf16 \
    trainer.max_epochs=100 \
    model.train_ds.batch_size=32 \
    model.optim.lr=0.01 \
    model.optim.weight_decay=1e-3 \
    exp_manager.exp_dir="/data/experiments/fastconformer_rnnt"
```

### 4-5. 전체 Python 훈련 코드

```python
import lightning.pytorch as pl
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.utils.exp_manager import exp_manager

# config 로드 및 수정
cfg = OmegaConf.load("conformer_transducer_bpe.yaml")

# Fast-Conformer 설정
cfg.model.encoder.subsampling = "dw_striding"
cfg.model.encoder.subsampling_factor = 8
cfg.model.encoder.att_context_size = [128, 128]
cfg.model.encoder.conv_kernel_size = 9

# 데이터 경로
cfg.model.train_ds.manifest_filepath = "/data/train_manifest.json"
cfg.model.validation_ds.manifest_filepath = "/data/val_manifest.json"
cfg.model.tokenizer.dir = "/data/tokenizer/tokenizer_spe_unigram_v1024"

# 훈련기 설정
trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    max_epochs=100,
    accumulate_grad_batches=4,
    log_every_n_steps=50,
)

exp_manager(trainer, cfg.get("exp_manager", None))

# 모델 생성 및 훈련
model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)
trainer.fit(model)

# .nemo 파일로 저장
model.save_to("/data/models/fastconformer_rnnt_ko.nemo")
```

### 4-6. 주요 하이퍼파라미터 가이드

| 파라미터 | 권장값 | 설명 |
|----------|--------|------|
| `subsampling_factor` | 8 | Fast-Conformer 핵심, 시퀀스 압축 |
| `att_context_size` | [128, 128] | 로컬 어텐션 윈도우 (약 10초) |
| `d_model` | 512 (Large) / 1024 (XLarge) | 모델 크기 |
| `n_layers` | 17 (Large) / 24 (XLarge) | 레이어 수 |
| `precision` | bf16 | A100/H100은 bf16, V100은 fp16 |
| `batch_size` | 16~32 (GPU당) | bf16 기준 80GB GPU |
| `lr` | 0.005~0.01 | NoamAnnealing 스케줄러와 함께 |

---

## 5. 파인튜닝 (Fine-tuning)

### 5-1. 사전학습 모델로부터 파인튜닝

```bash
python examples/asr/asr_transducer/speech_to_text_rnnt.py \
    --config-path="../conf/conformer" \
    --config-name="conformer_transducer_bpe" \
    init_from_pretrained_model="stt_en_fastconformer_transducer_large" \
    model.train_ds.manifest_filepath="/data/ko_train_manifest.json" \
    model.validation_ds.manifest_filepath="/data/ko_val_manifest.json" \
    model.tokenizer.dir="/data/tokenizer/tokenizer_spe_unigram_v2048" \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.max_epochs=50 \
    model.optim.lr=0.001 \
    trainer.precision=bf16
```

### 5-2. .nemo 체크포인트에서 파인튜닝

```python
from nemo.collections.asr.models import EncDecRNNTBPEModel
import lightning.pytorch as pl

# 기존 모델 로드
model = EncDecRNNTBPEModel.restore_from("/data/models/stt_en_fastconformer_transducer_large.nemo")

# 토크나이저 교체 (언어 변경 시)
model.change_vocabulary(
    new_tokenizer_dir="/data/tokenizer/tokenizer_spe_unigram_v2048",
    new_tokenizer_type="bpe"
)

# 인코더 동결 (선택 사항 - 데이터가 적을 때 권장)
model.encoder.freeze()

# 훈련 데이터 설정
model.setup_training_data(train_data_config={
    "manifest_filepath": "/data/ko_train_manifest.json",
    "sample_rate": 16000,
    "batch_size": 16,
    "shuffle": True,
})
model.setup_validation_data(val_data_config={
    "manifest_filepath": "/data/ko_val_manifest.json",
    "sample_rate": 16000,
    "batch_size": 16,
    "shuffle": False,
})

trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=50, precision="bf16-mixed")
trainer.fit(model)
model.save_to("/data/models/fastconformer_rnnt_ko_finetuned.nemo")
```

---

## 6. 배치 추론 (Batch Inference)

### 6-1. 커맨드라인 사용

```bash
# .nemo 파일 사용
python examples/asr/transcribe_speech.py \
    model_path=/data/models/fastconformer_rnnt_ko.nemo \
    audio_dir=/data/test_audio/ \
    output_filename=/data/results/transcriptions.json \
    batch_size=32 \
    cuda=0 \
    amp=True

# NGC 사전학습 모델 사용
python examples/asr/transcribe_speech.py \
    pretrained_name="stt_en_fastconformer_transducer_large" \
    audio_dir=/data/test_audio/ \
    output_filename=/data/results/transcriptions.json \
    batch_size=16 \
    cuda=0 \
    amp=True

# manifest 파일로 WER 계산 포함
python examples/asr/transcribe_speech.py \
    model_path=/data/models/fastconformer_rnnt_ko.nemo \
    dataset_manifest=/data/test_manifest.json \
    output_filename=/data/results/output.json \
    batch_size=32 \
    cuda=0 \
    amp=True \
    calculate_wer=True
```

### 6-2. Python API로 추론

```python
import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

# 모델 로드
model = EncDecRNNTBPEModel.restore_from("/data/models/fastconformer_rnnt_ko.nemo")
# 또는 NGC에서 자동 다운로드
# model = EncDecRNNTBPEModel.from_pretrained("stt_en_fastconformer_transducer_large")

model.eval()
if torch.cuda.is_available():
    model = model.cuda()

# 단일 파일 추론
transcriptions = model.transcribe(["/data/audio/test.wav"])
print(transcriptions[0])

# 여러 파일 배치 추론
audio_files = [
    "/data/audio/sample_001.wav",
    "/data/audio/sample_002.wav",
    "/data/audio/sample_003.wav",
]
transcriptions = model.transcribe(audio_files, batch_size=8)
for audio, text in zip(audio_files, transcriptions):
    print(f"{audio}: {text}")

# 타임스탬프 포함 추론
transcriptions = model.transcribe(
    audio_files,
    return_hypotheses=True,
)
for hyp in transcriptions:
    print(f"텍스트: {hyp.text}")
    if hyp.timestep is not None:
        print(f"타임스탬프: {hyp.timestep}")
```

---

## 7. 버퍼드 스트리밍 추론

긴 오디오(20초 이상)나 실시간에 가까운 처리에 사용합니다.
청크를 1.6초 단위로 잘라 순차 처리합니다.

### 7-1. 커맨드라인

```bash
# Middle Token Merge 알고리즘 (기본, 빠름)
python examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py \
    model_path=/data/models/fastconformer_rnnt_ko.nemo \
    audio_dir=/data/long_audio/ \
    output_filename=/data/results/streaming_output.json \
    total_buffer_in_secs=4.0 \
    chunk_len_in_secs=1.6 \
    batch_size=16 \
    cuda=0

# LCS Merge 알고리즘 (더 정확, 느림)
python examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py \
    model_path=/data/models/fastconformer_rnnt_ko.nemo \
    audio_dir=/data/long_audio/ \
    output_filename=/data/results/streaming_output.json \
    total_buffer_in_secs=4.0 \
    chunk_len_in_secs=1.6 \
    batch_size=16 \
    merge_algo="lcs" \
    cuda=0
```

### 7-2. 파라미터 설명

| 파라미터 | 설명 | 권장값 |
|----------|------|--------|
| `chunk_len_in_secs` | 한 번에 처리할 청크 크기 | 1.6초 (스트리밍), 5~10초 (배치) |
| `total_buffer_in_secs` | 청크 + 앞뒤 컨텍스트 버퍼 크기 | chunk × 2.5 |
| `merge_algo` | 청크 결합 알고리즘 | `middle` (빠름) / `lcs` (정확) |

### 7-3. 마이크 실시간 추론 예시

```python
import pyaudio
import numpy as np
import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

model = EncDecRNNTBPEModel.restore_from("/data/models/fastconformer_rnnt_ko.nemo")
model.eval().cuda()

# 스트리밍 설정 (att_context_size=[128,128] 모델 기준)
model.change_decoding_strategy(
    decoding_cfg={"strategy": "greedy", "greedy": {"max_symbols": 10}}
)

CHUNK = 1600   # 0.1초 @ 16kHz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

print("마이크 입력 시작 (Ctrl+C로 종료)")
buffer = []
try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        buffer.append(audio)
        # 1.6초 분량(16 청크) 쌓이면 추론
        if len(buffer) >= 16:
            chunk_audio = np.concatenate(buffer[-16:])
            result = model.transcribe([chunk_audio], batch_size=1)
            if result and result[0].strip():
                print(f"\r인식: {result[0]}", end="", flush=True)
except KeyboardInterrupt:
    print("\n종료")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
```

---

## 8. Python API 추론

### 디코딩 전략 변경

```python
from nemo.collections.asr.models import EncDecRNNTBPEModel

model = EncDecRNNTBPEModel.restore_from("fastconformer_rnnt_ko.nemo")
model.eval().cuda()

# Greedy 디코딩 (빠름, 기본)
model.change_decoding_strategy(decoding_cfg={
    "strategy": "greedy_batch",
    "greedy": {"max_symbols": 10}
})

# Beam Search 디코딩 (정확, 느림)
model.change_decoding_strategy(decoding_cfg={
    "strategy": "beam",
    "beam": {
        "beam_size": 4,
        "score_norm": True,
        "return_best_hypothesis": True,
    }
})

result = model.transcribe(["test.wav"])
print(result)
```

### WER 평가

```python
from nemo.collections.asr.metrics.wer import WER

wer_metric = WER(vocabulary=model.decoder.vocabulary)
hypotheses, references = [], []

for batch in test_dataloader:
    audio, audio_len, tokens, tokens_len = batch
    with torch.no_grad():
        log_probs, encoded_len, greedy_predictions = model(
            input_signal=audio.cuda(),
            input_signal_length=audio_len.cuda()
        )
    hypotheses += model.decoding.ctc_decoder_predictions_tensor(greedy_predictions)
    references += [model.tokenizer.ids_to_text(t[:l]) for t, l in zip(tokens, tokens_len)]

wer_value = wer_metric(hypotheses, references)
print(f"WER: {wer_value:.4f}")
```

---

## 9. 사전학습 모델 목록

### Fast-Conformer Transducer (NGC)

| 모델명 | 파라미터 | 학습 데이터 | 특징 |
|--------|----------|-------------|------|
| `stt_en_fastconformer_transducer_large` | ~115M | LibriSpeech 960h | 영어, 기본 |
| `stt_en_fastconformer_transducer_large_ls` | ~115M | LibriSpeech + 추가 | 영어, 고성능 |
| `stt_en_fastconformer_transducer_xlarge` | ~600M | 대규모 영어 | 영어, 최고성능 |
| `stt_en_fastconformer_transducer_xxlarge` | ~1.1B | 대규모 영어 | 영어, 최대 모델 |

```python
# 모델 자동 다운로드 및 사용
from nemo.collections.asr.models import EncDecRNNTBPEModel

# 사용 가능한 모델 목록 확인
models = EncDecRNNTBPEModel.list_available_models()
for m in models:
    print(m.pretrained_model_name)

# 모델 다운로드 & 로드
model = EncDecRNNTBPEModel.from_pretrained("stt_en_fastconformer_transducer_large")
```

---

## 참고 링크

- NeMo ASR 문서: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/
- Fast-Conformer 논문: https://arxiv.org/abs/2305.05084
- NeMo ASR 결과: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/results.html
- NGC 모델 허브: https://catalog.ngc.nvidia.com/models?filters=&orderBy=weightPopularDESC&query=fastconformer
