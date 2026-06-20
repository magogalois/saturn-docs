# Saturn Docs

![Documentation](https://img.shields.io/badge/docs-documentation-blue)
![MAGO](https://img.shields.io/badge/MAGO-research-green)
![Saturn](https://img.shields.io/badge/project-Saturn-orange)
![Speech AI](https://img.shields.io/badge/topic-speech%20AI-purple)
![Research](https://img.shields.io/badge/status-research-informational)
![Markdown](https://img.shields.io/badge/format-markdown-black)

Saturn Docs는 음성 기술을 중심으로 기술 개념, 모델 연구, Survey 자료, 응용 분야를 정리하는 문서 저장소입니다.

현재는 음성 신호가 수집되고 디지털 데이터로 변환되는 과정부터 정리하고 있으며, 이후 음성 인식, 음성 합성, 화자 인식, 음성 에이전트 등으로 범위를 확장할 예정입니다.

## 문서 구성

| 상위 폴더 | 하위 폴더 | 설명 | 주요 내용 |
| --- | --- | --- | --- |
| `tech-notes` | - | 기술 개념과 구현 원리를 설명하는 문서를 정리합니다. | 음성 기술 기초, 모델 구조, 시스템 구성, 구현 원리 |
| `tech-notes` | `signal_processing` | 음성 신호 처리의 기초 내용을 정리합니다. | 음성 신호, 샘플링, 양자화, 코덱, DSP |
| `tech-surveys` | - | 논문, 모델, 오픈소스, 제품, 벤치마크 등 Survey 자료를 정리할 예정입니다. | 모델 비교, 기술 동향, 데이터셋, 평가 기준 |

## 주요 연구 주제

### 음성 신호 처리

- 음성 신호의 물리적 특성
- 마이크와 아날로그 프론트엔드
- Sampling rate, Nyquist theorem, aliasing
- Quantization, codec, PCM, DSP
- 잡음 제거, 필터링, 특징 추출

### 음성 모델

- ASR(Automatic Speech Recognition)
- TTS(Text-to-Speech)
- Speaker recognition / verification
- Voice activity detection
- Speech enhancement
- Speech-to-speech model
- Multimodal audio-language model

### 모델 Survey

- Whisper 계열 모델
- wav2vec, HuBERT, WavLM 계열 모델
- SeamlessM4T, AudioLM, SpeechT5 등 음성 기반 foundation model
- 경량 온디바이스 음성 모델
- 한국어 음성 데이터셋과 평가 벤치마크

### 응용 분야

- 음성 인식 기반 STT 서비스
- 콜센터 통화 분석
- 회의록 자동 생성
- 음성 에이전트와 실시간 대화 시스템
- 음성 품질 평가와 음성 바이오마커 분석
- 임베디드/IoT 음성 인터페이스

## 작성 방향

문서는 다음 원칙에 따라 정리합니다.

- 기술 개념은 배경, 핵심 원리, 실제 활용 순서로 설명합니다.
- 모델 Survey는 모델 구조, 학습 데이터, 성능, 장단점, 적용 가능성을 함께 기록합니다.
- 응용 분야는 문제 정의, 필요한 기술 요소, 시스템 구성, 한계와 고려사항을 중심으로 정리합니다.

## 앞으로 작성할 문서

- `speech_signal_processing.md`: 필터링, feature extraction, STFT, Mel spectrogram, MFCC 정리
- `asr_models.md`: ASR 모델 구조와 주요 모델 Survey
- `tts_models.md`: TTS 모델 구조와 주요 모델 Survey
- `speaker_models.md`: 화자 인식 및 화자 검증 기술 정리
- `voice_agent.md`: 실시간 음성 에이전트 시스템 구성
- `korean_speech_datasets.md`: 한국어 음성 데이터셋 및 평가 기준 정리
