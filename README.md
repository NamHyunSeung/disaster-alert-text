# 재난문자 긴급도 분류 모델

한국 재난문자(CBS)를 **5단계 긴급도(L0~L4)**로 자동 분류하는 딥러닝 모델.  
KoELECTRA v3 기반, 키워드 마스킹 증강 + 3-component 손실 함수 적용.

## 최종 성능 (v9n + threshold=0.69)

| 클래스 | Precision | Recall | F1 |
|--------|-----------|--------|----|
| L0 (긴급아님) | 100.00% | 99.32% | 99.66% |
| L1 (낮음) | 100.00% | 99.55% | 99.77% |
| L2 (중간) | 98.47% | 98.16% | **98.01%** |
| L3 (높음) | 98.37% | 98.06% | **98.21%** |
| L4 (매우높음) | 97.61% | 99.35% | **98.47%** |
| **Macro** | — | — | **98.83%** |
| **Accuracy** | — | — | **99.51%** |

> 목표: L2/L3/L4 마스킹 F1 >= 98%, Recall >= 98% — **전부 달성**

---

## 프로젝트 구조

```
프로젝트/
├── 완성 모델/               # 최종 모델 관련 파일
│   ├── src/
│   │   ├── dataset.py       # 텍스트 전처리
│   │   ├── dataset_v2.py    # 키워드 마스킹 데이터셋
│   │   ├── model.py         # KoELECTRA 모델 로더
│   │   ├── loss.py          # FocalLoss (γ=2.0)
│   │   ├── utils.py         # 평가 지표 계산
│   │   ├── train_v2.py      # 학습 스크립트 (v9 손실)
│   │   └── threshold_sweep.py  # L3 임계값 탐색
│   └── labeling_criteria.md # 레이블링 기준 문서
├── 중요파일/
│   ├── server.py            # FastAPI 추론 서버
│   ├── push_to_hub.py       # HuggingFace Hub 업로드
│   ├── predict.py           # 단일 추론 스크립트
│   ├── hf_space/app.py      # HuggingFace Spaces 데모
│   ├── requirements_server.txt
│   ├── scripts/             # 학습 자동화 파이프라인 (step1~7)
│   ├── data/                # 데이터 (gitignore: *.xlsx)
│   └── android_app/         # Android CBS 수신 앱
├── model_v9n/               # 최종 모델 가중치 (gitignore: *.safetensors)
├── tokenizer_v9n/           # 토크나이저 설정
├── results/                 # 평가 결과 및 시각화
├── 실험/                    # 실험용 구버전 스크립트
└── .gitignore
```

---

## 레이블 정의

| 레이블 | 이름 | 정의 |
|--------|------|------|
| L0 | 긴급아님 | 행동 불필요 (해제, 종료, 예방 안내) |
| L1 | 낮음 | 주의 권고 (주의보 수준, 관망) |
| L2 | 중간 | 대비 행동 권고 (경보 수준, 주의 필요) |
| L3 | 높음 | 즉각 대피 준비 (명백한 위험, 신속 행동) |
| L4 | 매우높음 | 즉시 대피 명령 (생명 위협, 지금 당장) |

**핵심 원칙**: 레이블은 **상황의 심각성**으로 결정, 특정 키워드 유무로 결정하지 않음.  
예: "경보"가 있어도 해제 공지이면 L0; "대피"가 없어도 즉각 위험이면 L4.

---

## 모델 아키텍처

**Base**: `monologg/koelectra-base-v3-discriminator` (KoELECTRA v3, ~110M params) → 5-class classifier

**학습 핵심 기법 (v9)**:

### 1. 키워드 마스킹 증강
학습 시 긴급도 관련 키워드 26개를 `' '`로 치환해 모델이 키워드 없이도 맥락으로 분류하도록 강제.

```python
_MASK_KEYWORDS = ['즉시 대피', '대피명령', '긴급대피', '경보', '주의보', '대피', ...]
```

### 2. 3-Component 손실 함수

```
L_total = CE(원본) + α_masked × CE(마스킹) + α_kl × KL(p_원본 ∥ p_마스킹)
```

- **CE(원본)**: 원본 텍스트에 대한 FocalLoss (γ=2.0)
- **CE(마스킹)**: 마스킹 텍스트에 대한 FocalLoss — 키워드 없이 맥락만으로 학습
- **KL Divergence**: 원본 예측과 마스킹 예측의 분포 일치 — 일관성 학습

### 3. 클래스 불균형 처리
- **WeightedRandomSampler**: 희소 클래스 오버샘플링
- **FocalLoss**: 어려운 샘플에 집중 (γ=2.0)

### 4. 검증 기준
최적 모델 저장 기준: **마스킹 검증셋에서 L2/L3/L4 F1+Recall의 최솟값 최대화**

### 5. L3 임계값 후처리 (threshold=0.69)
```python
if P(L3) >= 0.69:
    pred = L3
else:
    pred = argmax(나머지 클래스)
```
재훈련 없이 L3 F1을 97.83% → 98.21%로 향상.

---

## 학습 방법

```bash
# 프로젝트 루트에서 실행
python "완성 모델/src/train_v2.py" \
    --data_path 중요파일/data/raw/재난문자_레이블링결과_dedup_v3.xlsx \
    --model_dir model_v9n \
    --tok_dir tokenizer_v9n \
    --v9 \
    --consistency_alpha 0.5 \
    --alpha_masked 0.5 \
    --save_by_masked_val \
    --save_criterion l234_min \
    --epochs 20 \
    --batch_size 32
```

---

## 추론 서버 실행

```bash
cd 중요파일
pip install -r requirements_server.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

```bash
# 요청 예시
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "태풍 링링이 접근 중입니다. 해안가 주민은 대피하세요."}'
```

응답:
```json
{
  "label": "L3",
  "confidence": 0.951,
  "priority": "high"
}
```

---

## HuggingFace Hub 배포

```bash
# 프로젝트 루트에서 실행
huggingface-cli login
python 중요파일/push_to_hub.py
```

HuggingFace Spaces 데모: `중요파일/hf_space/app.py`

---

## Android 앱

`중요파일/android_app/` — CBS(재난문자) 자동 수신 후 서버로 분류 요청.

- `CbsReceiver.kt`: CBS 수신 브로드캐스트 리시버
- `ApiClient.kt`: 분류 서버 HTTP 클라이언트
- `NotificationHelper.kt`: 긴급도별 알림 표시

---

## 환경

```
Python >= 3.9
torch >= 2.0
transformers >= 4.38
scikit-learn
pandas, openpyxl
```

---

## 모델-토크나이저 매칭표

| 모델 | 베이스 모델 | 토크나이저 | vocab | 매칭 |
|------|------------|------------|-------|------|
| model ~ model_v4_probe | KLUE-BERT (`klue/bert-base`) | tokenizer (32000) | 32000 | ✓ |
| model_v5 ~ model_v8 | KoELECTRA v3 | tokenizer_v5 ~ tokenizer_v7 | 35000 | ✓ |
| model_v9 ~ model_v9d | KoELECTRA v3 (35000) | tokenizer_v9 ~ tokenizer_v9d (32000) | **불일치** | ✗ |
| model_v9e ~ model_v9o | KoELECTRA v3 | tokenizer_v9e ~ tokenizer_v9o | 35000 | ✓ |
| **model_v9n (최종)** | **KoELECTRA v3** | **tokenizer_v9n** | **35000** | **✓** |

> v9~v9d는 실험 중 잘못된 토크나이저(vocab=32000)가 저장된 오류 버전 — 추론 시 사용 불가.  
> **실제 사용 가능한 최종 모델: `model_v9n` + `tokenizer_v9n`**

---

## 실험 이력

| 버전 | 베이스 모델 | 주요 변경 | Macro F1 |
|------|------------|-----------|----------|
| model ~ v4 | KLUE-BERT | 초기 실험, 기본 CE | ~94% |
| v5~v8 | **KoELECTRA v3** | FocalLoss, 키워드 마스킹 증강 | ~96~97% |
| v9~v9d | KoELECTRA v3 | 3-component 손실 (토크나이저 오류) | — |
| v9e~v9m | KoELECTRA v3 | 3-component 손실 (정상) | ~98.5% |
| **v9n** | KoELECTRA v3 | 레이블링 기준 개선 | **98.83%** |
| v9n + thr=0.69 | — | L3 임계값 후처리 | **98.83% (L2/3/4 >= 98%)** |

실험용 스크립트는 `실험/` 폴더 참고.
