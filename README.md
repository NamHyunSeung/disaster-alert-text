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
├── tokenizer_v9n/           # 최종 토크나이저 (vocab 35000)
├── results/                 # 평가 결과 및 시각화
│   └── outputs/             # 초기 실험 분석 결과 (attention, baseline 등)
├── 실험/                    # 실험용 구버전 스크립트 및 모델
│   ├── 베이스모델/          # 사전학습 베이스 모델 (gitignore)
│   │   ├── klue_bert_model/
│   │   ├── koelectra_model/
│   │   └── koelectra_v2_model/
│   ├── klue_bert/           # KLUE-BERT 초기 실험 (vocab 32000)
│   │   ├── model ~ model_v4_probe/
│   │   └── tokenizer/
│   ├── koelectra_v5_v8/     # KoELECTRA v3, v5~v8 (vocab 35000, ✓)
│   │   ├── model_v5 ~ model_v8/
│   │   └── tokenizer_v5 ~ tokenizer_v7/
│   ├── koelectra_v9_mismatch/  # v9~v9d, vocab 불일치 ✗ (사용 불가)
│   │   ├── model_v9 ~ model_v9d/
│   │   └── tokenizer_v9 ~ tokenizer_v9d/   ← vocab=32000 오류
│   ├── koelectra_v9e_v9o/   # KoELECTRA v3, v9e~v9o (vocab 35000, ✓)
│   │   ├── model_v9e ~ model_v9o/
│   │   └── tokenizer_v9e ~ tokenizer_v9o/
│   ├── analyze_*.py         # 오류·단계별 분석 스크립트
│   └── utils/               # 공통 유틸리티
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

## 데이터셋

| 구분 | 샘플 수 | 비율 |
|------|---------|------|
| Train | 118,118 | 70% |
| Validation | 25,311 | 15% |
| Test | 25,312 | 15% |
| **합계** | **168,741** | 100% |

- 출처: 공공데이터포털 재난문자 데이터 + 수동 레이블링
- 마스킹 테스트셋: 전체 테스트 25,312건 중 56.1%(14,194건)가 키워드 마스킹 적용됨
- **평가 지표**: 키워드 마스킹 테스트셋에서의 Macro F1 — 키워드 암기가 아닌 맥락 이해력 측정

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

## 개발 이력

### 단계 1: 3-class 분류 (프로젝트 초기)

최초 목표는 재난문자를 **긴급 / 주의 / 일반** 3단계로 분류하는 것이었다.  
규칙 기반 Stage 1 + KoELECTRA Stage 2의 2단계 파이프라인 구조로 실험했으며,  
최종 모델(v3, 3-class) 기준 **Macro F1 = 96.58%, Accuracy = 98.50%** 달성.  
→ 이후 긴급도 세분화 필요성으로 5-class(L0~L4) 체계로 전환.

---

### 단계 2: 5-class 전환 — KLUE-BERT 기반 (model ~ v4_probe)

| 버전 | 베이스 모델 | 주요 변경 | 비마스킹 F1 | **마스킹 F1** |
|------|------------|-----------|------------|-------------|
| model | KLUE-BERT (`klue/bert-base`) | 5-class 기본 CE | — | — |
| model_v3 | KLUE-BERT | 레이블 정제 | 99.43% | 97.36% |
| model_v4 | KLUE-BERT | 하이퍼파라미터 튜닝 | 99.40% | 97.49% |
| model_v4_probe | KLUE-BERT | 레이어 프로빙 실험 | 98.94% | — |

KLUE-BERT로 빠르게 98~99%대 비마스킹 F1 달성. 그러나 **키워드 마스킹 시 97.5% 수준**으로, 키워드에 의존하는 경향이 확인됨.

---

### 단계 3: KoELECTRA 전환 — 마스킹 증강 본격 도입 (v5 ~ v8)

| 버전 | 주요 변경 | 비마스킹 F1 | **마스킹 F1** | 비고 |
|------|-----------|------------|-------------|------|
| v5 | KoELECTRA v3 전환, FocalLoss, WeightedSampler | 99.15% | 97.17% | KLUE-BERT 대비 마스킹 소폭 하락 |
| v6 | 발신기관명 마스킹(`[기관]`) + 학습률 조정 | 99.02% | **94.51%** | **⚠ 마스킹 F1 급락** |
| v7 | v6 변경 일부 롤백 | 98.96% | 97.15% | 회복 |
| v8 | 배치 크기·max_length 튜닝 | 98.88% | 97.31% | 안정화 |

**v6 회귀 원인**: 발신기관명 마스킹이 오히려 맥락 정보를 파괴 — 학습 신호가 불안정해진 것으로 분석.  
v5~v8 전반적으로 마스킹 F1이 97.1~97.6% 수준에서 정체.

---

### 단계 4: 3-component 손실 도입 — 토크나이저 오류 (v9 ~ v9d)

3-component 손실(CE + Masked CE + KL Divergence) 적용을 시도했으나,  
**토크나이저 vocab 불일치**(모델 vocab=35000, 저장된 토크나이저 vocab=32000)로 인해 이 버전들은 추론 불가.

---

### 단계 5: 3-component 손실 정상 작동 — 성능 정체 (v9e ~ v9m)

| 버전 | 주요 변경 | 비마스킹 F1 | **마스킹 F1** | 비고 |
|------|-----------|------------|-------------|------|
| v9e | 3-comp 손실 첫 정상 실행 | 98.67% | **93.41%** | ⚠ 마스킹 F1 예상 외 급락 |
| v9f | α 조정 | 99.15% | 97.36% | 회복 |
| v9g | 스케줄러 변경 | 98.95% | 97.10% | — |
| v9h | 검증 기준 변경 | 99.09% | 97.59% | — |
| v9i | save_criterion l234_min | 99.02% | 97.62% | — |
| v9j | α_kl 조정 | 98.94% | 97.50% | — |
| v9k | α_masked 조정 | 98.60% | 97.12% | — |
| v9m | 에포크·배치 조정 | 99.08% | 97.55% | — |

**v9e 회귀 원인**: α_masked / α_kl 초기값이 너무 커서 마스킹 손실이 지배적으로 작용.  
v9f~v9m: 하이퍼파라미터를 반복 튜닝했지만 마스킹 F1이 97.5~97.6% 천장에서 정체.  
**핵심 인식**: 아키텍처/손실 함수 개선만으로는 돌파구가 없었음 → 레이블 품질 문제로 진단.

---

### 단계 6: 레이블링 기준 개선 — 돌파구 (v9n)

| 버전 | 주요 변경 | 비마스킹 F1 | **마스킹 F1** | 비고 |
|------|-----------|------------|-------------|------|
| **v9n** | **레이블링 기준 문서화 + 데이터 재레이블** | **99.58%** | **98.75%** | **+1.2%p 급등** |

`labeling_criteria.md`를 작성하고 경계 케이스(해제 공지, 권고 vs 명령 등)의 레이블을 일관되게 재정립.  
**v9m(97.55%) → v9n(98.75%): 1.2%p 상승은 아키텍처 개선이 아닌 레이블 품질 개선의 결과.**

---

### 단계 7: L3 임계값 후처리 — 최종 (v9n + threshold=0.69)

| threshold | L3 F1 | Macro F1 | 달성 |
|-----------|-------|----------|------|
| 없음 | 97.83% | 98.75% | L3 미달 |
| 0.69 | **98.21%** | **98.83%** | **L2/L3/L4 전부 >= 98%** |

재훈련 없이 L3 임계값만 조정해 목표 달성. **최종 모델: `model_v9n` + `tokenizer_v9n` + threshold=0.69**

---

### 단계 8: OOD 개선 실험 — 데이터 보강 (v14 ~ v17)

v9n 달성 이후 **신종 감염병 등 OOD(Out-of-Distribution) 케이스 처리** 개선을 위한 데이터 보강 실험.

#### 데이터 체인 (dedup_v2 → dedup_v6)

- **dedup_v3**: L1 감염병 문자 중 중증 키워드 포함 45건 → L3 재레이블 (`실험/relabel_v3.py`)
- **dedup_v4**: 신종 감염병 합성 L3 500건 추가 (`실험/generate_synthetic_l3.py`)  
  - 마스킹 후에도 L3 신호 잔존: `원인불명`, `사망자`, `집단 발생`, `치명률`, `신종`
- **dedup_v6**: L3=13,586건(+245 vs v2), L4=4,995건(+200 vs v2)

#### 학습 실험 결과

| 버전 | 데이터 | init_model | LR | alpha(KL) | epochs | 비마스킹 F1 | **마스킹 F1** |
|------|--------|-----------|-----|-----------|--------|------------|-------------|
| v14 | dedup_v6 | v9n | 5e-6 | 0.0 | 10 | **99.49%** | 95.90% |
| v15 | dedup_v6 | v9n | 5e-6 | 0.5 | 10 | — | 95.96% |
| v16 | dedup_v6 | v15 | 2e-6 | 1.0 | 5 | **99.48%** | 96.29% |
| v17 | dedup_v2 | v9n | 5e-6 | 0.5 | 20 | — | 95.54% |

→ dedup_v6 감염병 합성 데이터(500건) 추가 시 masked L4 F1이 87~89%대로 하락, masked MacroF1 96%대에 고착.

#### v16 마스킹 클래스별 성능

| 클래스 | Precision | Recall | F1 | 지지(건) |
|--------|-----------|--------|----|---------|
| L0 긴급아님 | 99.98% | 99.77% | 99.87% | 14,711 |
| L1 낮음 | 99.63% | 99.37% | 99.50% | 6,022 |
| L2 중간 | 98.47% | 97.46% | 97.96% | 1,852 |
| L3 높음 | 94.25% | 95.78% | 95.01% | 2,038 |
| **L4 매우높음** | **87.21%** | **91.05%** | **89.09%** | 749 |
| **Macro** | — | — | **96.29%** | — |

#### OOD 케이스 평가 (7케이스, 마스킹 기준)

| # | 케이스 | 정답 | v9n | v16 |
|---|--------|------|-----|-----|
| 1 | 태양흑점/지자기폭풍 | L4 | L0 X | L1 X |
| 2 | 사이버공격/정전 | L4 | L4 O | L4 O |
| 3 | 드론 위협 | L4 | L0 X | L0 X |
| 4 | 산불+유해가스 | L4 | L2 X | L2 X |
| **5** | **신종 감염병** | **L3** | **L1 X** | **L3 O** |
| 6 | 해저화산/쓰나미 | L4 | L4 O | L4 O |
| 7 | 소행성 파편 낙하 | L4 | L4 O | L4 O |
| | **합계** | | **3/7** | **4/7** |

- v9n: 감염병 L1(98.6%) 오분류 — 훈련 데이터에 감염병 패턴 없음
- v16: 감염병 L3(95.4%) 정분류 — 합성 데이터 효과, 단 masked MacroF1 96.29%로 v9n 대비 하락

---

### 단계 9: 전략 전환 — dedup_v7 데이터 재구성 + 처음부터 학습 (v19 ~ v21)

v9n fine-tune(v18) 대신 전략을 바꿔 **COVID 편향 제거 + 처음부터 전체 재학습**으로 접근.

#### dedup_v7 데이터셋 구성

- **COVID L0 제거**: 2021년 이후 L0 COVID 문자 70% 제거 → 모델이 COVID=L0으로 암기하는 편향 완화
- **합성 데이터 추가**: COVID 키워드 없이 작성한 합성 L1 50건 + L2 50건 → OOD 일반화
- **규모**: 136,987건 (train 95,890 / val 20,548 / test 20,549)
- **upsample_synthetic 5**: 합성 61건 → 305건으로 오버샘플링

#### 학습 실험 결과

| 버전 | 베이스 모델 | epochs | lr | Accuracy | **MacroF1** | 비고 |
|------|-----------|--------|-----|---------|-----------|------|
| v19 | KLUE-BERT (`klue/bert-base`) | 3 | 2e-5 | 97.19% | 95.72% | 기준선 |
| v20 | KoELECTRA v3 | 3 | 2e-5 | 95.97% | 93.78% | epoch 부족(과소학습) |
| v21 | KoELECTRA v3 | 5 | 2e-5 | 97.43% | **95.86%** | 최고 성능 |

#### v21 클래스별 성능 (테스트 세트)

| 클래스 | Precision | Recall | F1 | 오분류 |
|--------|-----------|--------|----|--------|
| L0 긴급아님 | 98.8% | 98.6% | 98.7% | 135건 (1.4%) |
| L1 낮음 | 98.0% | 96.7% | 97.3% | 202건 (3.3%) |
| L2 중간 | 96.8% | 93.1% | 94.9% | 129건 (6.9%) |
| L3 높음 | 92.8% | 98.2% | 95.4% | 36건 (1.8%) |
| **L4 매우높음** | **89.9%** | **96.2%** | **92.9%** | 27건 (3.8%) |
| **Macro** | — | — | **95.86%** | — |

- **L4 Precision 89.9% 문제**: L3 샘플 ~78건이 L4로 과잉예측 — L3↔L4 경계 혼란
- **L2 Recall 93.1% 문제**: 6.9% 오분류 — L3↔L2 상향 오분류 가능성
- v20 epoch 1 loss=1.299 (v19 BERT 0.758 대비 70% 높음) → KoELECTRA는 lr=2e-5에서 초기 수렴이 느림

---

### 단계 10: Ordinal Label Smoothing + LR 최적화 (v22, 진행 예정)

v21의 근본 문제인 **L3↔L4 경계 혼란**을 구조적으로 해결하기 위한 실험.

**핵심 아이디어**: 표준 Label Smoothing은 smoothing mass를 모든 클래스에 균등 배분하지만,  
**Ordinal Smoothing**은 인접 클래스에만 배분하여 순서형 구조를 명시적으로 학습.

```python
# 예: L4에 대한 soft label (smoothing=0.1)
# 표준: [0.02, 0.02, 0.02, 0.02, 0.92]
# Ordinal: [0.0,  0.0,  0.0,  0.10, 0.90]  ← L3에만 배분
```

| 파라미터 | v21 | **v22 (계획)** | 변경 이유 |
|----------|-----|--------------|---------|
| lr | 2e-5 | **5e-5** | epoch 1 loss=1.299 → KoELECTRA 초기 수렴 불충분 |
| smoothing | — | **ordinal (α=0.1)** | L3↔L4 경계 혼란 구조적 해결 |
| warmup | 10% | **20%** | 높은 lr 안정화 |
| epochs | 5 | **7** | 충분한 수렴 보장 |

**목표**: MacroF1 ≥ 98%, L4 Precision ≥ 95%, Accuracy ≥ 98%

---

실험용 스크립트는 `실험/` 폴더 참고.
