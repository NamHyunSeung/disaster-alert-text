# 재난문자 긴급도 분류 모델

## Summary (English)

A deep learning model that automatically classifies
Korean disaster alert messages (CBS) into 5 urgency
levels (L0~L4). Based on KoELECTRA v3 with keyword
masking augmentation, 3-component loss, and Ordinal
Label Smoothing.
**Final performance (v22):** Macro F1 = 98.75%,
Accuracy = 99.27% on 20,549 test samples.

---

한국 재난문자(CBS)를 **5단계 긴급도(L0~L4)**로 자동 분류하는 딥러닝 모델.  
KoELECTRA v3 기반, 키워드 마스킹 증강 + 3-component 손실 함수 + Ordinal Label Smoothing 적용.

## 최종 성능 (v22)

| 클래스 | Precision | Recall | F1 |
|--------|-----------|--------|----|
| L0 (긴급아님) | 99.8% | 99.7% | **99.7%** |
| L1 (낮음) | 99.7% | 98.6% | **99.2%** |
| L2 (중간) | 98.4% | 99.1% | **98.7%** |
| L3 (높음) | 96.9% | 99.5% | **98.2%** |
| L4 (매우높음) | **96.9%** | **99.0%** | **97.9%** |
| **Macro** | — | — | **98.75%** |
| **Accuracy** | — | — | **99.27%** |

> 목표: MacroF1 ≥ 98%, L4 Precision ≥ 95%, Accuracy ≥ 98% — **전부 달성**  
> 평가 데이터: dedup_v7 test set 20,549건

---

## 프로젝트 구조

```
프로젝트/
├── final_model/             # 최종 모델 관련 파일
│   ├── src/
│   │   ├── dataset.py       # 텍스트 전처리
│   │   ├── dataset_v2.py    # 키워드 마스킹 데이터셋
│   │   ├── model.py         # KoELECTRA 모델 로더
│   │   ├── loss.py          # FocalLoss (γ=2.0) + Ordinal Label Smoothing
│   │   ├── utils.py         # 평가 지표 계산
│   │   └── train_v2.py      # 학습 스크립트
│   └── labeling_criteria.md # 레이블링 기준 문서
├── core_files/
│   ├── server.py            # FastAPI 추론 서버
│   ├── push_to_hub.py       # HuggingFace Hub 업로드
│   ├── predict.py           # 단일 추론 스크립트
│   ├── hf_space/app.py      # HuggingFace Spaces 데모
│   ├── requirements_server.txt
│   ├── scripts/             # 학습 자동화 파이프라인 (step1~7)
│   ├── data/                # 데이터 (gitignore: *.xlsx)
│   └── android_app/         # Android CBS 수신 앱
├── pipeline_v22/            # v22 파이프라인 최종 파일 일체
│   ├── model/               # 모델 가중치 (gitignore: *.safetensors)
│   ├── tokenizer/           # 토크나이저 (vocab 35000)
│   ├── ood/                 # KNN OOD 인덱스 (knn_ood_v22.npz, meta.pt, ood_stats.pt)
│   ├── scripts/             # OOD 재빌드 스크립트
│   ├── results/             # 평가 결과 txt
│   ├── evaluate_pipeline.py # 전체 파이프라인 평가 (핵심)
│   ├── predict_v22_knn_ood.py
│   ├── predict_v22_knn_filter.py
│   ├── predict_v22_ood.py
│   └── llm_cache.json       # LLM 호출 캐시 (Gemini/Groq)
├── results/                 # 학습 로그
│   ├── evaluation_report_model_v22.txt   # v22 테스트 평가
│   ├── milestone_log_model_v22.txt       # v22 에포크별 milestone
│   └── overfitting_v22.txt               # v22 과적합 모니터링
├── experiments/             # 실험용 스크립트 (이전 모델 포함)
│   └── predict_full_pipeline.py          # KNN-OOD + confidence 파이프라인 (구버전)
└── .gitignore
```

---

## 레이블 정의

| 레이블 | 이름 | 정의 |
|--------|------|------|
| L0 | 긴급아님 | 행동 불필요 (해제, 종료, 예방 안내, 훈련) |
| L1 | 낮음 | 주의 권고 (예보 수준, 잠재적 위험) |
| L2 | 중간 | 행정 조치 / 방역 협조 (상황 발생, 시설 대응) |
| L3 | 높음 | 야외 자제 / 예방적 조치 (재난 진행 중) |
| L4 | 매우높음 | 즉각 대피 명령 (생명 위협, 지금 당장) |

**핵심 원칙**: 레이블은 **상황의 심각성과 요구되는 행동의 즉시성**으로 결정, 특정 키워드 유무로 결정하지 않음.  
예: "경보"가 있어도 해제 공지이면 L0; "대피"가 없어도 즉각 위험이면 L4.

---

## 데이터셋 (dedup_v7)

| 구분 | 샘플 수 | 비율 |
|------|---------|------|
| Train | 95,890 | 70% |
| Validation | 20,548 | 15% |
| Test | 20,549 | 15% |
| **합계** | **136,987** | 100% |

- 출처: 공공데이터포털 재난문자 + 수동 레이블링
- **COVID 편향 제거**: 2021년 이후 L0 COVID 문자 70% 제거 → COVID=L0 암기 편향 완화
- **합성 데이터 추가**: COVID 키워드 없는 합성 L1 50건 + L2 50건 (upsample 5배 → 305건) → OOD 일반화
- **평가 지표**: 일반 test set 및 마스킹 val set에서의 MacroF1 — 키워드 암기가 아닌 맥락 이해력 측정

---

## 모델 아키텍처

**Base**: `monologg/koelectra-base-v3-discriminator` (KoELECTRA v3, ~110M params) → 5-class classifier

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

### 3. Ordinal Label Smoothing (v22 신규)
표준 Label Smoothing은 smoothing mass를 모든 클래스에 균등 배분하지만,  
**Ordinal Smoothing**은 인접 클래스에만 배분하여 순서형 구조를 명시적으로 학습.

```python
# 예: L4에 대한 soft label (smoothing=0.1)
# 표준:   [0.02, 0.02, 0.02, 0.02, 0.92]
# Ordinal:[0.0,  0.0,  0.0,  0.10, 0.90]  ← L3에만 배분 (L3↔L4 경계 혼란 구조적 해결)
```

### 4. 클래스 불균형 처리
- **WeightedRandomSampler**: 희소 클래스 오버샘플링
- **FocalLoss**: 어려운 샘플에 집중 (γ=2.0)

---

## 3단계 추론 파이프라인 (완성)

```
입력 재난문자
     │
     ▼
[1단계] KNN OOD 탐지
 KNN cosine 거리 > p99 threshold?
     │YES → "OOD 거부" (신종 재난 / 비재난)
     │NO
     ▼
[2단계] Confidence Threshold (70%, Temperature Scaling T=1.5)
 temperature-scaled softmax 최댓값 < 0.70?
     │YES → LLM fallback
     │        Gemini 2.5-flash (1순위) → Groq llama-3.3-70b (2순위, quota 소진 시 skip)
     │        20건 배치 처리 / 결과 llm_cache.json 캐싱
     │NO
     ▼
[3단계] v22 분류
     → L0 / L1 / L2 / L3 / L4 출력
```

**KNN OOD 파라미터**: K=20, cosine distance, brute force, p99 class-level threshold  
**p99 threshold**: L0=0.018056, L1=0.015686, L2=0.014583, L3=0.007599, L4=0.003537  
**Temperature Scaling**: T=1.5 (confidence 보정)  
**Confidence Threshold**: 0.70 (70%)  
**LLM Fallback**: Gemini 2.5-flash → Groq llama-3.3-70b-versatile, 20건 배치, `llm_cache.json` 캐싱

---

## 학습 명령어 (v22)

```bash
python "final_model/src/train_v2.py" \
    --data_path core_files/data/raw/재난문자_레이블링결과_dedup_v7.xlsx \
    --model_dir model_v22 \
    --tok_dir tokenizer_v22 \
    --v9 \
    --ordinal_smoothing 0.1 \
    --consistency_alpha 0.5 \
    --alpha_masked 0.5 \
    --lr 5e-5 \
    --warmup_ratio 0.2 \
    --epochs 7 \
    --batch_size 32
```

---

## 추론 서버 실행

```bash
cd core_files
pip install -r requirements_server.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

```bash
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
| model ~ model_v4_probe | KLUE-BERT (`klue/bert-base`) | tokenizer | 32000 | ✓ |
| model_v5 ~ model_v8 | KoELECTRA v3 | tokenizer_v5 ~ tokenizer_v7 | 35000 | ✓ |
| model_v9 ~ model_v9d | KoELECTRA v3 (35000) | tokenizer_v9 ~ tokenizer_v9d (32000) | **불일치** | ✗ |
| model_v9e ~ model_v9o | KoELECTRA v3 | tokenizer_v9e ~ tokenizer_v9o | 35000 | ✓ |
| model_v9n | KoELECTRA v3 | tokenizer_v9n | 35000 | ✓ |
| model_v19 ~ model_v21 | KoELECTRA v3 | tokenizer_v20 ~ tokenizer_v21 | 35000 | ✓ |
| **pipeline_v22/model (최종)** | **KoELECTRA v3** | **pipeline_v22/tokenizer** | **35000** | **✓** |

> v9~v9d는 실험 중 잘못된 토크나이저(vocab=32000)가 저장된 오류 버전 — 추론 시 사용 불가.  
> **실제 사용 가능한 최종 모델: `pipeline_v22/model` + `pipeline_v22/tokenizer`**

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

### 단계 7: L3 임계값 후처리 (v9n + threshold=0.69)

| threshold | L3 F1 | Macro F1 |
|-----------|-------|----------|
| 없음 | 97.83% | 98.75% |
| 0.69 | **98.21%** | **98.83%** |

재훈련 없이 L3 임계값만 조정해 목표 달성. v9n은 dedup_v2 기반(168,741건) 학습.

---

### 단계 8: OOD 개선 실험 — 데이터 보강 (v14 ~ v17)

v9n 달성 이후 **신종 감염병 등 OOD(Out-of-Distribution) 케이스 처리** 개선을 위한 데이터 보강 실험.

#### 데이터 체인 (dedup_v2 → dedup_v6)

- **dedup_v3**: L1 감염병 문자 중 중증 키워드 포함 45건 → L3 재레이블
- **dedup_v4**: 신종 감염병 합성 L3 500건 추가
- **dedup_v6**: L3=13,586건(+245 vs v2), L4=4,995건(+200 vs v2)

#### 학습 실험 결과

| 버전 | 데이터 | init_model | LR | alpha(KL) | epochs | 비마스킹 F1 | **마스킹 F1** |
|------|--------|-----------|-----|-----------|--------|------------|-------------|
| v14 | dedup_v6 | v9n | 5e-6 | 0.0 | 10 | **99.49%** | 95.90% |
| v15 | dedup_v6 | v9n | 5e-6 | 0.5 | 10 | — | 95.96% |
| v16 | dedup_v6 | v15 | 2e-6 | 1.0 | 5 | **99.48%** | 96.29% |
| v17 | dedup_v2 | v9n | 5e-6 | 0.5 | 20 | — | 95.54% |

→ dedup_v6 감염병 합성 데이터(500건) 추가 시 masked L4 F1이 87~89%대로 하락, masked MacroF1 96%대에 고착.

#### OOD 케이스 평가 (7케이스)

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

---

### 단계 9: 전략 전환 — dedup_v7 데이터 재구성 + 처음부터 학습 (v19 ~ v21)

v9n fine-tune 대신 전략을 바꿔 **COVID 편향 제거 + 처음부터 전체 재학습**으로 접근.

#### dedup_v7 데이터셋 구성

- **COVID L0 제거**: 2021년 이후 L0 COVID 문자 70% 제거 → 모델이 COVID=L0으로 암기하는 편향 완화
- **합성 데이터 추가**: COVID 키워드 없이 작성한 합성 L1 50건 + L2 50건 → OOD 일반화
- **규모**: 136,987건 (train 95,890 / val 20,548 / test 20,549)

#### 학습 실험 결과

| 버전 | 베이스 모델 | epochs | lr | Accuracy | **MacroF1** | 비고 |
|------|-----------|--------|-----|---------|-----------|------|
| v19 | KLUE-BERT (`klue/bert-base`) | 3 | 2e-5 | 97.19% | 95.72% | 기준선 |
| v20 | KoELECTRA v3 | 3 | 2e-5 | 95.97% | 93.78% | epoch 부족(과소학습) |
| v21 | KoELECTRA v3 | 5 | 2e-5 | 97.43% | **95.86%** | 최고 성능 |

v21에서 **L4 Precision 89.9%** 문제 확인 — L3↔L4 경계 혼란으로 L3 샘플 ~78건이 L4로 과잉예측.

---

### 단계 10: Ordinal Label Smoothing + LR 최적화 — 목표 달성 (v22) ✅

v21의 근본 문제인 **L3↔L4 경계 혼란**을 구조적으로 해결.

| 파라미터 | v21 | **v22** | 변경 이유 |
|----------|-----|---------|---------|
| lr | 2e-5 | **5e-5** | KoELECTRA 초기 수렴 불충분 |
| smoothing | — | **ordinal (α=0.1)** | L3↔L4 경계 혼란 구조적 해결 |
| warmup | 10% | **20%** | 높은 lr 안정화 |
| epochs | 5 | **7** | 충분한 수렴 보장 |

#### v22 학습 과정 (epoch별)

| Epoch | TrainLoss | ValLoss | Gap | ValF1 | MaskedF1 | L4 P(추정) |
|-------|-----------|---------|-----|-------|----------|----------|
| 1 | 0.4434 | 0.0157 | +0.4277 | 98.87% | 93.39% | 77.8% |
| 2 | 0.1224 | 0.0158 | +0.1066 | 98.82% | 94.24% | 81.0% |
| 3 | 0.0879 | 0.0181 | +0.0698 | 98.63% | 94.27% | 86.0% |
| 4 | 0.0504 | 0.0188 | +0.0316 | 98.77% | 94.66% | 82.5% |
| 5 | 0.0304 | 0.0158 | +0.0146 | 99.15% | 95.09% | 87.7% |
| 6 | 0.0181 | 0.0131 | +0.0050 | 99.35% | 95.15% | 90.0% |
| **7** | **0.0085** | **0.0112** | **-0.0027** | **99.44%** | **95.40%** | **89.0%** |

> Val Loss < Train Loss (Gap=-0.0027): 과적합 없음  
> Milestone log 마지막 체크포인트 기준 일반 MacroF1=99.95%, 마스킹 MacroF1=99.83% (val set)

#### v22 최종 테스트 성능 (test set 20,549건)

| 클래스 | Precision | Recall | F1 | 오분류 |
|--------|-----------|--------|----|--------|
| L0 긴급아님 | 99.8% | 99.7% | **99.7%** | 33건 (0.3%) |
| L1 낮음 | 99.7% | 98.6% | **99.2%** | 82건 (1.4%) |
| L2 중간 | 98.4% | 99.1% | **98.7%** | 17건 (0.9%) |
| L3 높음 | 96.9% | 99.5% | **98.2%** | 11건 (0.5%) |
| **L4 매우높음** | **96.9%** | **99.0%** | **97.9%** | 7건 (1.0%) |
| **Macro** | — | — | **98.75%** | 150건 (0.73%) |

**Accuracy: 99.27% / Macro F1: 98.75% — 목표 전부 달성**

---

### 단계 11: KNN OOD + 3단계 파이프라인 구축 (완성)

v22 완성 후 **신종 재난 / 비재난 텍스트 탐지** 시스템 구축.

#### KNN OOD 탐지 (sklearn NearestNeighbors)
- K=20, cosine distance, brute force
- train+val 임베딩(CLS 토큰)으로 인덱스 구축 — 116,438건
- **p99 class-level threshold**: 분류 예측 클래스에 따라 다른 임계값 적용

| 클래스 | p99 threshold |
|--------|:------------:|
| L0 | 0.018056 |
| L1 | 0.015686 |
| L2 | 0.014583 |
| L3 | 0.007599 |
| L4 | 0.003537 |

#### OOD 실험 결과 (20케이스 신종 재난 테스트)

| 지표 | 결과 |
|------|------|
| KNN OOD 거부 | 7건 / 20건 (35%) |
| 통과 후 분류 | 13건 (정답 3/13, 23.1%) |
| 전체 기본 예측 정확도 | 5/20 (25.0%) |

**한계**: 신종감염병 문자(#13)가 ID로 판정되어 OOD 탐지 실패 — 훈련 데이터에 유사한 합성 감염병 문자가 있어 분포 내로 인식됨. Confidence threshold로도 해결 불가 (모델이 85~95% 확신으로 오분류).

---

---

### 단계 12: evaluate_pipeline.py 버그 수정 3건 + 파이프라인 평가

#### 수정 사항

1. **`_groq_exhausted` 플래그 추가**: Groq 429 오류 첫 발생 시 이후 배치 모두 skip — 나머지 배치(22배치 × 2.5초)에서 반복 오류 호출 제거
2. **캐시 중간 저장**: 전체 완료 후 1회 저장 → 5배치마다 `save_cache()` 호출로 변경 — 중단 시 캐시 유실 방지
3. **dead code 삭제**: 배치 전환 후 사용되지 않는 단건 처리 함수 4개 (`_parse_label`, `call_gemini`, `call_groq_llm`, `call_llm`) 제거

#### 파이프라인 평가 결과 (T=1.5, CONF_THR=0.70, test set 20,549건)

| 구분 | Accuracy | MacroF1 |
|------|---------|---------|
| **전체 파이프라인** | **99.21%** | **98.65%** |
| v22 단독 (LLM 없음) | 99.26% | 98.73% |
| v22 직접 분류 (20,103건) | 99.82% | — |
| LLM 대상 (446건) | 71.97% | — |

> LLM 대상 446건 중 20건만 처리됨 (Gemini quota 소진, Groq TPD 99,400/100,000 초과) → 나머지 426건은 v22 예측 사용  
> LLM quota 완전 활용 시 전체 파이프라인 성능이 v22 단독(99.26%) 초과 예상

---

### 단계 13: v22 관련 파일 `pipeline_v22/` 통합 정리 (앱 개발 준비)

프로젝트 루트·`experiments/` 폴더에 분산된 v22 관련 파일(모델, 토크나이저, OOD 인덱스, 스크립트, 결과)을  
`pipeline_v22/` 단일 폴더로 통합. 각 스크립트의 내부 경로도 새 구조에 맞게 일괄 수정.

```
pipeline_v22/
├── model/               ← model_v22/ (이동)
├── tokenizer/           ← tokenizer_v22/ (이동)
├── ood/                 ← experiments/knn_ood_v22.{npz,pt} + ood_stats_v22.pt (이동)
├── scripts/             ← experiments/build_knn_ood_v22.py, build_ood_detector_v22.py (이동)
├── results/             ← 각 predict 결과 txt (이동)
├── evaluate_pipeline.py
├── predict_v22_knn_ood.py
├── predict_v22_knn_filter.py
├── predict_v22_ood.py
└── llm_cache.json
```

실험용 스크립트는 `experiments/` 폴더 참고.
