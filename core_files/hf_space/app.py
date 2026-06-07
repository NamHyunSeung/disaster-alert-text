"""
재난문자 분류 API — Hugging Face Spaces 배포용
  v22 (KLUE-BERT 5-class) + KNN-OOD 탐지 + Temperature Scaling + LLM fallback
  LLM 순서: OpenRouter → Gemini 2.5 Flash → Groq → Cerebras (실패 시 순차 전환, 모두 실패하면 v22 예측 사용)

model/ , tokenizer/ , knn_ood_v22.npz / knn_ood_v22_meta.pt 는 app.py와 같은 폴더에 두세요 (Space에 git-lfs로 업로드).
LLM API 키는 Space의 Secrets(GEMINI_API_KEY / GROQ_API_KEY / CEREBRAS_API_KEY / OPENROUTER_API_KEY)에 등록하세요.
"""

import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.neighbors import NearestNeighbors

try:
    from google import genai as _genai_module
    from google.genai import types as _genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

MODEL_DIR     = "model"
TOKENIZER_DIR = "tokenizer"

EMB_FILE  = "knn_ood_v22.npz"
META_FILE = "knn_ood_v22_meta.pt"

MAX_LENGTH  = 96
TEMPERATURE = 1.5
CONF_THR    = 0.70
LABEL_NAMES = ['긴급 아님', '낮은 긴급성', '중간 긴급성', '높은 긴급성', '매우 높은 긴급성']
class_thresholds = {0: 0.018056, 1: 0.015686, 2: 0.014583, 3: 0.007599, 4: 0.003537}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
CEREBRAS_MODEL   = "gpt-oss-120b"
CEREBRAS_URL     = "https://api.cerebras.ai/v1/chat/completions"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = "openai/gpt-oss-120b:free"
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

_ORG_PATTERN = re.compile(r'\[[^\]]{1,20}\]')

_SYS = """재난문자를 긴급도 L0~L4로 분류하는 전문가입니다.
키워드가 아닌 상황 심각도·행동 즉시성·생명위협 여부로 판단합니다.

【L0】행동 불필요
  재난 해제·훈련 종료·경보 해제·행사 안내

【L1】주의·정보 전달 (행동 변화 불필요)
  기상 예보, 발생 알림, 주의 권고 없는 정보

【L2】사회 기능 마비·행정 조치 (직접 생명위협 없음)
  기능이 제한·마비되거나 행정 조치가 필요하지만 생명 위협 없음
  예: 사이버공격으로 금융·전력·통신 인프라 마비 + 대안 사용 권고
  예: 수돗물 제한급수·음용 금지, 집합금지·시설폐쇄·방역협조
  예: 바이러스·전염병·해충 매개 감염 확산으로 야외 접근 금지 또는 활동 자제
  ※ 바이러스·전염병·진드기·해충 관련 접근 금지·활동 자제는 항상 L2
  판단 힌트: "현금 사용 권고", "생수 배급", "서비스 중단", "제한급수", "바이러스 확산", "진드기·해충 매개"

【L3】재난 진행 중 + 예방적 자제 권고 (즉각 이동 지시 없음)
  위험이 존재하지만 즉각 이동보다는 자제·주의·대기 수준
  예: 정전으로 신호등 불통 → 이동 자제, 원인불명 사고 현장 → 외부 대기
  예: 태양폭풍·통신 두절 → 이동 자제, 야외 활동 자제, 창문 닫기
  판단 힌트: "이동 자제", "야외 자제", "외부 대기", "접근 금지"(비감염성·물리적 위험)

【L4】즉각 대피·이동 지시 (생명 직접 위협)
  지금 당장 이동·대피하지 않으면 사망 가능한 상황
  예: 쓰나미·해수면 급상승 → 즉시 고지대 이동
  예: 댐 붕괴·가스관 폭발·방사성 물질 누출 → 즉시 대피
  예: 기온 48도 등 생명 직결 극한 환경 → 외출 금지
  판단 힌트: "즉시", "지금 즉시", "즉각", "대피소로", "고지대로", "실내 대피"
  특수 규칙: 방사성·핵·폭발·독성물질 누출은 "접근 금지"만 있어도 L4

【주의: 같은 재난 소재라도 "단계어"에 따라 L0~L4 전 구간에 분포】
  산불·산사태·폭염·한파·호우·강풍·감염병 등은 표현이 비슷해도 경보 단계에 따라 등급이 갈립니다.
  문장에 등장하는 단계어를 우선 단서로 삼으세요.
  예: "관심"/"해제"/단순 통계·신청·동선 안내 → L0
  예: "주의"·"주의보"·일반 예방 권고("~하세요","~바랍니다") → L1~L2
  예: "경보"·"위기경보 경계 단계" + 구체적 금지·자제 행동요령 → L2~L3
  예: "위기경보 심각 단계" 또는 "지금 즉시 대피"형 명령 → L3~L4
  ※ 감염병(코로나19 등)도 예외 아님: 확진자 수·검사 안내처럼 평범해 보여도
     "심각 단계", "이동·모임 강력 자제" 등 고강도 표현이 동반되면 L2 이상(드물게 L4)일 수 있음

【주의: 위협 표현 없는 행정·정보성 공지는 기본 L0】
  순환정전 시행 지역 안내, 확진자 이동경로(동선) 안내, 경보·통제 해제,
  운행 중단/재개, 지원금·신청 안내, 실종자 수배 등은
  특별한 위협·행동지시 표현이 없으면 L0으로 분류

【경계 구분 규칙】
  L2 vs L3: L2=기능 마비(생명위협 없음), L3=위험 진행 중 + 신체 자제 권고
  L3 vs L4: L3=자제·주의 권고, L4="즉시/지금 당장" 이동·대피 명령 또는 생명직결 위험 물질

반드시 JSON 형식으로만 답변: {"label": "L2"}"""


def label_to_priority(idx: int) -> str:
    if idx == 4:
        return '긴급'
    if idx in (2, 3):
        return '주의'
    return '일반'


def _build_user_msg(text: str) -> str:
    return (
        f'다음 재난문자를 분류하세요.\n\n"{text}"\n\n'
        '반드시 JSON 형식으로만 답변: {"label": "L?"}'
    )


def _parse_label(content: str) -> int:
    m = re.search(r'[Ll]([0-4])', content or '')
    return int(m.group(1)) if m else -1


# ── 모델 / KNN-OOD 로드 (Space 기동 시 1회) ─────────────────────────────
app = FastAPI(title="재난문자 분류 API (v22 + OOD + LLM)")

device    = torch.device("cpu")
print("모델 로드 중...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

print("KNN-OOD 인덱스 로드 중...", flush=True)
_emb_data = np.load(EMB_FILE)
_ood_meta = torch.load(META_FILE, weights_only=False)
_knn = NearestNeighbors(n_neighbors=_ood_meta['k'], algorithm='brute',
                        metric='cosine', n_jobs=-1)
_knn.fit(_emb_data['embs'])
print(f"로드 완료 (K={_ood_meta['k']}, 학습 임베딩 {len(_emb_data['embs']):,}개)", flush=True)

_gemini_model = None
if _GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        _gemini_model = _genai_module.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[경고] Gemini 초기화 실패: {e}", flush=True)
_groq = Groq(api_key=GROQ_API_KEY) if (_GROQ_AVAILABLE and GROQ_API_KEY) else None


# ── LLM fallback (단건 호출, 우선순위: OpenRouter → Gemini → Groq → Cerebras) ──
def _call_openrouter(text: str) -> int:
    if not OPENROUTER_API_KEY:
        return -1
    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://github.com/disaster-classifier",
                     "X-Title": "disaster-msg-classifier"},
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "system", "content": _SYS},
                             {"role": "user", "content": _build_user_msg(text)}],
                "temperature": 0.0,
                "reasoning_effort": "low",
                "max_tokens": 530,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""
        return _parse_label(content)
    except Exception as e:
        print(f"  [OpenRouter 오류] {e}", flush=True)
        return -1


def _call_gemini(text: str) -> int:
    if _gemini_model is None:
        return -1
    try:
        resp = _gemini_model.models.generate_content(
            model=GEMINI_MODEL,
            contents=_build_user_msg(text),
            config=_genai_types.GenerateContentConfig(
                system_instruction=_SYS,
                temperature=0.0,
                max_output_tokens=75,
            ),
        )
        return _parse_label(resp.text or "")
    except Exception as e:
        print(f"  [Gemini 오류] {e}", flush=True)
        return -1


def _call_groq(text: str) -> int:
    if _groq is None:
        return -1
    try:
        resp = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user", "content": _build_user_msg(text)}],
            temperature=0.0,
            max_tokens=75,
        )
        return _parse_label(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"  [Groq 오류] {e}", flush=True)
        return -1


def _call_cerebras(text: str) -> int:
    if not CEREBRAS_API_KEY:
        return -1
    try:
        resp = requests.post(
            CEREBRAS_URL,
            headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": CEREBRAS_MODEL,
                "messages": [{"role": "system", "content": _SYS},
                             {"role": "user", "content": _build_user_msg(text)}],
                "temperature": 0.0,
                "reasoning_effort": "low",
                "max_tokens": 1030,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""
        return _parse_label(content)
    except Exception as e:
        print(f"  [Cerebras 오류] {e}", flush=True)
        return -1


def _call_llm_fallback(text: str) -> int:
    for fn in (_call_openrouter, _call_gemini, _call_groq, _call_cerebras):
        pred = fn(text)
        if pred >= 0:
            return pred
    return -1


class ClassifyRequest(BaseModel):
    message: str


@app.post("/classify")
async def classify(req: ClassifyRequest):
    text = _ORG_PATTERN.sub('[기관]', req.message)

    inputs = tokenizer(text, truncation=True, padding='max_length',
                       max_length=MAX_LENGTH, return_tensors='pt')
    with torch.no_grad():
        out      = model(**inputs, output_hidden_states=True)
        probs    = F.softmax(out.logits / TEMPERATURE, dim=-1)[0]
        cls_emb  = out.hidden_states[-1][:, 0, :].numpy().astype(np.float32)

    probs_np   = probs.cpu().numpy()
    pred_idx   = int(probs_np.argmax())
    confidence = float(probs_np[pred_idx])

    dist, _   = _knn.kneighbors(cls_emb)
    knn_score = float(dist.mean())
    is_ood    = knn_score > class_thresholds.get(pred_idx, 0.018056)
    low_conf  = (not is_ood) and confidence < CONF_THR
    need_llm  = is_ood or low_conf

    stage = "model"
    if need_llm:
        llm_pred = _call_llm_fallback(text)
        if llm_pred >= 0:
            pred_idx   = llm_pred
            confidence = float(probs_np[pred_idx])
            stage      = "llm"

    label    = LABEL_NAMES[pred_idx]
    priority = label_to_priority(pred_idx)

    return {
        "label":      label,
        "priority":   priority,
        "confidence": round(confidence, 4),
        "stage":      stage,
        "uncertain":  need_llm and stage == "model",
        "ood":        bool(is_ood),
        "probs":      {LABEL_NAMES[i]: round(float(probs_np[i]), 4) for i in range(5)},
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
