"""
재난문자 분류 API — Hugging Face Spaces 배포용

HUB_MODEL_ID를 push_to_hub.py 실행 후 업로드한 모델 ID로 변경하세요.
"""

import re
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── 수정 필요 ──────────────────────────────────
HUB_MODEL_ID = "nhs0327/koelectra-disaster-v3"
# ──────────────────────────────────────────────

app = FastAPI(title="재난문자 분류 API")

MAX_LENGTH  = 96
LABEL_NAMES = ['긴급', '주의', '일반']
UNCERTAIN_THRESH = {'긴급': 0.60, '주의': 0.70, '일반': 0.70}

_ORG_PATTERN  = re.compile(r'\[[^\]]{1,20}\]')
_CERT_EMERG   = [
    '즉시 대피', '대피명령', '대피 명령', '긴급대피', '긴급 대피', '신속히 대피',
    '지진 발생', '쓰나미', '민방공 경보', '민방공경보', '테러 발생',
]
_CERT_CAUTION = [
    '호우경보', '호우주의보', '태풍경보', '태풍주의보',
    '한파경보', '한파주의보', '폭염경보', '폭염주의보',
    '대설경보', '대설주의보', '강풍경보', '강풍주의보',
    '풍랑경보', '풍랑주의보',
]
_CERT_GENERAL = ['찾습니다', '실종된']

device    = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(HUB_MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(HUB_MODEL_ID)
model.eval()


class ClassifyRequest(BaseModel):
    message: str


@app.post("/classify")
async def classify(req: ClassifyRequest):
    text = _ORG_PATTERN.sub('[기관]', req.message)

    if any(kw in text for kw in _CERT_EMERG):
        return {"label": "긴급", "confidence": 1.0, "stage": "rule", "uncertain": False}
    if any(kw in text for kw in _CERT_CAUTION):
        return {"label": "주의", "confidence": 1.0, "stage": "rule", "uncertain": False}
    if any(kw in text for kw in _CERT_GENERAL) and 'cm' in text:
        return {"label": "일반", "confidence": 1.0, "stage": "rule", "uncertain": False}

    inputs = tokenizer(text, truncation=True, padding='max_length',
                       max_length=MAX_LENGTH, return_tensors='pt')
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0]
    pred_idx   = probs.argmax().item()
    label      = LABEL_NAMES[pred_idx]
    confidence = probs[pred_idx].item()

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "stage":      "model",
        "uncertain":  confidence < UNCERTAIN_THRESH[label],
        "probs":      {LABEL_NAMES[i]: round(probs[i].item(), 4) for i in range(3)},
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
