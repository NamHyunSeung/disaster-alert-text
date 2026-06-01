"""
재난문자 분류 API — Hugging Face Spaces 배포용 (KLUE-BERT 5-class)

HUB_MODEL_ID를 push_to_hub.py 실행 후 업로드한 모델 ID로 변경하세요.
"""

import re
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── 수정 필요 ──────────────────────────────────
HUB_MODEL_ID = "nhs0327/klue-bert-disaster-v1"
# ──────────────────────────────────────────────

app = FastAPI(title="재난문자 분류 API")

MAX_LENGTH  = 128
LABEL_NAMES = ['긴급 아님', '낮은 긴급성', '중간 긴급성', '높은 긴급성', '매우 높은 긴급성']
UNCERTAIN_THRESH = 0.70
L3_THRESHOLD = 0.69

_ORG_PATTERN = re.compile(r'\[[^\]]{1,20}\]')


def label_to_priority(idx: int) -> str:
    if idx == 4:
        return '긴급'
    if idx in (2, 3):
        return '주의'
    return '일반'


device    = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(HUB_MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(HUB_MODEL_ID)
model.eval()


class ClassifyRequest(BaseModel):
    message: str


@app.post("/classify")
async def classify(req: ClassifyRequest):
    text = _ORG_PATTERN.sub('[기관]', req.message)

    inputs = tokenizer(text, truncation=True, padding='max_length',
                       max_length=MAX_LENGTH, return_tensors='pt')
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0]

    import numpy as np
    probs_np = probs.cpu().numpy()
    if probs_np[3] >= L3_THRESHOLD:
        pred_idx = 3
    else:
        probs_mod = probs_np.copy()
        probs_mod[3] = -1.0
        pred_idx = int(probs_mod.argmax())

    label      = LABEL_NAMES[pred_idx]
    confidence = float(probs_np[pred_idx])
    priority   = label_to_priority(pred_idx)

    return {
        "label":      label,
        "priority":   priority,
        "confidence": round(confidence, 4),
        "stage":      "model",
        "uncertain":  confidence < UNCERTAIN_THRESH,
        "probs":      {LABEL_NAMES[i]: round(probs[i].item(), 4) for i in range(5)},
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
