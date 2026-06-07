"""
재난문자 분류 API 서버 (KLUE-BERT 5-class)

설치: pip install fastapi uvicorn
실행: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import re
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="재난문자 분류 API")

MODEL_DIR  = "../model_v9n"
TOK_DIR    = "../tokenizer_v9n"
MAX_LENGTH = 128
LABEL_NAMES = ['긴급 아님', '낮은 긴급성', '중간 긴급성', '높은 긴급성', '매우 높은 긴급성']
UNCERTAIN_THRESH = 0.70
# L3(높은 긴급성) threshold sweep으로 결정된 값: masked F1/Recall >= 98% 달성
L3_THRESHOLD = 0.69

_ORG_PATTERN = re.compile(r'\[[^\]]{1,20}\]')


def label_to_priority(idx: int) -> str:
    if idx == 4:
        return '긴급'
    if idx in (2, 3):
        return '주의'
    return '일반'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"모델 로드 중... ({device})", flush=True)
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()
print("로드 완료", flush=True)


class ClassifyRequest(BaseModel):
    message: str


@app.post("/classify")
async def classify(req: ClassifyRequest):
    text = _ORG_PATTERN.sub('[기관]', req.message)

    inputs = tokenizer(text, truncation=True, padding='max_length',
                       max_length=MAX_LENGTH, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0]

    probs_np = probs.cpu().numpy()
    if probs_np[3] >= L3_THRESHOLD:
        pred_idx = 3
    else:
        probs_mod = probs_np.copy()
        probs_mod[3] = -1.0
        pred_idx = int(probs_mod.argmax())

    label      = LABEL_NAMES[pred_idx]
    confidence = probs_np[pred_idx]
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
    return {"status": "ok", "device": str(device)}
