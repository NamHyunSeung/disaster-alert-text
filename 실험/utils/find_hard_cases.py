"""
판단하기 어려운 재난문자 탐색
- 모델 확신도가 낮은 경우 (top prob < 70%)
- 1위와 2위 클래스 차이가 작은 경우 (margin < 20%)
- 오분류된 경우
"""

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR   = "koelectra_model"
MAX_LENGTH  = 128
LABEL_NAMES = ['긴급', '주의', '일반']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device).eval()

def predict_batch(texts):
    inputs = tokenizer(
        texts, truncation=True, padding='max_length',
        max_length=MAX_LENGTH, return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
    return probs

# 테스트 데이터 로드
df = pd.read_csv('data_test.csv', encoding='utf-8-sig').fillna({'메시지내용': ''})
texts  = df['메시지내용'].tolist()
labels = df['label'].tolist()

# 배치 예측
BATCH = 64
all_probs = []
for i in range(0, len(texts), BATCH):
    all_probs.extend(predict_batch(texts[i:i+BATCH]))

import numpy as np
all_probs = np.array(all_probs)
preds     = all_probs.argmax(axis=1)
top_prob  = all_probs.max(axis=1)
margin    = np.sort(all_probs, axis=1)[:, -1] - np.sort(all_probs, axis=1)[:, -2]

df['pred']       = preds
df['pred_name']  = [LABEL_NAMES[p] for p in preds]
df['top_prob']   = top_prob
df['margin']     = margin
df['correct']    = df['label'] == df['pred']
for i, name in enumerate(LABEL_NAMES):
    df[f'p_{name}'] = all_probs[:, i]

out = []

# ── 1. 오분류 케이스 ──────────────────────────────────
errors = df[~df['correct']].sort_values('margin')
out.append("=" * 65)
out.append("1. 오분류 케이스 (실제 ≠ 예측)")
out.append("=" * 65)
for _, r in errors.head(15).iterrows():
    out.append(
        f"\n[실제: {LABEL_NAMES[r['label']]} → 예측: {r['pred_name']}]  "
        f"확신도 {r['top_prob']*100:.1f}%  margin {r['margin']*100:.1f}%"
    )
    out.append(f"긴급 {r['p_긴급']*100:.1f}%  주의 {r['p_주의']*100:.1f}%  "
               f"일반 {r['p_일반']*100:.1f}%")
    out.append(r['메시지내용'][:200])

# ── 2. 확신도 낮은 케이스 (경계선) ───────────────────
low_conf = df[df['correct'] & (df['top_prob'] < 0.70)].sort_values('top_prob')
out.append("\n\n" + "=" * 65)
out.append("2. 확신도 낮은 케이스 (정답이지만 top prob < 70%)")
out.append("=" * 65)
for _, r in low_conf.head(15).iterrows():
    out.append(
        f"\n[{r['pred_name']}]  확신도 {r['top_prob']*100:.1f}%  margin {r['margin']*100:.1f}%"
    )
    out.append(f"긴급 {r['p_긴급']*100:.1f}%  주의 {r['p_주의']*100:.1f}%  "
               f"일반 {r['p_일반']*100:.1f}%")
    out.append(r['메시지내용'][:200])

# ── 3. 긴급↔주의 경계 케이스 ─────────────────────────
boundary = df[
    (df['p_긴급'] > 0.15) & (df['p_주의'] > 0.15) &
    (df['label_name'].isin(['긴급', '주의']))
].sort_values('margin')
out.append("\n\n" + "=" * 65)
out.append("3. 긴급 ↔ 주의 경계 케이스")
out.append("=" * 65)
for _, r in boundary.head(15).iterrows():
    out.append(
        f"\n[실제: {LABEL_NAMES[r['label']]}]  "
        f"긴급 {r['p_긴급']*100:.1f}%  주의 {r['p_주의']*100:.1f}%  "
        f"일반 {r['p_일반']*100:.1f}%"
    )
    out.append(r['메시지내용'][:200])

text = '\n'.join(out)
with open('hard_cases.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print("저장 완료: hard_cases.txt")
