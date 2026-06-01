import torch
import torch.nn.functional as F
import pandas as pd
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

MODEL_DIR   = '중요파일/koelectra_v3_model'
MAX_LENGTH  = 96
LABEL_NAMES = ['긴급', '주의', '일반']
THRESH_EMERG = 0.12
_ORG = re.compile(r'\[[^\]]{1,20}\]')

_CERT_EMERG   = ['즉시 대피','대피명령','대피 명령','긴급대피','긴급 대피','신속히 대피',
                 '지진 발생','쓰나미','민방공 경보','민방공경보','테러 발생']
_CERT_CAUTION = ['호우경보','호우주의보','태풍경보','태풍주의보','한파경보','한파주의보',
                 '폭염경보','폭염주의보','대설경보','대설주의보','강풍경보','강풍주의보',
                 '풍랑경보','풍랑주의보']
_CERT_GENERAL = ['찾습니다','실종된']

def rule_classify(msg):
    has_emerg   = any(kw in msg for kw in _CERT_EMERG)
    has_caution = any(kw in msg for kw in _CERT_CAUTION)
    has_general = any(kw in msg for kw in _CERT_GENERAL) and 'cm' in msg
    if has_emerg and not has_caution:                     return 0
    if has_caution and not has_emerg:                     return 1
    if has_general and not has_emerg and not has_caution: return 2
    return None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device).eval()

df = pd.read_csv('중요파일/data/processed/data_test.csv', encoding='utf-8-sig')
df['메시지내용'] = df['메시지내용'].fillna('').apply(lambda x: _ORG.sub('[기관]', x))
texts  = df['메시지내용'].tolist()
y_true = df['label'].tolist()

rule_preds = [rule_classify(t) for t in texts]
model_idx  = [i for i, r in enumerate(rule_preds) if r is None]
rule_idx   = [i for i, r in enumerate(rule_preds) if r is not None]

model_texts = [texts[i] for i in model_idx]
model_preds = []
for i in range(0, len(model_texts), 64):
    batch = model_texts[i:i+64]
    enc = tok(batch, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1)
    for p in probs:
        model_preds.append(0 if p[0] > THRESH_EMERG else p.argmax().item())

final = list(rule_preds)
for i, res in zip(model_idx, model_preds):
    final[i] = res

s1_err = [(LABEL_NAMES[rule_preds[i]], LABEL_NAMES[y_true[i]]) for i in rule_idx if rule_preds[i] != y_true[i]]
s2_err = [(LABEL_NAMES[y_true[model_idx[i]]], LABEL_NAMES[model_preds[i]])
          for i in range(len(model_preds)) if model_preds[i] != y_true[model_idx[i]]]
total_err = len(s1_err) + len(s2_err)

print(f'Stage 1 판정: {len(rule_idx)}건  오탐: {len(s1_err)}건')
for (p,t), c in Counter(s1_err).items(): print(f'  {t}→{p}: {c}건')
print(f'\nStage 2 판정: {len(model_idx)}건  오분류: {len(s2_err)}건')
for (t,p), c in sorted(Counter(s2_err).items()): print(f'  {t}→{p}: {c}건')
print(f'\n전체 오분류: {total_err}건 / 32,999건')
print(f'정확도: {(32999-total_err)/32999*100:.2f}%')
print()
print(classification_report(y_true, final, target_names=LABEL_NAMES, digits=4))
