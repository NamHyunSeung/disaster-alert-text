import torch
import torch.nn.functional as F
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR   = '중요파일/koelectra_v3_model'
MAX_LENGTH  = 96
LABEL_NAMES = ['긴급', '주의', '일반']
THRESH_EMERG = 0.12
_ORG = re.compile(r'\[[^\]]{1,20}\]')

_CERT_EMERG   = ['즉시 대피','대피명령','대피 명령','긴급대피','긴급 대피','신속히 대피',
                 '지진 발생','쓰나미','민방공 경보','민방공경보','테러 발생']
_CERT_GENERAL = ['찾습니다','실종된']

def rule_classify(msg):
    if any(kw in msg for kw in _CERT_EMERG): return 0
    if any(kw in msg for kw in _CERT_GENERAL) and 'cm' in msg: return 2
    return None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device).eval()

df = pd.read_csv('중요파일/data/processed/data_test.csv', encoding='utf-8-sig')
df['메시지내용'] = df['메시지내용'].fillna('').apply(lambda x: _ORG.sub('[기관]', x))
texts = df['메시지내용'].tolist()
y_true = df['label'].tolist()

# Stage 1
rule_preds = [rule_classify(t) for t in texts]
model_idx  = [i for i, r in enumerate(rule_preds) if r is None]
rule_idx   = [i for i, r in enumerate(rule_preds) if r is not None]

# Stage 1 오탐
s1_errors = [(i, rule_preds[i], y_true[i]) for i in rule_idx if rule_preds[i] != y_true[i]]
print(f'=== Stage 1 ===')
print(f'판정 건수: {len(rule_idx)}건')
print(f'오탐 건수: {len(s1_errors)}건')
for pred, true, cnt in [(p,t,1) for _,p,t in s1_errors]:
    pass
from collections import Counter
s1_err_types = Counter((LABEL_NAMES[p], LABEL_NAMES[t]) for _,p,t in s1_errors)
for (pred, true), cnt in sorted(s1_err_types.items()):
    print(f'  {true}→{pred} 오분류: {cnt}건')

# Stage 2
model_texts = [texts[i] for i in model_idx]
model_true  = [y_true[i] for i in model_idx]
model_preds = []
BATCH = 64
for i in range(0, len(model_texts), BATCH):
    batch = model_texts[i:i+BATCH]
    enc = tok(batch, truncation=True, padding='max_length',
               max_length=MAX_LENGTH, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1)
    for p in probs:
        model_preds.append(0 if p[0] > THRESH_EMERG else p.argmax().item())

s2_errors = [(model_true[i], model_preds[i]) for i in range(len(model_preds)) if model_true[i] != model_preds[i]]
print(f'\n=== Stage 2 ===')
print(f'판정 건수: {len(model_idx)}건')
print(f'오분류 건수: {len(s2_errors)}건')
s2_err_types = Counter((LABEL_NAMES[t], LABEL_NAMES[p]) for t,p in s2_errors)
for (true, pred), cnt in sorted(s2_err_types.items()):
    print(f'  {true}→{pred} 오분류: {cnt}건')

print(f'\n=== 전체 ===')
print(f'총 오분류: {len(s1_errors)+len(s2_errors)}건 / 32,999건')
print(f'전체 정확도: {(32999-len(s1_errors)-len(s2_errors))/32999*100:.2f}%')
