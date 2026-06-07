import torch
import torch.nn.functional as F
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

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
print(f'장치: {device}')
tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device).eval()

df = pd.read_csv('중요파일/data/processed/data_test.csv', encoding='utf-8-sig')
df['메시지내용'] = df['메시지내용'].fillna('').apply(lambda x: _ORG.sub('[기관]', x))

preds, stages = [], []
BATCH = 64
texts = df['메시지내용'].tolist()

# Stage 1
rule_preds = [rule_classify(t) for t in texts]
model_idx  = [i for i, r in enumerate(rule_preds) if r is None]
model_texts = [texts[i] for i in model_idx]

print(f'Stage 1 판정: {len(texts)-len(model_idx)}건 / Stage 2 위임: {len(model_idx)}건')

# Stage 2
model_results = []
for i in range(0, len(model_texts), BATCH):
    batch = model_texts[i:i+BATCH]
    enc = tok(batch, truncation=True, padding='max_length',
               max_length=MAX_LENGTH, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1)
    for p in probs:
        if p[0] > THRESH_EMERG:
            model_results.append(0)
        else:
            model_results.append(p.argmax().item())

# 합치기
final = list(rule_preds)
for i, res in zip(model_idx, model_results):
    final[i] = res

y_true = df['label'].tolist()
print('\n=== 수정 후 전체 성능 (주의 규칙 제거) ===')
print(classification_report(y_true, final, target_names=LABEL_NAMES, digits=4))
