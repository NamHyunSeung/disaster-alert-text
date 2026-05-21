import torch
import torch.nn.functional as F
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR   = '중요파일/koelectra_v3_model'
MAX_LENGTH  = 96
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
texts  = df['메시지내용'].tolist()
y_true = df['label'].tolist()

# 모델 예측 (Stage 2만)
rule_preds = [rule_classify(t) for t in texts]
model_idx  = [i for i, r in enumerate(rule_preds) if r is None]
model_texts = [texts[i] for i in model_idx]

all_probs = []
for i in range(0, len(model_texts), 64):
    batch = model_texts[i:i+64]
    enc = tok(batch, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1)
    all_probs.extend(probs.cpu().tolist())

model_preds = [0 if p[0] > THRESH_EMERG else p.index(max(p)) for p in all_probs]

# 일반→주의 오분류 케이스 추출
cases = []
for i, (idx, pred, prob) in enumerate(zip(model_idx, model_preds, all_probs)):
    true = y_true[idx]
    if true == 2 and pred == 1:  # 일반→주의
        cases.append({
            'msg': texts[idx],
            '주의확률': round(prob[1]*100, 1),
            '일반확률': round(prob[2]*100, 1),
        })

cases.sort(key=lambda x: -x['주의확률'])

print(f'일반→주의 오분류 총 {len(cases)}건\n')
print('=== 확신도 높은 상위 20건 ===')
for i, c in enumerate(cases[:20], 1):
    print(f'[{i:02d}] 주의{c["주의확률"]}% / 일반{c["일반확률"]}%')
    print(f'     {c["msg"][:100]}')
    print()

# 확신도 분포
bins = [0,50,60,70,80,90,100]
import collections
dist = collections.Counter()
for c in cases:
    for b in range(len(bins)-1):
        if bins[b] <= c['주의확률'] < bins[b+1]:
            dist[f'{bins[b]}~{bins[b+1]}%'] += 1
            break
    else:
        dist['100%'] += 1

print('=== 주의 확신도 분포 ===')
for k in [f'{bins[i]}~{bins[i+1]}%' for i in range(len(bins)-1)]:
    print(f'  {k}: {dist[k]}건')
