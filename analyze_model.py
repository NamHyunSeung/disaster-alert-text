import torch
import torch.nn.functional as F
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

MODEL_DIR   = '중요파일/koelectra_v3_model'
MAX_LENGTH  = 96
LABEL_NAMES = ['긴급', '주의', '일반']
THRESH_EMERG = 0.12
_ORG = re.compile(r'\[[^\]]{1,20}\]')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'장치: {device}')
tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device).eval()

df = pd.read_csv('중요파일/data/processed/data_test.csv', encoding='utf-8-sig')
df['메시지내용'] = df['메시지내용'].fillna('').apply(lambda x: _ORG.sub('[기관]', x))

preds = []
BATCH = 64
texts = df['메시지내용'].tolist()
for i in range(0, len(texts), BATCH):
    batch = texts[i:i+BATCH]
    enc = tok(batch, truncation=True, padding='max_length',
               max_length=MAX_LENGTH, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1)
    for p in probs:
        if p[0] > THRESH_EMERG:
            preds.append(0)
        else:
            preds.append(p.argmax().item())
    if i % 5000 == 0:
        print(f'  {i}/{len(texts)}')

y_true = df['label'].tolist()
print('\n=== KoELECTRA-v3 전체 성능 (Test set) ===')
print(classification_report(y_true, preds, target_names=LABEL_NAMES, digits=4))

cm = confusion_matrix(y_true, preds)
print('=== 혼동 행렬 ===')
print(f'{"":>8}  {"긴급예측":>8}  {"주의예측":>8}  {"일반예측":>8}')
for i, name in enumerate(LABEL_NAMES):
    print(f'{name+"(실제)":>8}  {cm[i][0]:>8}  {cm[i][1]:>8}  {cm[i][2]:>8}')
