"""
model_v19 + dedup_v7 검증 세트 오분류 샘플 추출.
신종호흡기(L1→L0), 바이러스변종(L2→L0) 개선 여부 확인.
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '완성 모델', 'src'))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset_v2 import load_and_split_v2, DisasterDatasetAug
from collections import defaultdict

MODEL_DIR = os.path.join(ROOT, 'model_v19')
TOK_DIR   = os.path.join(ROOT, 'tokenizer')
DATA_PATH = os.path.join(ROOT, '중요파일', 'data', 'raw',
                         '재난문자_레이블링결과_dedup_v7.xlsx')
MAX_LEN    = 128
BATCH_SIZE = 128

# v9n OOD 미탐지 5개 — v19에서 개선 여부 확인
OOD_MISSED = [
    (3, 1, "태양폭풍으로 GPS통신위성 오작동. 항공선박 운항 위험. 야외 활동 자제."),
    (3, 1, "대규모 산불 확산 중. 바람 방향 주의 요망. 연기 흡입 피해야."),
    (1, 0, "신종 호흡기 바이러스 지역사회 전파 확인. 마스크 착용 손씻기 생활화."),
    (2, 0, "바이러스 변종 출현. 기존 백신 효능 불명확. 외출 최소화 권고."),
    (3, 1, "도심 집중호우로 지하차도 침수. 우회 도로 이용 바람."),
]

LABEL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("데이터 로드 중...")
_, val_df, _ = load_and_split_v2(DATA_PATH)
print(f"  검증 세트: {len(val_df)}개")

print("모델/토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device); model.eval()

val_ds     = DisasterDatasetAug(val_df, tokenizer, MAX_LEN, augment=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("추론 중...")
all_preds, all_labels = [], []
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        ids   = batch['input_ids'].to(device)
        mask  = batch['attention_mask'].to(device)
        ttids = batch['token_type_ids'].to(device)
        lbls  = batch['label']

        logits = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
        preds  = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(lbls.tolist())
        print(f"\r  {min((i+1)*BATCH_SIZE, len(val_df))}/{len(val_df)}", end='', flush=True)
print()

# 오분류 추출
misclassified = []
for idx, (true, pred) in enumerate(zip(all_labels, all_preds)):
    if true != pred:
        text = val_df.iloc[idx]['메시지내용']
        misclassified.append((true, pred, text))

n_val   = len(val_df)
n_wrong = len(misclassified)
print(f"\n전체 오분류: {n_wrong}개 / {n_val}개 ({n_wrong/n_val*100:.2f}%)")

# 패턴별 그룹화
groups = defaultdict(list)
for true, pred, text in misclassified:
    groups[(true, pred)].append(text)

missed_patterns = set((t, p) for t, p, _ in OOD_MISSED)

lines = []

# OOD 5개 — v19에서 어떻게 예측하는지
lines.append("=" * 90)
lines.append("[ v9n OOD 미탐지 5개 → v19 예측 결과 ]")
lines.append("=" * 90)
with torch.no_grad():
    for true_lbl, old_pred, text in OOD_MISSED:
        enc = tokenizer(text, return_tensors='pt', truncation=True,
                        max_length=MAX_LEN, padding='max_length')
        enc = {k: v.to(device) for k, v in enc.items()}
        if 'token_type_ids' not in enc:
            enc['token_type_ids'] = torch.zeros_like(enc['input_ids'])
        logits = model(**enc).logits
        new_pred = logits.argmax(dim=-1).item()
        status = "OK" if new_pred == true_lbl else f"오답(L{new_pred})"
        lines.append(f"  정답 L{true_lbl} | v9n→L{old_pred} | v19→{status}  |  {text}")

lines.append("")
lines.append("=" * 90)
lines.append("[ 검증 세트 전체 오분류 현황 ]")
lines.append("=" * 90)
for (true, pred), samples in sorted(groups.items(), key=lambda x: -len(x[1])):
    marker = " ← 신종호흡기/바이러스변종 패턴" if (true, pred) in missed_patterns else ""
    lines.append(f"  L{true} → L{pred} : {len(samples):3d}개{marker}")

lines.append("")
lines.append("=" * 90)
lines.append("[ 동일 패턴 (v9n OOD 미탐지와 같은 L1→L0, L2→L0) 샘플 ]")
lines.append("=" * 90)
for (true, pred) in sorted(missed_patterns):
    samples = groups.get((true, pred), [])
    lines.append(f"\n--- L{true} → L{pred}  ({len(samples)}개) ---")
    for text in samples[:20]:
        lines.append(f"  {text}")
    if len(samples) > 20:
        lines.append(f"  ... (이하 {len(samples)-20}개 생략)")

output = "\n".join(lines)
out_path = os.path.join(ROOT, '실험', 'misclassified_v19.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(output)

print(f"\n-> {out_path} 저장 완료\n")
for (true, pred), samples in sorted(groups.items(), key=lambda x: -len(x[1])):
    marker = " <- OOD missed pattern" if (true, pred) in missed_patterns else ""
    print(f"  L{true} -> L{pred} : {len(samples):3d}{marker}")
