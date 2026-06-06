"""
v9n 검증 세트 오분류 샘플 추출.
OOD 미탐지 5개와 같은 오분류 패턴(true→pred)을 함께 출력.
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '완성 모델', 'src'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset_v2 import load_and_split_v2, DisasterDatasetAug

MODEL_DIR = os.path.join(ROOT, 'model_v9n')
TOK_DIR   = os.path.join(ROOT, 'tokenizer_v9n')
DATA_PATH = os.path.join(ROOT, '중요파일', 'data', 'raw',
                         '재난문자_레이블링결과_dedup_v2.xlsx')
MAX_LEN    = 96
BATCH_SIZE = 128

# OOD 미탐지 5개 (정답, 예측, 텍스트)
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

print("모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device); model.eval()

val_ds     = DisasterDatasetAug(val_df, tokenizer, MAX_LEN, augment=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("추론 중...")
all_preds, all_labels = [], []
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        ttids = batch['token_type_ids'].to(device)
        lbls = batch['label']

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
        text = val_df.iloc[idx]['text']
        misclassified.append((true, pred, text))

print(f"\n전체 오분류: {len(misclassified)}개 / {len(val_df)}개 "
      f"({len(misclassified)/len(val_df)*100:.2f}%)\n")

# OOD 미탐지 5개의 혼동 패턴
missed_patterns = set((t, p) for t, p, _ in OOD_MISSED)

# 패턴별로 그룹화
from collections import defaultdict
groups = defaultdict(list)
for true, pred, text in misclassified:
    groups[(true, pred)].append(text)

lines = []

lines.append("=" * 90)
lines.append("[ OOD 미탐지 5개 ]  (kNN이 in-distribution으로 판정, 실제 오답)")
lines.append("=" * 90)
for true, pred, text in OOD_MISSED:
    lines.append(f"  정답 L{true} → 예측 L{pred}  |  {text}")

lines.append("")
lines.append("=" * 90)
lines.append("[ 검증 세트 오분류 — 동일 패턴 ]  (같은 true→pred 조합)")
lines.append("=" * 90)

for (true, pred) in sorted(missed_patterns):
    samples = groups.get((true, pred), [])
    lines.append(f"\n--- L{true} → L{pred}  ({len(samples)}개) ---")
    for text in samples[:20]:  # 패턴당 최대 20개
        lines.append(f"  {text}")
    if len(samples) > 20:
        lines.append(f"  ... (이하 {len(samples)-20}개 생략)")

lines.append("")
lines.append("=" * 90)
lines.append("[ 검증 세트 전체 오분류 현황 ]")
lines.append("=" * 90)
for (true, pred), samples in sorted(groups.items(), key=lambda x: -len(x[1])):
    marker = " ← OOD 미탐지 패턴" if (true, pred) in missed_patterns else ""
    lines.append(f"  L{true} → L{pred} : {len(samples):3d}개{marker}")

output = "\n".join(lines)

out_path = os.path.join(ROOT, '실험', 'misclassified_v9n.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(output)

# stdout은 ASCII만 출력
print(f"-> {out_path} 저장 완료")
for (true, pred), samples in sorted(groups.items(), key=lambda x: -len(x[1])):
    marker = " <- OOD missed" if (true, pred) in missed_patterns else ""
    print(f"  L{true} -> L{pred} : {len(samples):3d}{marker}")
