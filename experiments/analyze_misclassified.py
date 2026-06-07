"""
마스킹 후 오분류 케이스 분석 — 96~97% ceiling 원인 파악
"""
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset_v2 import load_and_split_v2, DisasterDatasetAug, _mask_text
from model import load_model

LABEL_NAMES = ['긴급아님', '낮음', '중간', '높음', '매우높음']
MODEL_DIR = 'model_v9i'
TOK_DIR   = 'tokenizer_v9i'
DATA_PATH = '중요파일/data/raw/재난문자_레이블링결과_dedup_v3.xlsx'
TARGET_LABELS = [2, 3, 4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("데이터 로드 중...")
_, _, test_df = load_and_split_v2(DATA_PATH, seed=42)

tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model = load_model(MODEL_DIR, num_labels=5).to(device)
model.eval()

# 마스킹된 test 데이터
masked_test_df = test_df.copy()
masked_test_df['text'] = masked_test_df['text'].apply(_mask_text)
masked_test_df['orig_text'] = test_df['text'].values

masked_ds = DisasterDatasetAug(masked_test_df, tokenizer, 128, augment=False)
masked_loader = DataLoader(masked_ds, batch_size=64, shuffle=False, num_workers=0)

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in masked_loader:
        ids   = batch['input_ids'].to(device)
        mask  = batch['attention_mask'].to(device)
        ttids = batch['token_type_ids'].to(device)
        logits = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
        preds  = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch['label'].numpy())

preds_arr  = np.array(all_preds)
labels_arr = np.array(all_labels)

print(f"\n{'='*70}")
print("마스킹 후 오분류 케이스 분석 (L2/L3/L4)")
print(f"{'='*70}\n")

for true_lbl in TARGET_LABELS:
    idxs = np.where((labels_arr == true_lbl) & (preds_arr != true_lbl))[0]
    print(f"\n[Label {true_lbl} ({LABEL_NAMES[true_lbl]}) 오분류 - {len(idxs)}건]")
    print("-"*70)

    # 오분류 분포
    wrong_preds = preds_arr[idxs]
    from collections import Counter
    dist = Counter(wrong_preds)
    for pred_lbl, cnt in sorted(dist.items()):
        print(f"  → Label {pred_lbl} ({LABEL_NAMES[pred_lbl]}): {cnt}건")

    print("\n  [샘플 케이스 (최대 10개)]")
    sample_idxs = idxs[:10]
    for i, idx in enumerate(sample_idxs):
        orig_text   = masked_test_df.iloc[idx]['orig_text']
        masked_text = masked_test_df.iloc[idx]['text']
        pred_lbl    = preds_arr[idx]
        print(f"\n  {i+1}. 실제: L{true_lbl}({LABEL_NAMES[true_lbl]}) → 예측: L{pred_lbl}({LABEL_NAMES[pred_lbl]})")
        print(f"     원본:   {orig_text[:100]}")
        print(f"     마스킹: {masked_text[:100]}")

print(f"\n\n{'='*70}")
print("경계 클래스 간 오분류 패턴 (L2↔L3↔L4)")
print(f"{'='*70}")

# 혼동 행렬 (L2/L3/L4만)
from sklearn.metrics import confusion_matrix
target_mask = np.isin(labels_arr, TARGET_LABELS)
cm = confusion_matrix(labels_arr[target_mask], preds_arr[target_mask], labels=TARGET_LABELS)
print("\n혼동 행렬 (행=실제, 열=예측):")
header = "          " + "  ".join([f"예측L{l}" for l in TARGET_LABELS])
print(header)
for i, true_lbl in enumerate(TARGET_LABELS):
    row = f"실제L{true_lbl}({LABEL_NAMES[true_lbl]:4s})  " + "  ".join([f"{cm[i,j]:6d}" for j in range(len(TARGET_LABELS))])
    print(row)

# 마스킹 후 남은 텍스트 길이 분석
print(f"\n\n{'='*70}")
print("마스킹 후 텍스트 길이 분포 (L2/L3/L4)")
print(f"{'='*70}")
for lbl in TARGET_LABELS:
    lbl_mask = labels_arr == lbl
    texts = masked_test_df.iloc[np.where(lbl_mask)[0]]['text'].values
    lengths = [len(t) for t in texts]
    correct_mask = preds_arr[lbl_mask] == lbl
    wrong_mask   = ~correct_mask
    correct_len  = [len(t) for t, c in zip(texts, correct_mask) if c]
    wrong_len    = [len(t) for t, c in zip(texts, correct_mask) if not c]
    print(f"\nLabel {lbl} ({LABEL_NAMES[lbl]}):")
    print(f"  전체 평균 길이: {np.mean(lengths):.1f}자")
    if correct_len: print(f"  정답 평균 길이: {np.mean(correct_len):.1f}자 ({len(correct_len)}건)")
    if wrong_len:   print(f"  오답 평균 길이: {np.mean(wrong_len):.1f}자 ({len(wrong_len)}건)")

# 빈 마스킹 (모든 키워드 제거 후 매우 짧은) 케이스
print(f"\n\n{'='*70}")
print("마스킹 후 30자 미만 짧은 텍스트 (정보 소실 케이스)")
print(f"{'='*70}")
for lbl in TARGET_LABELS:
    lbl_idxs = np.where(labels_arr == lbl)[0]
    short_idxs = [i for i in lbl_idxs if len(masked_test_df.iloc[i]['text']) < 30]
    wrong_short = [i for i in short_idxs if preds_arr[i] != lbl]
    print(f"Label {lbl}: 30자 미만 {len(short_idxs)}건, 이 중 오분류 {len(wrong_short)}건 ({len(wrong_short)/max(len(short_idxs),1)*100:.1f}%)")
    for idx in wrong_short[:3]:
        print(f"  원본:   {masked_test_df.iloc[idx]['orig_text'][:80]}")
        print(f"  마스킹: '{masked_test_df.iloc[idx]['text']}'  → 예측 L{preds_arr[idx]}")
