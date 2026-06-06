"""
v22 테스트 데이터에서 confidence threshold별 성능 분석.
각 threshold에서 v22 오분류율과 LLM fallback 비율의 트레이드오프를 출력.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '완성 모델', 'src'))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MODEL_DIR  = "model_v22"
TOK_DIR    = "tokenizer_v22"
DATA_PATH  = "중요파일/data/raw/재난문자_레이블링결과_dedup_v7.xlsx"
MAX_LEN    = 96
BATCH_SIZE = 128

THRESHOLDS = [0.80, 0.85, 0.87, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            {k: v[idx] for k, v in self.enc.items()},
            self.labels[idx]
        )

# ── 로드 ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device); model.eval()

from dataset_v2 import load_and_split_v2
_, _, df = load_and_split_v2(DATA_PATH, seed=42)
texts  = df["text"].astype(str).tolist()
labels = df["label"].tolist()
print(f"테스트 데이터: {len(texts)}건")

dataset = TextDataset(texts, labels, tokenizer)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 추론 ──────────────────────────────────────────────────────────────
all_confs  = []
all_preds  = []
all_labels = []

with torch.no_grad():
    for batch_enc, batch_labels in tqdm(loader, desc="추론 중"):
        batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
        out   = model(**batch_enc)
        probs = F.softmax(out.logits, dim=-1).cpu()
        confs, preds = probs.max(dim=-1)
        all_confs.extend(confs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_labels.tolist())

all_confs  = np.array(all_confs)
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

correct = (all_preds == all_labels)
n_total = len(all_labels)
base_acc = correct.mean() * 100
print(f"\n기본 정확도 (전체): {base_acc:.2f}%\n")

# ── confidence 분포 ────────────────────────────────────────────────────
print("─" * 60)
print("Confidence 분포")
print("─" * 60)
for lo, hi in [(0.0, 0.80), (0.80, 0.85), (0.85, 0.90),
               (0.90, 0.92), (0.92, 0.94), (0.94, 0.95),
               (0.95, 0.97), (0.97, 0.99), (0.99, 1.01)]:
    mask = (all_confs >= lo) & (all_confs < hi)
    n = mask.sum()
    err = (~correct[mask]).sum() if n > 0 else 0
    print(f"  [{lo:.0%} ~ {hi:.0%})  {n:6d}건 ({n/n_total*100:5.1f}%)  오분류: {err:4d}건 ({err/n*100:.1f}%)" if n > 0 else f"  [{lo:.0%} ~ {hi:.0%})  {n:6d}건")
print()

# ── threshold별 트레이드오프 ───────────────────────────────────────────
print("─" * 90)
print(f"{'Threshold':>10}  {'v22처리':>7}  {'v22오류':>7}  {'v22정확도':>9}  {'LLM비율':>8}  {'LLM내오류':>9}  {'전체놓침':>8}")
print("─" * 90)

for thr in THRESHOLDS:
    v22_mask = all_confs >= thr
    llm_mask = ~v22_mask

    n_v22 = v22_mask.sum()
    n_llm = llm_mask.sum()

    v22_err = (~correct[v22_mask]).sum() if n_v22 > 0 else 0
    llm_err = (~correct[llm_mask]).sum() if n_llm > 0 else 0

    v22_acc  = correct[v22_mask].mean() * 100 if n_v22 > 0 else 0
    llm_rate = n_llm / n_total * 100
    llm_err_rate = llm_err / n_llm * 100 if n_llm > 0 else 0

    # v22가 처리하는 것 중 틀리는 건 = "놓치는 오류"
    missed_err = v22_err

    print(f"  {thr:.0%}      {n_v22:7d}  {v22_err:7d}  {v22_acc:8.3f}%  {llm_rate:7.2f}%  {llm_err_rate:8.2f}%  {missed_err:8d}")

print("─" * 90)
print()
print("* v22처리: confidence ≥ threshold인 샘플 (모델이 직접 분류)")
print("* v22오류: v22가 처리하는 샘플 중 오분류 수")
print("* LLM비율: LLM fallback으로 넘어가는 비율")
print("* LLM내오류: LLM으로 넘어가는 것 중 v22가 틀렸을 비율 (LLM이 잡아야 할 케이스)")
print("* 전체놓침: v22가 처리하면서 틀리는 수 (LLM으로 안 넘어가서 그냥 오분류)")

result_path = "실험/confidence_threshold_analysis.txt"
lines = []
lines.append(f"기본 정확도: {base_acc:.2f}%  (테스트 {n_total}건)\n")
lines.append(f"{'Threshold':>10}  {'v22처리':>7}  {'v22오류':>7}  {'v22정확도':>9}  {'LLM비율':>8}  {'LLM내오류':>9}  {'전체놓침':>8}")
lines.append("─" * 90)
for thr in THRESHOLDS:
    v22_mask = all_confs >= thr
    llm_mask = ~v22_mask
    n_v22 = v22_mask.sum()
    n_llm = llm_mask.sum()
    v22_err = (~correct[v22_mask]).sum() if n_v22 > 0 else 0
    llm_err = (~correct[llm_mask]).sum() if n_llm > 0 else 0
    v22_acc  = correct[v22_mask].mean() * 100 if n_v22 > 0 else 0
    llm_rate = n_llm / n_total * 100
    llm_err_rate = llm_err / n_llm * 100 if n_llm > 0 else 0
    lines.append(f"  {thr:.0%}      {n_v22:7d}  {v22_err:7d}  {v22_acc:8.3f}%  {llm_rate:7.2f}%  {llm_err_rate:8.2f}%  {v22_err:8d}")

with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n결과 저장: {result_path}")
