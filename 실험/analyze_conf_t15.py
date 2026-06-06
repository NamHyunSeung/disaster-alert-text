"""
T=1.5 기준 confidence 분포 분석
틀린 케이스 포착률 vs LLM 비율 trade-off 분석
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '완성 모델', 'src'))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset_v2 import load_and_split_v2

MODEL_DIR   = "model_v22"
TOK_DIR     = "tokenizer_v22"
DATA_PATH   = "중요파일/data/raw/재난문자_레이블링결과_dedup_v7.xlsx"
MAX_LEN     = 96
BATCH_SIZE  = 64
TEMPERATURE = 1.5


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.enc = tokenizer(texts, truncation=True, padding="max_length",
                             max_length=MAX_LEN, return_tensors="pt")

    def __len__(self):
        return len(self.enc['input_ids'])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    _, _, test_df = load_and_split_v2(DATA_PATH, seed=42)
    texts   = test_df["text"].astype(str).tolist()
    labels  = np.array(test_df["label"].tolist())
    n_total = len(texts)
    print(f"테스트 샘플: {n_total:,}건")

    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device); model.eval()

    dataset = TextDataset(texts, tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs_list = []
    print("\n배치 추론 중...")
    with torch.no_grad():
        for batch in tqdm(loader, ncols=90):
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            probs = F.softmax(out.logits / TEMPERATURE, dim=-1).cpu()
            all_probs_list.append(probs)

    all_probs = torch.cat(all_probs_list, dim=0)
    all_preds = all_probs.argmax(dim=-1).numpy()
    all_confs = all_probs.max(dim=-1).values.numpy()

    correct = (all_preds == labels)
    wrong   = ~correct
    n_wrong = wrong.sum()

    correct_confs = all_confs[correct]
    wrong_confs   = all_confs[wrong]

    print(f"\n[T={TEMPERATURE} 기준 전체 결과]")
    print(f"  정답: {correct.sum():,}건  |  오답: {n_wrong:,}건  |  Acc={correct.mean()*100:.2f}%")

    print(f"\n[오답({n_wrong}건)의 confidence 분포]")
    for p in [100, 95, 90, 85, 80, 75, 70, 65, 60, 50, 40, 30, 20]:
        n = (wrong_confs >= p / 100).sum()
        print(f"  >= {p:3d}%: {n:>4}건 ({n/n_wrong*100:5.1f}%)")

    print(f"\n[threshold별 trade-off]  (OOD 탐지 별도 적용 전 순수 confidence 기준)")
    print(f"  {'threshold':>10}  {'LLM건수':>8}  {'LLM%':>7}  {'오답포착':>8}  {'포착%':>7}  {'v22직접Acc':>11}")
    thresholds = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
    for thr in thresholds:
        llm_mask = all_confs < thr
        v22_mask = ~llm_mask
        n_llm    = llm_mask.sum()
        pct_llm  = n_llm / n_total * 100
        wrong_captured = (wrong_confs < thr).sum()
        capture_rate   = wrong_captured / n_wrong * 100 if n_wrong > 0 else 0.0
        v22_acc = correct[v22_mask].mean() * 100 if v22_mask.sum() > 0 else 100.0
        print(f"  {thr:>10.0%}  {n_llm:>8,}  {pct_llm:>6.1f}%  {wrong_captured:>8}  {capture_rate:>6.1f}%  {v22_acc:>10.2f}%")

    # 결과 저장
    out_lines = [
        f"[T={TEMPERATURE} confidence 분포 분석]",
        f"테스트 샘플: {n_total:,}건  |  오답: {n_wrong:,}건  |  Acc={correct.mean()*100:.2f}%",
        "",
        "threshold  | LLM건수  | LLM%   | 오답포착 | 포착%  | v22직접Acc",
    ]
    for thr in thresholds:
        llm_mask = all_confs < thr
        v22_mask = ~llm_mask
        n_llm    = llm_mask.sum()
        pct_llm  = n_llm / n_total * 100
        wrong_captured = (wrong_confs < thr).sum()
        capture_rate   = wrong_captured / n_wrong * 100 if n_wrong > 0 else 0.0
        v22_acc = correct[v22_mask].mean() * 100 if v22_mask.sum() > 0 else 100.0
        out_lines.append(
            f"  {thr:.0%}      | {n_llm:>8,} | {pct_llm:>5.1f}% | {wrong_captured:>8} | {capture_rate:>5.1f}% | {v22_acc:.2f}%"
        )

    result_path = "실험/conf_t15_analysis.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"\n결과 저장: {result_path}")


if __name__ == '__main__':
    main()
