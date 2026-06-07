"""
v13 threshold sweep - masked val MacroF1 최적화
좌표 하강법으로 클래스별 가중치를 탐색해 최적 threshold 조합 찾기
실행: 완성 모델/src/ 에서 python threshold_sweep_v13.py
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

from dataset_v2 import load_and_split_v2, DisasterDatasetAug, _mask_text
from model import load_model

MODEL_DIR = '../../model_v13'
TOK_DIR   = 'tokenizer'
DATA_PATH = '../../중요파일/data/raw/재난문자_레이블링결과_dedup_v2.xlsx'
OUT_PATH  = '../../results/threshold_sweep_v13.txt'
BATCH     = 128
MAX_LEN   = 128

lines = []

def log(msg=''):
    print(msg, flush=True)
    lines.append(str(msg))


def get_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttids = batch['token_type_ids'].to(device)
            lbls  = batch['label']
            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(lbls.numpy())
    return np.array(all_labels), np.array(all_probs)


def masked_macro_f1(y_true, probs, weights):
    # 가중치 적용 후 argmax
    adj = probs * weights[np.newaxis, :]
    y_pred = adj.argmax(axis=1)
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def coordinate_descent(y_true, probs, rounds=3):
    # 탐색 가중치 격자: 로그 스케일 7점
    grid = np.array([0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.5, 4.0])
    best_w = np.ones(5)
    best_f1 = masked_macro_f1(y_true, probs, best_w)

    for _ in range(rounds):
        improved = False
        for c in range(5):
            for w in grid:
                trial = best_w.copy()
                trial[c] = w
                f1 = masked_macro_f1(y_true, probs, trial)
                if f1 > best_f1 + 1e-6:
                    best_f1 = f1
                    best_w = trial
                    improved = True
        if not improved:
            break

    return best_w, best_f1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    log(f"Model:  {MODEL_DIR}")
    log(f"Data:   {DATA_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    model = load_model(MODEL_DIR, num_labels=5).to(device)

    log("\n데이터 로드 중...")
    _, val_df, _ = load_and_split_v2(DATA_PATH)
    log(f"Val: {len(val_df):,}건")

    val_masked = val_df.copy()
    val_masked['text'] = val_masked['text'].apply(_mask_text)

    ds = DisasterDatasetAug(val_masked, tokenizer, MAX_LEN, augment=False)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    log("추론 중 (masked val)...")
    y_true, probs = get_probs(model, loader, device)

    # 기본 argmax
    base_f1 = masked_macro_f1(y_true, probs, np.ones(5))
    base_pred = probs.argmax(axis=1)
    base_per = f1_score(y_true, base_pred, labels=[0,1,2,3,4], average=None, zero_division=0)
    log(f"\n[기본 argmax] Masked MacroF1: {base_f1*100:.2f}%")
    LABELS = ['긴급아님', '낮음', '중간', '높음', '매우높음']
    for i, (name, f) in enumerate(zip(LABELS, base_per)):
        log(f"  L{i} ({name}): F1={f*100:.2f}%")

    # 좌표 하강법 최적화
    log("\n[좌표 하강법] 클래스별 가중치 최적화 중 (3 rounds)...")
    best_w, best_f1 = coordinate_descent(y_true, probs, rounds=3)

    best_pred = (probs * best_w[np.newaxis, :]).argmax(axis=1)
    best_per  = f1_score(y_true, best_pred, labels=[0,1,2,3,4], average=None, zero_division=0)

    log(f"\n[최적 가중치] Masked MacroF1: {best_f1*100:.2f}%")
    log(f"  가중치: {['%.2f'%w for w in best_w]}")
    for i, (name, f) in enumerate(zip(LABELS, best_per)):
        log(f"  L{i} ({name}): F1={f*100:.2f}%")

    delta = (best_f1 - base_f1) * 100
    log(f"\n개선폭: {delta:+.2f}%p  ({base_f1*100:.2f}% → {best_f1*100:.2f}%)")

    if best_f1 >= 0.98:
        log("\n[SUCCESS] Masked MacroF1 98% 달성!")
        result = "SUCCESS"
    else:
        log(f"\n[FAIL] 98% 미달 ({best_f1*100:.2f}%) → v14 학습 필요")
        result = "FAIL"

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log(f"\n결과 저장: {OUT_PATH}")
    log(f"RESULT={result}")

    return result, best_f1, best_w


if __name__ == '__main__':
    result, f1, w = main()
    sys.exit(0 if result == 'SUCCESS' else 1)
