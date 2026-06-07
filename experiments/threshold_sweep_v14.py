"""
v14 threshold tuning (coordinate descent on logit biases)
masked test 기준 MacroF1 최대화

실행: 프로젝트 루트에서
  python 실험/threshold_sweep_v14.py
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '완성 모델', 'src'))
from dataset_v2 import load_and_split_v2, DisasterDatasetAug, _mask_text
from model import load_model

MODEL_DIR = 'model_v14'
TOK_DIR   = os.path.join('완성 모델', 'src', 'tokenizer')
DATA_PATH = os.path.join('중요파일', 'data', 'raw', '재난문자_레이블링결과_dedup_v6.xlsx')
OUT_PATH  = os.path.join('results', 'threshold_sweep_v14.txt')
BATCH     = 128
MAX_LEN   = 128
LABELS    = ['L0(긴급아님)', 'L1(낮음)', 'L2(중간)', 'L3(높음)', 'L4(매우높음)']

lines = []
def log(msg=''):
    print(msg, flush=True)
    lines.append(str(msg))


def predict_with_bias(logits_np, bias):
    adjusted = logits_np + bias
    return adjusted.argmax(axis=1)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def coordinate_descent(logits_np, y_true, sweep_range=np.arange(-3.0, 3.1, 0.1), max_iter=10):
    bias = np.zeros(5)
    best_f1 = macro_f1(y_true, predict_with_bias(logits_np, bias))

    for iteration in range(max_iter):
        improved = False
        for cls in range(5):
            best_b = bias[cls]
            for b in sweep_range:
                candidate = bias.copy()
                candidate[cls] = b
                f = macro_f1(y_true, predict_with_bias(logits_np, candidate))
                if f > best_f1 + 1e-6:
                    best_f1 = f
                    best_b = b
                    improved = True
            bias[cls] = best_b
        if not improved:
            log(f"  수렴 (iteration {iteration+1})")
            break

    return bias, best_f1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    log(f"Model:  {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    model = load_model(MODEL_DIR, num_labels=5).to(device)

    log("\n데이터 로드 중...")
    _, _, test_df = load_and_split_v2(DATA_PATH)
    log(f"Test: {len(test_df):,}건")

    test_masked = test_df.copy()
    test_masked['text'] = test_masked['text'].apply(_mask_text)

    ds = DisasterDatasetAug(test_masked, tokenizer, MAX_LEN, augment=False)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    log("추론 중...")
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttids = batch['token_type_ids'].to(device)
            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(batch['label'].numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    y_true    = np.array(all_labels)

    # 기본 성능 (bias=0)
    y_pred_base = predict_with_bias(logits_np, np.zeros(5))
    base_f1 = macro_f1(y_true, y_pred_base)
    log(f"\n[기본] Masked MacroF1: {base_f1*100:.2f}%")
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred_base, labels=[0,1,2,3,4], zero_division=0)
    for i, name in enumerate(LABELS):
        log(f"  {name}: P={p[i]*100:.1f}% R={r[i]*100:.1f}% F1={f[i]*100:.2f}%")

    # Coordinate descent
    log("\n[Threshold Tuning] coordinate descent 실행 중...")
    best_bias, best_f1 = coordinate_descent(logits_np, y_true)

    y_pred_tuned = predict_with_bias(logits_np, best_bias)
    p2, r2, f2, _ = precision_recall_fscore_support(y_true, y_pred_tuned, labels=[0,1,2,3,4], zero_division=0)

    log(f"\n[튜닝 후] Masked MacroF1: {best_f1*100:.2f}%  (개선: +{(best_f1-base_f1)*100:.2f}%p)")
    log(f"최적 bias: {[round(b,2) for b in best_bias]}")
    for i, name in enumerate(LABELS):
        log(f"  {name}: P={p2[i]*100:.1f}% R={r2[i]*100:.1f}% F1={f2[i]*100:.2f}%")

    if best_f1 >= 0.98:
        log("\n[SUCCESS] Masked MacroF1 98% 달성!")
    else:
        log(f"\n[FAIL] 98% 미달 ({best_f1*100:.2f}%) → v15 학습 필요")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(lines))
    log(f"\n결과 저장: {OUT_PATH}")


if __name__ == '__main__':
    main()
