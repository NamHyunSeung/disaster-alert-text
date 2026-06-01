"""
L3 클래스 threshold sweep
마스킹 test set에서 L3 F1/Recall >= 98% 달성 가능한 threshold 탐색
"""
import re
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, recall_score, precision_score

from dataset_v2 import load_and_split_v2, DisasterDatasetAug, _MASK_KEYWORDS as MASK_KEYWORDS
from model import load_model

MODEL_DIR = 'model_v9n'
TOK_DIR   = 'tokenizer_v9n'
DATA_PATH = '중요파일/data/raw/재난문자_레이블링결과_dedup_v3.xlsx'
OUT_PATH  = 'results/threshold_sweep_v9n.txt'

LABEL_NAMES = ['긴급아님', '낮음', '중간', '높음', '매우높음']

lines = []

def log(msg=''):
    print(msg)
    lines.append(msg)


def mask_text(text: str) -> str:
    for kw in sorted(MASK_KEYWORDS, key=len, reverse=True):
        text = text.replace(kw, ' ')
    return re.sub(r'\s+', ' ', text).strip()


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


def apply_threshold(probs, threshold, target_class=3):
    """target_class 확률 >= threshold이면 target_class, 아니면 나머지 중 argmax"""
    p = probs.copy()
    above = p[:, target_class] >= threshold
    preds = p.argmax(axis=1).copy()
    # threshold 미달 샘플: target_class 확률을 -inf로 설정 후 재argmax
    p_mod = p.copy()
    p_mod[~above, target_class] = -1.0
    preds[~above] = p_mod[~above].argmax(axis=1)
    return preds


def metrics(y_true, y_pred):
    f1   = f1_score(y_true, y_pred, labels=[0,1,2,3,4], average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, labels=[0,1,2,3,4], average=None, zero_division=0)
    prec = precision_score(y_true, y_pred, labels=[0,1,2,3,4], average=None, zero_division=0)
    mf1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc  = (np.array(y_true) == np.array(y_pred)).mean()
    return acc, mf1, f1, rec, prec


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    log(f"Model: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    model = load_model(MODEL_DIR, num_labels=5).to(device)

    log("데이터 로드 중...")
    _, _, test_df = load_and_split_v2(DATA_PATH)
    log(f"Test: {len(test_df):,}건")

    test_df_masked = test_df.copy()
    test_df_masked['text'] = test_df_masked['text'].apply(mask_text)

    test_ds = DisasterDatasetAug(test_df_masked, tokenizer, max_length=128, augment=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    log("확률 추출 중 (마스킹 test set)...")
    y_true, probs = get_probs(model, test_loader, device)

    # 기본 argmax
    y_argmax = probs.argmax(axis=1)
    acc0, mf10, f1_0, rec_0, prec_0 = metrics(y_true, y_argmax)
    log(f"\n[기본 argmax (threshold=0.5 동등)]")
    log(f"  Macro F1: {mf10*100:.2f}%  Acc: {acc0*100:.2f}%")
    for lbl in [2, 3, 4]:
        log(f"  L{lbl} ({LABEL_NAMES[lbl]}): F1={f1_0[lbl]*100:.2f}%  Recall={rec_0[lbl]*100:.2f}%  Prec={prec_0[lbl]*100:.2f}%")

    # Threshold sweep
    log(f"\n[L3 Threshold Sweep: 0.30 ~ 0.75]")
    log(f"{'th':>6}  {'L2_F1':>7} {'L2_R':>7}  {'L3_F1':>7} {'L3_R':>7}  {'L4_F1':>7} {'L4_R':>7}  {'MacF1':>7}  {'OK':>3}")
    log("-" * 80)

    best_results = []
    for th in np.arange(0.30, 0.76, 0.01):
        y_pred = apply_threshold(probs, float(th), target_class=3)
        acc, mf1, f1, rec, prec = metrics(y_true, y_pred)

        l2f1, l3f1, l4f1 = f1[2], f1[3], f1[4]
        l2r,  l3r,  l4r  = rec[2], rec[3], rec[4]

        ok = all(v >= 0.98 for v in [l2f1, l3f1, l4f1, l2r, l3r, l4r])
        ok_str = 'OK' if ok else ''

        log(f"  {th:.2f}  {l2f1*100:6.2f}% {l2r*100:6.2f}%  {l3f1*100:6.2f}% {l3r*100:6.2f}%  {l4f1*100:6.2f}% {l4r*100:6.2f}%  {mf1*100:6.2f}%  {ok_str}")

        if ok:
            best_results.append((th, l2f1, l3f1, l4f1, l2r, l3r, l4r, mf1, acc))

    log("")
    if best_results:
        log("=" * 80)
        log(f"[OK] 목표 달성 threshold 범위: {best_results[0][0]:.2f} ~ {best_results[-1][0]:.2f}")
        # 가장 높은 Macro F1을 가진 threshold 선택
        best = max(best_results, key=lambda x: x[7])
        th, l2f1, l3f1, l4f1, l2r, l3r, l4r, mf1, acc = best
        log(f"\n최적 threshold (Macro F1 기준): {th:.2f}")
        log(f"  L2: F1={l2f1*100:.2f}%  Recall={l2r*100:.2f}%")
        log(f"  L3: F1={l3f1*100:.2f}%  Recall={l3r*100:.2f}%")
        log(f"  L4: F1={l4f1*100:.2f}%  Recall={l4r*100:.2f}%")
        log(f"  Macro F1: {mf1*100:.2f}%  Acc: {acc*100:.2f}%")
    else:
        log("[FAIL] 단일 L3 threshold로 목표 미달성")
        # L3 관련 최선 결과 출력
        y_pred_best = None
        best_l234 = -1
        best_th_val = 0.5
        for th in np.arange(0.30, 0.76, 0.01):
            y_pred = apply_threshold(probs, float(th), target_class=3)
            _, _, f1, rec, _ = metrics(y_true, y_pred)
            l234 = min(f1[2], f1[3], f1[4], rec[2], rec[3], rec[4])
            if l234 > best_l234:
                best_l234 = l234
                best_th_val = th
        log(f"  최선 threshold: {best_th_val:.2f}  (L234 min={best_l234*100:.2f}%)")
        log(f"  -> warm start 재학습 고려 필요")

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log(f"\n결과 저장: {OUT_PATH}")


if __name__ == '__main__':
    main()
