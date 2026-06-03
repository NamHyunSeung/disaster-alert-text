"""
v14 masked test 평가 스크립트
test 세트에 마스킹 적용 후 MacroF1 측정
실행: 완성 모델/src/ 에서 python masked_eval_v14.py
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

MODEL_DIR = '../../model_v14'
TOK_DIR   = 'tokenizer'
DATA_PATH = '../../중요파일/data/raw/재난문자_레이블링결과_dedup_v6.xlsx'
OUT_PATH  = '../../results/masked_eval_v14.txt'
BATCH     = 128
MAX_LEN   = 128

LABELS = ['긴급아님', '낮음', '중간', '높음', '매우높음']

lines = []
def log(msg=''):
    print(msg, flush=True)
    lines.append(str(msg))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    log(f"Model:  {MODEL_DIR}")
    log(f"Data:   {DATA_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    model = load_model(MODEL_DIR, num_labels=5).to(device)

    log("\n데이터 로드 중...")
    _, _, test_df = load_and_split_v2(DATA_PATH)
    log(f"Test: {len(test_df):,}건")

    test_masked = test_df.copy()
    test_masked['text'] = test_masked['text'].apply(_mask_text)

    ds = DisasterDatasetAug(test_masked, tokenizer, MAX_LEN, augment=False)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttids = batch['token_type_ids'].to(device)
            lbls  = batch['label']
            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())
            if (i+1) % 50 == 0:
                log(f"  배치 {i+1}/{len(loader)}")

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_f1   = f1_score(y_true, y_pred, labels=[0,1,2,3,4], average=None, zero_division=0)
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3,4], zero_division=0)

    log(f"\n[Masked Test] Macro F1: {macro_f1*100:.2f}%")
    for i, name in enumerate(LABELS):
        log(f"  L{i} ({name}): P={p[i]*100:.1f}% / R={r[i]*100:.1f}% / F1={f[i]*100:.2f}%")

    if macro_f1 >= 0.98:
        log("\n[SUCCESS] Masked Test MacroF1 98% 달성!")
    else:
        log(f"\n[FAIL] 98% 미달 ({macro_f1*100:.2f}%)")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(lines))
    log(f"\n결과 저장: {OUT_PATH}")


if __name__ == '__main__':
    main()
