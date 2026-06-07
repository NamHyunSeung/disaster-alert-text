"""
v9n 모델 R2 스코어 + Accuracy 계산 (원본 + 마스킹)
원본 평가와 동일하게: 재난문자_레이블링결과_dedup_v3.xlsx + dataset_v2
"""
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import r2_score, accuracy_score, f1_score, recall_score

sys.path.insert(0, '../완성 모델/src')
from model import load_model
from dataset import preprocess_text
from dataset_v2 import load_and_split_v2, DisasterDatasetAug, _mask_text, _MASK_KEYWORDS

DATA_PATH  = '../중요파일/data/raw/재난문자_레이블링결과_dedup_v3.xlsx'
MODEL_PATH = '../model_v9n'
TOK_PATH   = '../tokenizer_v9n'
BATCH_SIZE = 64
MAX_LEN    = 128
SEED       = 42

LABEL_NAMES = {0: '긴급아님', 1: '낮음', 2: '중간', 3: '높음', 4: '매우높음'}


def evaluate_loader(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttids = batch['token_type_ids'].to(device)
            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['label'].numpy())
    return np.array(all_labels), np.array(all_preds)


def print_report(tag, labels, preds):
    print(f"\n{'='*62}")
    print(f"{tag}")
    print(f"{'='*62}")

    acc    = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average='macro')
    r2_ord = r2_score(labels, preds)

    print(f"\n  {'레벨':<4} {'클래스':<8} {'F1':>8} {'Recall':>8} {'R2(이진)':>10}")
    print(f"  {'-'*48}")
    for c in range(5):
        idx = labels == c
        f1_c  = f1_score(labels[idx], preds[idx], labels=[c], average=None, zero_division=0)[0] if idx.sum() > 0 else 0.0
        # 실제 F1/Recall: 전체 기준 해당 클래스만
        f1_c  = f1_score(labels, preds, labels=[c], average='macro', zero_division=0)
        rec_c = recall_score(labels, preds, labels=[c], average='macro', zero_division=0)
        r2_c  = r2_score((labels == c).astype(int), (preds == c).astype(int))
        print(f"  L{c}   {LABEL_NAMES[c]:<8} {f1_c*100:>7.2f}%  {rec_c*100:>7.2f}%  {r2_c:>9.4f}")

    print(f"  {'-'*48}")
    print(f"  {'전체':<12} {f1_mac*100:>7.2f}%          {r2_ord:>9.4f}  (R2 ordinal)")
    print(f"\n  Accuracy: {acc*100:.4f}%")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("데이터 로드 중...")
    _, _, test_df = load_and_split_v2(DATA_PATH, seed=SEED)
    print(f"Test: {len(test_df):,}건")
    print(test_df['label'].value_counts().sort_index().to_string())

    tokenizer = AutoTokenizer.from_pretrained(TOK_PATH)
    model     = load_model(MODEL_PATH, num_labels=5).to(device)

    # ── 원본 ──
    ds     = DisasterDatasetAug(test_df, tokenizer, MAX_LEN, augment=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    labels, preds = evaluate_loader(model, loader, device)
    print_report("원본 테스트셋", labels, preds)

    # ── 마스킹 ──
    masked_df = test_df.copy()
    masked_df['text'] = masked_df['text'].apply(_mask_text)
    ds_m     = DisasterDatasetAug(masked_df, tokenizer, MAX_LEN, augment=False)
    loader_m = DataLoader(ds_m, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    labels_m, preds_m = evaluate_loader(model, loader_m, device)
    print_report("마스킹 테스트셋", labels_m, preds_m)


if __name__ == '__main__':
    main()
