"""
복구된 test_v9n.xlsx로 v9n 모델 성능 검증
원본 test → 마스킹 test 순서로 평가
"""
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, r2_score

sys.path.insert(0, '../완성 모델/src')
from model import load_model
from dataset_v2 import DisasterDatasetAug, _mask_text

TEST_PATH  = '../중요파일/data/processed/test_v9n.xlsx'
MODEL_PATH = '../model_v9n'
TOK_PATH   = '../tokenizer_v9n'
BATCH_SIZE = 64
MAX_LEN    = 128

LABEL_NAMES = ['긴급아님(L0)', '낮음(L1)', '중간(L2)', '높음(L3)', '매우높음(L4)']


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttids = batch['token_type_ids'].to(device)
            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch['label'].numpy())
    return np.array(all_labels), np.array(all_preds)


def print_report(tag, labels, preds):
    N = len(labels)
    acc    = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average='macro', zero_division=0)
    r2_ord = r2_score(labels, preds)

    print(f"\n{'='*68}")
    print(f"  {tag}")
    print(f"{'='*68}")
    print(f"\n  {'레벨':<16} {'F1':>8} {'Recall':>8} {'R2(이진)':>10}")
    print(f"  {'-'*44}")
    for c in range(5):
        f1_c  = f1_score(labels, preds, labels=[c], average='macro', zero_division=0)
        rec_c = recall_score(labels, preds, labels=[c], average='macro', zero_division=0)
        r2_c  = r2_score((labels == c).astype(int), (preds == c).astype(int))
        print(f"  L{c} {LABEL_NAMES[c]:<14} {f1_c*100:>7.2f}%  {rec_c*100:>7.2f}%  {r2_c:>9.4f}")
    print(f"  {'-'*44}")
    print(f"  {'전체 (Macro)':<16} {f1_mac*100:>7.2f}%              {r2_ord:>9.4f}  <- R2 ordinal")
    print(f"\n  Accuracy: {acc*100:.4f}%   |   총 {N:,}건")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("테스트 데이터 로드 중...")
    test_df = pd.read_excel(TEST_PATH)
    test_df['text'] = test_df['text'].fillna('').astype(str)
    print(f"Test: {len(test_df):,}건")
    print(test_df['label'].value_counts().sort_index().to_string())

    tokenizer = AutoTokenizer.from_pretrained(TOK_PATH)
    model     = load_model(MODEL_PATH, num_labels=5).to(device)

    # 원본 test
    ds     = DisasterDatasetAug(test_df, tokenizer, MAX_LEN, augment=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    labels, preds = evaluate(model, loader, device)
    print_report("원본 테스트셋 (복구된 test_v9n.xlsx)", labels, preds)

    # confusion matrix 출력
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    print("\n  Confusion Matrix (rows=True, cols=Pred):")
    header = "       " + "".join(f"{'P'+str(c):>7}" for c in range(5))
    print(f"  {header}")
    for r in range(5):
        row = "  " + f"T{r}: " + "".join(f"{cm[r,c]:>7}" for c in range(5))
        print(row)

    # 마스킹 test
    masked_df = test_df.copy()
    masked_df['text'] = masked_df['text'].apply(_mask_text)
    ds_m     = DisasterDatasetAug(masked_df, tokenizer, MAX_LEN, augment=False)
    loader_m = DataLoader(ds_m, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    labels_m, preds_m = evaluate(model, loader_m, device)
    print_report("마스킹 테스트셋 (복구된 test_v9n.xlsx)", labels_m, preds_m)


if __name__ == '__main__':
    main()
