"""
마스킹 후 오분류 샘플 심층 분석
: 원본 정답 → 마스킹 후 오답인 샘플만 추출, 원인 분류
"""

import re
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from collections import Counter

from dataset_v2 import load_and_split_v2, DisasterDatasetAug
from model import load_model

# v5 검증과 동일한 원본 25개 키워드 (dataset_v2 변경 무관하게 고정)
_MASK_KEYWORDS_V5 = sorted([
    '즉시 대피', '대피명령', '대피 명령', '긴급대피', '긴급 대피', '신속히 대피',
    '지진 발생', '쓰나미', '민방공', '테러',
    '경보', '주의보', '특보', '예비특보',
    '대피', '발생', '화재', '산불', '홍수', '태풍', '침수', '범람',
    '해제', '종료', '완료',
], key=len, reverse=True)

_SPACES = __import__('re').compile(r'\s+')

def _mask_text(text: str) -> str:
    for kw in _MASK_KEYWORDS_V5:
        text = text.replace(kw, ' ')
    return _SPACES.sub(' ', text).strip()

_MASK_KEYWORDS = _MASK_KEYWORDS_V5

LABEL_NAMES = {0: '긴급아님', 1: '낮음', 2: '중간', 3: '높음', 4: '매우높음'}


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttids = batch['token_type_ids'].to(device)
            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
    return all_preds


def classify_error(orig_text, masked_text, true_lbl, pred_lbl):
    """오류 원인 분류"""
    removed = orig_text.replace(masked_text, '').strip()
    remaining_len = len(masked_text.replace(' ', ''))
    orig_len = len(orig_text.replace(' ', ''))
    removal_ratio = 1 - remaining_len / max(orig_len, 1)

    if removal_ratio > 0.5:
        return "AMBIGUOUS_EMPTY"   # 마스킹으로 텍스트 절반 이상 소실
    elif remaining_len < 10:
        return "AMBIGUOUS_TOO_SHORT"  # 남은 텍스트 너무 짧음
    else:
        return "MODEL_FAILURE"     # 텍스트 남아있으나 오분류


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='v5')
    parser.add_argument('--target_label', type=int, default=None, help='특정 레이블만 분석')
    parser.add_argument('--top_n', type=int, default=30, help='레이블당 출력 개수')
    args = parser.parse_args()

    model_dirs = {
        'v5': ('model_v5', 'tokenizer_v5'),
        'v6': ('model_v6', 'tokenizer_v6'),
    }
    model_dir, tok_dir = model_dirs[args.model]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Model: {args.model}\n")

    _, _, test_df = load_and_split_v2('중요파일/data/raw/재난문자_레이블링결과_dedup_v3.xlsx')
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    model = load_model(model_dir, num_labels=5).to(device)

    # 원본 예측
    orig_ds = DisasterDatasetAug(test_df, tokenizer, max_length=128, augment=False)
    orig_loader = DataLoader(orig_ds, batch_size=64, shuffle=False, num_workers=0)
    y_pred_orig = evaluate(model, orig_loader, device)

    # 마스킹 후 예측
    masked_df = test_df.copy()
    masked_df['text'] = masked_df['text'].apply(_mask_text)
    masked_ds = DisasterDatasetAug(masked_df, tokenizer, max_length=128, augment=False)
    masked_loader = DataLoader(masked_ds, batch_size=64, shuffle=False, num_workers=0)
    y_pred_masked = evaluate(model, masked_loader, device)

    y_true = test_df['label'].tolist()
    texts_orig = test_df['text'].tolist()
    texts_masked = masked_df['text'].tolist()

    # 마스킹 전 정답 → 마스킹 후 오답인 샘플만
    newly_wrong = [
        i for i in range(len(y_true))
        if y_pred_orig[i] == y_true[i] and y_pred_masked[i] != y_true[i]
    ]

    print(f"마스킹 전 정답 → 마스킹 후 오답: {len(newly_wrong)}건\n")

    # 레이블별 분류
    target_labels = [args.target_label] if args.target_label is not None else [2, 3, 4]

    for lbl in target_labels:
        lbl_errors = [i for i in newly_wrong if y_true[i] == lbl]
        print("=" * 70)
        print(f"Label {lbl} ({LABEL_NAMES[lbl]}) - {len(lbl_errors)}건")
        print("=" * 70)

        # 오분류 방향 집계
        directions = Counter(y_pred_masked[i] for i in lbl_errors)
        print(f"오분류 방향: {dict(sorted(directions.items()))}")

        # 오류 원인 분류
        causes = Counter()
        for i in lbl_errors:
            cause = classify_error(texts_orig[i], texts_masked[i], lbl, y_pred_masked[i])
            causes[cause] += 1
        print(f"원인 분류: {dict(causes)}")
        print()

        # 샘플 출력
        for rank, i in enumerate(lbl_errors[:args.top_n], 1):
            orig = texts_orig[i]
            masked = texts_masked[i]
            pred = y_pred_masked[i]
            cause = classify_error(orig, masked, lbl, pred)

            removed_kws = [kw for kw in _MASK_KEYWORDS if kw in orig]

            print(f"  [{rank}] {cause} | 정답:{lbl}({LABEL_NAMES[lbl]}) → 예측:{pred}({LABEL_NAMES[pred]})")
            print(f"  원본  : {orig[:100]}")
            print(f"  마스킹: {masked[:100]}")
            print(f"  제거됨: {removed_kws}")
            print()

    # 전체 원인 요약
    print("=" * 70)
    print("전체 원인 요약 (Labels 2/3/4)")
    all_target = [i for i in newly_wrong if y_true[i] in target_labels]
    all_causes = Counter(
        classify_error(texts_orig[i], texts_masked[i], y_true[i], y_pred_masked[i])
        for i in all_target
    )
    total = len(all_target)
    for cause, cnt in sorted(all_causes.items(), key=lambda x: -x[1]):
        print(f"  {cause}: {cnt}건 ({cnt/total*100:.1f}%)")
    print(f"  합계: {total}건")


if __name__ == '__main__':
    main()
