"""
문제 수정 평가
  1. 데이터 누수 수정: Train과 중복된 Test 샘플 제거 후 재평가
  2. 키워드 분석: 레이블 결정 키워드 유무별 성능 비교

사용법:
  python evaluate_clean.py --data 재난문자_레이블링결과.xlsx
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from dataset import load_and_split, DisasterDataset
from model import load_model

# 레이블 생성 규칙에서 사용한 결정 키워드 (전체)
DECISION_KEYWORDS = [
    # 긴급 키워드
    '즉시 대피', '대피명령', '대피 명령', '긴급대피', '긴급 대피', '신속히 대피',
    '지진 발생', '쓰나미', '민방공', '테러', '폭발 발생',
    '범람', '침수 발생',
    # 주의 키워드
    '경보', '주의보', '특보', '예비특보',
    '대피', '침수 우려',
    '화재 발생', '산불 발생', '산불 위험', '산불발생',
]

LABEL_NAMES = {0: '긴급아님', 1: '낮음', 2: '중간', 3: '높음', 4: '매우높음'}


def get_preds(df, tokenizer, model, device, max_length, batch_size):
    if len(df) == 0:
        return [], []
    ds = DisasterDataset(df, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, ncols=90, leave=False):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttids = batch['token_type_ids'].to(device)
            logits = model(
                input_ids=ids, attention_mask=mask, token_type_ids=ttids
            ).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch['label'].numpy())
    return all_labels, all_preds


def fmt_row(name, y_true, y_pred):
    if len(y_true) == 0:
        return f"  {name:<24} | n=0\n"
    acc = accuracy_score(y_true, y_pred) * 100
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    err = sum(t != p for t, p in zip(y_true, y_pred))
    return f"  {name:<24} | n={len(y_true):>6,}  Acc={acc:6.2f}%  MacroF1={f1:6.2f}%  오분류={err:>4}건\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       type=str, default='재난문자_레이블링결과.xlsx')
    parser.add_argument('--model_path', type=str, default='model/')
    parser.add_argument('--tok_path',   type=str, default='tokenizer/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 데이터 로드 (기존 train.py와 동일한 seed)
    print("데이터 로드 중...")
    train_df, _, test_df = load_and_split(args.data, seed=args.seed)
    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

    # ── 문제 1: 중복 제거
    train_texts = set(train_df['text'].tolist())
    test_df = test_df.copy()
    test_df['is_dup'] = test_df['text'].isin(train_texts)
    dup_count    = int(test_df['is_dup'].sum())
    clean_df     = test_df[~test_df['is_dup']].reset_index(drop=True)

    print(f"\n[중복 현황]")
    print(f"  Train-Test 중복 샘플: {dup_count:,}건 ({dup_count/len(test_df)*100:.1f}%)")
    print(f"  중복 제거 후: {len(clean_df):,}건")

    # ── 문제 2: 키워드 유무 분리 (중복 제거 후 기준)
    clean_df = clean_df.copy()
    clean_df['has_kw'] = clean_df['text'].apply(
        lambda t: any(kw in t for kw in DECISION_KEYWORDS)
    )
    kw_df   = clean_df[clean_df['has_kw']].reset_index(drop=True)
    nokw_df = clean_df[~clean_df['has_kw']].reset_index(drop=True)

    print(f"\n[키워드 현황 - 중복제거 후]")
    print(f"  키워드 있음: {len(kw_df):,}건 ({len(kw_df)/len(clean_df)*100:.1f}%)")
    print(f"  키워드 없음: {len(nokw_df):,}건 ({len(nokw_df)/len(clean_df)*100:.1f}%)")

    # ── 모델 로드
    print("\n모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)
    model     = load_model(args.model_path, num_labels=5).to(device)
    model.eval()

    # ── 평가
    sets = [
        ("원본 전체",          test_df),
        ("중복제거",           clean_df),
        ("중복제거+키워드有",   kw_df),
        ("중복제거+키워드無",   nokw_df),
    ]

    results = {}
    for name, df in sets:
        print(f"\n[{name}] 평가 중... ({len(df):,}건)")
        y_true, y_pred = get_preds(df, tokenizer, model, device,
                                   args.max_length, args.batch_size)
        results[name] = (y_true, y_pred)

    # ── 결과 출력 및 저장
    sep = "─" * 70
    lines = [
        "=== 수정 평가: 데이터 누수 + 키워드 분석 ===\n\n",
        f"[중복 현황]\n",
        f"  전체 Test:       {len(test_df):>6,}건\n",
        f"  Train 중복 제거: {dup_count:>6,}건 ({dup_count/len(test_df)*100:.1f}%)\n",
        f"  중복 제거 후:    {len(clean_df):>6,}건\n\n",
        f"[키워드 현황 - 중복제거 후]\n",
        f"  키워드 있음: {len(kw_df):>6,}건 ({len(kw_df)/len(clean_df)*100:.1f}%)\n",
        f"  키워드 없음: {len(nokw_df):>6,}건 ({len(nokw_df)/len(clean_df)*100:.1f}%)\n\n",
        sep + "\n",
        f"  {'구분':<24} | 샘플수   Accuracy  MacroF1   오분류\n",
        sep + "\n",
    ]
    for name, (y_true, y_pred) in results.items():
        lines.append(fmt_row(name, y_true, y_pred))
    lines.append(sep + "\n")

    # 키워드無 클래스별 상세
    nokw_true, nokw_pred = results["중복제거+키워드無"]
    if nokw_true:
        lines.append("\n[키워드 없음 — 클래스별 상세]\n")
        report = classification_report(
            nokw_true, nokw_pred,
            target_names=[LABEL_NAMES[i] for i in range(5)],
            digits=4, zero_division=0,
        )
        for line in report.splitlines():
            lines.append("  " + line + "\n")

    os.makedirs('results', exist_ok=True)
    out_path = 'results/evaluate_clean_report.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print("\n" + "".join(lines))
    print(f"저장: {out_path}")


if __name__ == '__main__':
    main()
