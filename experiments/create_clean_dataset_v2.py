"""
Level 2 중복 제거: 전처리 후 날짜·시간·숫자 정규화 기준 중복 제거

사용법:
  python create_clean_dataset_v2.py
"""

import argparse
import re
import pandas as pd
from dataset import preprocess_text

_DATE_FULL = re.compile(r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일')
_DATE_SHORT = re.compile(r'\d{1,2}월\s*\d{1,2}일')
_TIME = re.compile(r'\d{1,2}시\s*\d{0,2}분?|\d{2}:\d{2}')
_NUM = re.compile(r'\d+')


def normalize_for_dedup(text: str) -> str:
    text = _DATE_FULL.sub('DATE', text)
    text = _DATE_SHORT.sub('DATE', text)
    text = _TIME.sub('TIME', text)
    text = _NUM.sub('NUM', text)
    return re.sub(r'\s+', ' ', text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='중요파일/data/raw/재난문자_레이블링결과.xlsx')
    parser.add_argument('--out',  default='중요파일/data/raw/재난문자_레이블링결과_dedup_v2.xlsx')
    args = parser.parse_args()

    print(f"로드: {args.data}")
    df = pd.read_excel(args.data)

    before_invalid = len(df)
    df = df[df['label'] != -1].copy()
    df['label'] = df['label'].astype(int)
    removed_invalid = before_invalid - len(df)

    df['text'] = df['메시지내용'].fillna('').apply(preprocess_text)
    df['_norm'] = df['text'].apply(normalize_for_dedup)

    before_l1 = len(df)
    df_l1 = df.drop_duplicates(subset='text', keep='first')
    removed_l1 = before_l1 - len(df_l1)

    before_l2 = len(df_l1)
    df_l2 = df_l1.drop_duplicates(subset='_norm', keep='first').reset_index(drop=True)
    removed_l2 = before_l2 - len(df_l2)

    df_out = df_l2.drop(columns=['_norm'])

    print(f"\n원본 (label!=-1 제거 후): {before_l1:,}건")
    print(f"Level 1 (exact) 중복 제거: {removed_l1:,}건")
    print(f"Level 2 (날짜·숫자 정규화) 추가 제거: {removed_l2:,}건")
    print(f"최종: {len(df_out):,}건")
    if removed_invalid:
        print(f"  (label==-1 제거: {removed_invalid}건)")

    print("\n[클래스별 분포]")
    label_names = {0: '긴급아님', 1: '낮음', 2: '중간', 3: '높음', 4: '매우높음'}
    l1_counts   = df_l1['label'].value_counts().sort_index()
    l2_counts   = df_out['label'].value_counts().sort_index()
    print(f"  {'레벨':<10} {'L1후':>8} {'L2후':>8} {'추가제거':>8}")
    for lbl in sorted(l1_counts.index):
        a = l1_counts.get(lbl, 0)
        b = l2_counts.get(lbl, 0)
        print(f"  {lbl} {label_names[lbl]:<8} {a:>8,} {b:>8,} {a-b:>8,}")

    df_out.to_excel(args.out, index=False)
    print(f"\n저장: {args.out}")


if __name__ == '__main__':
    main()
