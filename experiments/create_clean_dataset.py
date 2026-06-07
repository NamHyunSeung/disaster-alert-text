"""
전처리 후 텍스트 기준 중복 제거 → 새 Excel 저장

사용법:
  python create_clean_dataset.py
  python create_clean_dataset.py --data 중요파일/data/raw/재난문자_레이블링결과.xlsx
                                  --out  중요파일/data/raw/재난문자_레이블링결과_dedup.xlsx
"""

import argparse
import re
import pandas as pd

_ORG_PATTERN = re.compile(r'\[[^\]]{1,20}\]')


def preprocess_text(text: str) -> str:
    text = str(text)
    text = _ORG_PATTERN.sub(' ', text)
    text = text.replace('\n', ' ')
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='중요파일/data/raw/재난문자_레이블링결과.xlsx')
    parser.add_argument('--out',  default='중요파일/data/raw/재난문자_레이블링결과_dedup.xlsx')
    args = parser.parse_args()

    print(f"로드: {args.data}")
    df = pd.read_excel(args.data)

    before_invalid = len(df)
    df = df[df['label'] != -1].copy()
    df['label'] = df['label'].astype(int)
    removed_invalid = before_invalid - len(df)

    df['text'] = df['메시지내용'].fillna('').apply(preprocess_text)

    before = len(df)
    df_dedup = df.drop_duplicates(subset='text', keep='first').reset_index(drop=True)
    removed = before - len(df_dedup)

    print(f"\n원본 (label!=-1 제거 후): {before:,}건")
    print(f"중복 제거: {removed:,}건 ({removed/before*100:.1f}%)")
    print(f"결과: {len(df_dedup):,}건")
    if removed_invalid:
        print(f"  (label==-1 제거: {removed_invalid}건)")

    print("\n[클래스별 분포]")
    label_names = {0: '긴급아님', 1: '낮음', 2: '중간', 3: '높음', 4: '매우높음'}
    orig_counts  = df['label'].value_counts().sort_index()
    dedup_counts = df_dedup['label'].value_counts().sort_index()
    print(f"  {'레벨':<10} {'원본':>8} {'중복제거후':>10} {'제거수':>8}")
    for lbl in sorted(orig_counts.index):
        o = orig_counts.get(lbl, 0)
        d = dedup_counts.get(lbl, 0)
        print(f"  {lbl} {label_names[lbl]:<8} {o:>8,} {d:>10,} {o-d:>8,}")

    # 저장 (text 컬럼 포함 — load_and_split_clean에서 재사용 가능)
    df_dedup.to_excel(args.out, index=False)
    print(f"\n저장: {args.out}")


if __name__ == '__main__':
    main()
