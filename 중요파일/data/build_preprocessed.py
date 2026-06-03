"""
재난문자 학습용 전처리 완료 데이터셋 빌드 스크립트

수행 작업:
  1. 레이블링 결과 xlsx 로드 (label != -1 필터)
  2. preprocess_text: 기관명 [..] 제거, 개행 → 공백
  3. 70/15/15 stratified split (seed=42)
  4a. [mode=clean] 증강 없음 - v9n(완성 모델) 학습 데이터와 동일
  4b. [mode=masked_ft] masked_ft 증강 사전 계산:
       train: 완전마스킹 1x + L2/3/4 부분마스킹(60%) 2x
       val  : 원본 + 완전마스킹 각 1벌
       test : 원본 + 완전마스킹 각 1벌
  5. parquet 저장

출력 컬럼 (clean):
  text      : preprocess_text 적용 텍스트
  label     : 0~4
  split     : train / val / test

출력 컬럼 (masked_ft):
  text      : 전처리(+마스킹) 완료 텍스트
  label     : 0~4
  split     : train / val / test
  aug_type  : full_mask / partial_mask_1 / partial_mask_2 / original / masked

사용법:
  python build_preprocessed.py --mode clean
  python build_preprocessed.py --mode clean --out processed/disaster_v4_clean.parquet
  python build_preprocessed.py --mode masked_ft --out processed/disaster_v4_masked_ft.parquet
"""

import os
import re
import sys
import random
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# preprocess_text (dataset.py 동일 로직)
# ─────────────────────────────────────────────
_ORG_PATTERN = re.compile(r'\[[^\]]{1,20}\]')

def preprocess_text(text: str) -> str:
    text = str(text)
    text = _ORG_PATTERN.sub(' ', text)
    text = text.replace('\n', ' ')
    return text.strip()


# ─────────────────────────────────────────────
# 마스킹 함수 (dataset_v2.py 동일 로직)
# ─────────────────────────────────────────────
_MASK_KEYWORDS = sorted([
    '즉시 대피', '대피명령', '대피 명령', '긴급대피', '긴급 대피', '신속히 대피',
    '지진 발생', '쓰나미', '민방공', '테러',
    '경보', '주의보', '특보', '예비특보',
    '대피', '발생', '화재', '산불', '홍수', '태풍', '침수', '범람',
    '해제', '종료', '완료',
], key=len, reverse=True)

_SPACES = re.compile(r'\s+')


def _mask_text(text: str) -> str:
    for kw in _MASK_KEYWORDS:
        text = text.replace(kw, ' ')
    return _SPACES.sub(' ', text).strip()


def _mask_text_partial(text: str, mask_ratio: float = 0.6) -> str:
    present = [kw for kw in _MASK_KEYWORDS if kw in text]
    if not present:
        return text
    n_mask = max(1, round(len(present) * mask_ratio))
    to_mask = sorted(random.sample(present, min(n_mask, len(present))), key=len, reverse=True)
    for kw in to_mask:
        text = text.replace(kw, ' ')
    return _SPACES.sub(' ', text).strip()


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='clean',
                        choices=['clean', 'masked_ft'],
                        help='clean: 증강 없음(v9n 동일), masked_ft: 마스킹 증강 포함')
    parser.add_argument('--data', type=str,
                        default='raw/재난문자_레이블링결과_dedup_v4.xlsx',
                        help='입력 xlsx 경로 (중요파일/data/ 기준 상대경로 or 절대경로)')
    parser.add_argument('--out', type=str, default=None,
                        help='출력 parquet 경로 (미지정 시 mode에 따라 자동 설정)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.out is None:
        args.out = ('processed/disaster_v4_clean.parquet' if args.mode == 'clean'
                    else 'processed/disaster_v4_masked_ft.parquet')

    # 스크립트 위치 기준 상대경로 해석
    base = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data if os.path.isabs(args.data) else os.path.join(base, args.data)
    out_path  = args.out  if os.path.isabs(args.out)  else os.path.join(base, args.out)

    random.seed(args.seed)

    # ── 1. 로드 & 전처리 ──────────────────────
    print(f"데이터 로드: {data_path}")
    df = pd.read_excel(data_path)
    df = df[df['label'] != -1].copy()
    df['label'] = df['label'].astype(int)
    df['text']  = df['메시지내용'].fillna('').apply(preprocess_text)
    print(f"  유효 샘플: {len(df):,}건")

    label_names = {0: '긴급아님', 1: '낮음', 2: '중간', 3: '높음', 4: '매우높음'}
    for lbl, cnt in df['label'].value_counts().sort_index().items():
        print(f"  L{lbl} ({label_names[lbl]}): {cnt:,}건 ({cnt/len(df)*100:.1f}%)")

    # ── 2. Train / Val / Test 분할 ───────────
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['label'], random_state=args.seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label'], random_state=args.seed
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f"\nTrain {len(train_df):,} / Val {len(val_df):,} / Test {len(test_df):,}")

    # ── 3. 증강 계산 ─────────────────────────
    records = []

    if args.mode == 'clean':
        # 증강 없음 - v9n(완성 모델) 학습 데이터와 동일
        print("\n[Clean mode] 증강 없이 원본 텍스트 저장 (v9n 동일)")
        for _, row in train_df.iterrows():
            records.append({'text': row['text'], 'label': row['label'], 'split': 'train'})
        for _, row in val_df.iterrows():
            records.append({'text': row['text'], 'label': row['label'], 'split': 'val'})
        for _, row in test_df.iterrows():
            records.append({'text': row['text'], 'label': row['label'], 'split': 'test'})

    else:
        # masked_ft 증강
        hard_labels = {2, 3, 4}

        # Train: 완전 마스킹 1x (원본 제거, 전체 레이블)
        print("\n[Train] 완전 마스킹 적용 중...")
        for _, row in train_df.iterrows():
            records.append({
                'text':     _mask_text(row['text']),
                'label':    row['label'],
                'split':    'train',
                'aug_type': 'full_mask',
            })

        # Train: L2/3/4 부분 마스킹(60%) 2x
        hard_train = train_df[train_df['label'].isin(hard_labels)].reset_index(drop=True)
        for rep in range(2):
            random.seed(args.seed + rep + 1)
            print(f"[Train] 부분 마스킹 {rep+1}/2 - L2/3/4 {len(hard_train):,}건...")
            for _, row in hard_train.iterrows():
                records.append({
                    'text':     _mask_text_partial(row['text'], mask_ratio=0.6),
                    'label':    row['label'],
                    'split':    'train',
                    'aug_type': f'partial_mask_{rep+1}',
                })

        # Val: 원본 + 완전 마스킹
        print("[Val] 원본 + 완전 마스킹...")
        for _, row in val_df.iterrows():
            records.append({'text': row['text'],             'label': row['label'], 'split': 'val', 'aug_type': 'original'})
            records.append({'text': _mask_text(row['text']), 'label': row['label'], 'split': 'val', 'aug_type': 'masked'})

        # Test: 원본 + 완전 마스킹
        print("[Test] 원본 + 완전 마스킹...")
        for _, row in test_df.iterrows():
            records.append({'text': row['text'],             'label': row['label'], 'split': 'test', 'aug_type': 'original'})
            records.append({'text': _mask_text(row['text']), 'label': row['label'], 'split': 'test', 'aug_type': 'masked'})

    out_df = pd.DataFrame(records)

    # ── 4. 저장 ──────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_parquet(out_path, index=False, engine='pyarrow')
    size_mb = os.path.getsize(out_path) / 1024 / 1024

    print(f"\n저장 완료: {out_path}  ({size_mb:.1f} MB)")
    print(f"총 {len(out_df):,}건\n")

    for split in ['train', 'val', 'test']:
        sub = out_df[out_df['split'] == split]
        lbl_dist = sub['label'].value_counts().sort_index()
        dist_str = '  '.join(f"L{l}={v:,}" for l, v in lbl_dist.items())
        print(f"  {split} ({len(sub):,}건)  |  {dist_str}")
        if 'aug_type' in out_df.columns:
            for aug in sorted(sub['aug_type'].unique()):
                cnt = sub[sub['aug_type'] == aug]
                print(f"    {aug:16s}: {len(cnt):,}건")


if __name__ == '__main__':
    main()
