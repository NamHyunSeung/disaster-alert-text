"""
dedup_v7.xlsx 데이터셋 검증 (Step 4)
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '완성 모델', 'src'))

import pandas as pd
import numpy as np
from dataset_v2 import load_and_split_v2

DATA_V7 = os.path.join(ROOT, '중요파일', 'data', 'raw', '재난문자_레이블링결과_dedup_v7.xlsx')
COVID_KW = ['코로나', '확진', '백신', '접종', '격리', '방역']

LABEL_NAMES = {0: 'L0', 1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4'}

def has_covid_kw(text):
    return any(kw in str(text) for kw in COVID_KW)

print("=" * 60)
print("[ 1. 원본 파일 로드 ]")
df = pd.read_excel(DATA_V7)
print(f"총 행 수: {len(df):,}")
print(f"컬럼: {list(df.columns)}")
print(f"is_synthetic 값 분포: {df['is_synthetic'].value_counts().to_dict()}")

print()
print("[ 2. 전체 레이블 분포 ]")
for lbl in range(5):
    cnt = (df['label'] == lbl).sum()
    syn = ((df['label'] == lbl) & (df['is_synthetic'] == 1)).sum()
    print(f"  {LABEL_NAMES[lbl]}: {cnt:7,}건  (합성 {syn}건, {cnt/len(df)*100:.1f}%)")

print()
print("[ 3. 합성 데이터 검증 ]")
syn_df = df[df['is_synthetic'] == 1]
print(f"합성 행 총 수: {len(syn_df)}")
covid_in_syn = syn_df[syn_df['메시지내용'].apply(has_covid_kw)]
if len(covid_in_syn) == 0:
    print("  [OK] COVID 특이어 포함 합성 메시지 없음")
else:
    print(f"  [경고] COVID 특이어 포함 합성 메시지 {len(covid_in_syn)}건:")
    for _, row in covid_in_syn.iterrows():
        print(f"    L{int(row['label'])}: {row['메시지내용']}")

print()
print("[ 4. 합성 메시지 샘플 확인 ]")
for lbl in [1, 2]:
    subset = syn_df[syn_df['label'] == lbl]
    print(f"  -- L{lbl} 합성 ({len(subset)}건) 샘플 5개 --")
    for _, row in subset.head(5).iterrows():
        print(f"    {row['메시지내용']}")
    print()

print("[ 5. 70/15/15 분할 후 분포 ]")
train_df, val_df, test_df = load_and_split_v2(DATA_V7)
print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
print()
print(f"  {'레이블':<6} {'Train':>8} {'Val':>7} {'Test':>7}")
for lbl in range(5):
    tr = (train_df['label'] == lbl).sum()
    va = (val_df['label'] == lbl).sum()
    te = (test_df['label'] == lbl).sum()
    print(f"  {LABEL_NAMES[lbl]:<6} {tr:>8,} {va:>7,} {te:>7,}")

print()
print("[ 6. Train 내 합성 데이터 분포 ]")
if 'is_synthetic' in train_df.columns:
    syn_train = train_df[train_df['is_synthetic'] == 1]
    print(f"  Train 합성 행: {len(syn_train)}건")
    for lbl in [1, 2]:
        cnt = (syn_train['label'] == lbl).sum()
        print(f"    L{lbl}: {cnt}건  (upsample_synthetic 5 적용 시 → {cnt*5}건 추가 복제)")
else:
    print("  [경고] is_synthetic 컬럼 없음")

print()
print("[ 7. dedup_v2 대비 변화 요약 ]")
DATA_V2 = os.path.join(ROOT, '중요파일', 'data', 'raw', '재난문자_레이블링결과_dedup_v2.xlsx')
df_v2 = pd.read_excel(DATA_V2)
df_v2 = df_v2[df_v2['label'] != -1]
print(f"  dedup_v2: {len(df_v2):,}건 → dedup_v7: {len(df):,}건  (차이: {len(df)-len(df_v2):+,}건)")
for lbl in range(5):
    v2 = (df_v2['label'] == lbl).sum()
    v7 = (df['label'] == lbl).sum()
    print(f"  {LABEL_NAMES[lbl]}: {v2:,} → {v7:,}  ({v7-v2:+,})")

print()
print("검증 완료")
