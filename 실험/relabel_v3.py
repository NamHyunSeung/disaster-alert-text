"""
dedup_v2.xlsx -> dedup_v3.xlsx
L1 infection messages with severe keywords -> L3
"""
import os
import pandas as pd

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(ROOT, "중요파일", "data", "raw", "재난문자_레이블링결과_dedup_v2.xlsx")
DST_PATH = os.path.join(ROOT, "중요파일", "data", "raw", "재난문자_레이블링결과_dedup_v3.xlsx")

INF_KW    = ["코로나", "감염", "확진", "격리", "방역", "마스크", "호흡기",
             "질병관리", "인플루엔자", "독감", "바이러스", "전염"]
SEVERE_KW = ["사망", "집단 발생", "집단발생", "원인불명", "원인 불명"]

df = pd.read_excel(SRC_PATH)
print(f"원본 데이터: {len(df):,}건")
print("원본 레이블 분포:")
print(df["label"].value_counts().sort_index().to_string())

inf_mask    = df["text"].str.contains("|".join(INF_KW), na=False)
severe_mask = df["text"].str.contains("|".join(SEVERE_KW), na=False)
target      = (df["label"] == 1) & inf_mask & severe_mask

print(f"\n재레이블링 대상 (L1 -> L3): {target.sum()}건")
for _, row in df[target].iterrows():
    print(f"  [L1->L3] {row['text'][:100]}")

df.loc[target, "label"] = 3

print("\n변경 후 레이블 분포:")
print(df["label"].value_counts().sort_index().to_string())

df.to_excel(DST_PATH, index=False)
print(f"\n저장 완료: {DST_PATH}")
