"""
v3 원본 데이터 + 합성 감염병 데이터(L3/L4) → dedup_v5.xlsx

- 베이스: raw/재난문자_레이블링결과_dedup_v3.xlsx  (v9n 학습 데이터)
- 추가:   raw/감염병_합성데이터.xlsx               (신종감염병 L3/L4 140건)
- 출력:   raw/재난문자_레이블링결과_dedup_v5.xlsx
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
V3   = os.path.join(BASE, 'raw', '재난문자_레이블링결과_dedup_v3.xlsx')
SYN  = os.path.join(BASE, 'raw', '감염병_합성데이터.xlsx')
OUT  = os.path.join(BASE, 'raw', '재난문자_레이블링결과_dedup_v5.xlsx')


def main():
    # ── 1. 원본 v3 로드 ──────────────────────
    df_v3 = pd.read_excel(V3)
    print(f"v3 로드: {len(df_v3):,}건")
    print(f"  컬럼: {list(df_v3.columns)}")

    # label -1 제거 (build_preprocessed와 동일)
    df_v3 = df_v3[df_v3['label'] != -1].copy()
    print(f"  유효 샘플: {len(df_v3):,}건")
    label_v3 = df_v3['label'].value_counts().sort_index()
    for lbl, cnt in label_v3.items():
        print(f"    L{lbl}: {cnt:,}건")

    # ── 2. 합성 감염병 데이터 로드 ─────────
    df_syn = pd.read_excel(SYN)
    print(f"\n합성 데이터 로드: {len(df_syn):,}건")
    for lbl, cnt in df_syn['label'].value_counts().sort_index().items():
        print(f"    L{lbl}: {cnt:,}건")

    # v3 컬럼 구조에 맞게 행 생성 (build_preprocessed에서 메시지내용, label만 사용)
    rows = []
    for _, r in df_syn.iterrows():
        row = {c: None for c in df_v3.columns}
        row['메시지내용'] = r['메시지내용']
        row['label']     = int(r['label'])
        rows.append(row)

    df_add = pd.DataFrame(rows, columns=df_v3.columns)

    # ── 3. 합치기 & 저장 ────────────────────
    df_v5 = pd.concat([df_v3, df_add], ignore_index=True)
    df_v5.to_excel(OUT, index=False)

    print(f"\n[v5 저장] {OUT}")
    print(f"총 {len(df_v5):,}건  (+{len(df_add):,}건)")
    label_v5 = df_v5[df_v5['label'] != -1]['label'].value_counts().sort_index()
    for lbl, cnt in label_v5.items():
        before = label_v3.get(lbl, 0)
        diff   = cnt - before
        print(f"  L{lbl}: {cnt:,}건  (이전 {before:,} → +{diff:,})")


if __name__ == '__main__':
    main()
