"""
dedup_v5.xlsx + 감염병_합성데이터_v2.xlsx → dedup_v6.xlsx
- is_synthetic 컬럼 추가 (합성=1, 원본=0)
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
V5   = os.path.join(BASE, 'raw', '재난문자_레이블링결과_dedup_v5.xlsx')
SYN2 = os.path.join(BASE, 'raw', '감염병_합성데이터_v2.xlsx')
OUT  = os.path.join(BASE, 'raw', '재난문자_레이블링결과_dedup_v6.xlsx')


def main():
    df_v5 = pd.read_excel(V5)
    df_v5 = df_v5[df_v5['label'] != -1].copy()
    df_v5['is_synthetic'] = 0
    print(f"v5 로드: {len(df_v5):,}건")

    df_syn2 = pd.read_excel(SYN2)
    print(f"합성v2 로드: {len(df_syn2):,}건  (L3={( df_syn2['label']==3).sum()}, L4={(df_syn2['label']==4).sum()})")

    rows = []
    for _, r in df_syn2.iterrows():
        row = {c: None for c in df_v5.columns}
        row['메시지내용']   = r['메시지내용']
        row['label']        = int(r['label'])
        row['is_synthetic'] = 1
        rows.append(row)

    df_add = pd.DataFrame(rows, columns=df_v5.columns)
    df_v6  = pd.concat([df_v5, df_add], ignore_index=True)
    df_v6.to_excel(OUT, index=False)

    print(f"\n[v6 저장] {OUT}")
    print(f"총 {len(df_v6):,}건  (+{len(df_add):,}건)")
    for lbl in sorted(df_v6['label'].dropna().unique()):
        cnt = (df_v6['label'] == lbl).sum()
        syn = ((df_v6['label'] == lbl) & (df_v6['is_synthetic'] == 1)).sum()
        print(f"  L{int(lbl)}: {cnt:,}건  (합성 {syn}건)")


if __name__ == '__main__':
    main()
