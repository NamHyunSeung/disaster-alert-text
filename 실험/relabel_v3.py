"""
v3 재레이블링: beta 기반 Label 3→4 업그레이드

변경 기준:
  기존: score(alpha+beta) 기준 → score 5-6 = Label 3
  변경: beta=3 (위험지역 대피+경보) → Label 4 (alpha/지역크기 무관)

  즉, alpha=3, beta=3, score=6인 2,675건:
    Label 3 → Label 4

결과: 재난문자_레이블링결과_v3.xlsx
"""

import pandas as pd

SRC = '중요파일/data/raw/재난문자_레이블링결과.xlsx'
OUT = '중요파일/data/raw/재난문자_레이블링결과_v3.xlsx'

COLS = ['일련번호', '생성일시', '메시지내용', '재해구분명', '긴급단계명',
        '발송단계명', '발송일시', '수신일시', 'label', 'alpha', 'beta', 'score']

LABEL_NAMES = {0: '긴급아님', 1: '낮음', 2: '중간', 3: '높음', 4: '매우높음'}


def main():
    print(f"로드: {SRC}")
    df = pd.read_excel(SRC)
    df.columns = COLS

    before = df['label'].value_counts().sort_index()

    # beta=3 이면서 현재 Label 3 → Label 4 업그레이드
    mask = (df['label'] == 3) & (df['beta'] == 3)
    changed = mask.sum()
    df.loc[mask, 'label'] = 4

    after = df['label'].value_counts().sort_index()

    print(f"\n변경: {changed:,}건 (Label 3 beta=3 → Label 4)")
    print(f"\n{'레벨':<12} {'변경 전':>8} {'변경 후':>8} {'차이':>6}")
    print("-" * 38)
    for lbl in sorted(set(before.index) | set(after.index)):
        if lbl == -1:
            continue
        b = before.get(lbl, 0)
        a = after.get(lbl, 0)
        name = LABEL_NAMES.get(lbl, str(lbl))
        print(f"  {lbl} {name:<8} {b:>8,} {a:>8,} {a-b:>+6,}")

    df.to_excel(OUT, index=False)
    print(f"\n저장: {OUT}")


if __name__ == '__main__':
    main()
