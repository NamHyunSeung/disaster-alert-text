"""
Step 1 (v2): 명확한 기준으로 레이블 재생성

[긴급] 즉각적 생명 위협 + 강제적 행동 지시
  - 위급재난 / 긴급재난
  - 즉시/신속 대피, 대피명령, 긴급대피
  - 지진 발생, 쓰나미, 민방공, 테러
  - 하천 범람 발생 + 대피

[주의] 잠재적 위험, 준비·행동 변화 필요
  - 기상 특보(경보/주의보/특보) 발효
  - 대피 (권고·당부 포함, 긴급 제외)
  - 주요 재해 카테고리 발생 (통제 중)
  - 감염병 경보/주의보 발령

[일반] 정보성 안내, 즉각 위험 없음
  - 확진자 수 현황 (경보 미발령)
  - 기상 예상 (특보 미발효)
  - 실종자 수색, 교통 안내, 단수/정전 등
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────
# 레이블 생성 규칙
# ─────────────────────────────────────────
EMERG_KW = [
    '즉시 대피', '대피명령', '대피 명령', '긴급대피', '긴급 대피', '신속히 대피',
    '지진 발생', '쓰나미', '민방공', '테러',
]
HIGH_RISK_CATEG = frozenset({'지진해일', '테러', '폭발'})

ALERT_KW = ['경보', '주의보', '특보', '예비특보']

CAUTION_CATEG = frozenset({
    '호우', '태풍', '산불', '홍수', '한파', '폭염', '대설', '강풍',
    '산사태', '풍랑', '화재', '환경오염사고', '붕괴',
})


def create_label(row) -> int:
    urgency  = str(row.get('긴급단계명', '') or '')
    category = str(row.get('재해구분명', '') or '')
    msg      = str(row.get('메시지내용', '') or '')

    # ── 긴급 ─────────────────────────────
    if urgency in ('위급재난', '긴급재난'):
        return 0
    if any(kw in msg for kw in EMERG_KW):
        return 0
    if category in HIGH_RISK_CATEG and '경보' in msg:
        return 0
    # 하천 범람 발생 + 대피 → 긴급
    if ('범람' in msg or '침수 발생' in msg) and '대피' in msg:
        return 0

    # ── 주의 ─────────────────────────────
    # 범람/침수 우려 (대피 없음 → 긴급 아님, 주의 수준)
    if '범람' in msg or '침수 우려' in msg:
        return 1
    # 기상 특보 발효
    if any(kw in msg for kw in ALERT_KW):
        return 1
    # 대피 언급 (긴급 키워드 없음 → 권고 수준)
    if '대피' in msg:
        return 1
    # 주요 재해 카테고리 발생
    if category in CAUTION_CATEG and '발생' in msg:
        return 1
    # 메시지 내용 기반 추가 주의 패턴 (재해구분명 없어도 판단)
    if any(kw in msg for kw in ['화재 발생', '산불 발생', '산불 위험', '산불발생']):
        return 1
    if any(kw in msg for kw in ['연기 발생', '연기가 발생', '연기발생', '다량의 연기']):
        return 1
    if any(kw in msg for kw in ['낙석', '월파']):
        return 1
    # 감염병 경보/주의보 (확진자 수 안내는 일반)
    if category == '전염병' and any(kw in msg for kw in ['경보', '주의보', '비상사태']):
        return 1

    # ── 일반 ─────────────────────────────
    return 2


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
print("데이터 로드 중...")
df = pd.read_excel('../data/raw/재난문자_통합_2011~현재_분류완성.xlsx')
df['메시지내용'] = df['메시지내용'].fillna('')
print(f"전체 {len(df):,}행")

print("레이블 생성 중...")
df['label'] = df.apply(create_label, axis=1)
LABEL_NAMES = {0: '긴급', 1: '주의', 2: '일반'}
df['label_name'] = df['label'].map(LABEL_NAMES)

counts = df['label_name'].value_counts()
print("\n=== 레이블 분포 ===")
for name in ['긴급', '주의', '일반']:
    print(f"  {name}: {counts[name]:>7,}건  ({counts[name]/len(df)*100:.1f}%)")

# 분포 시각화
colors = ['#e74c3c', '#f39c12', '#3498db']
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle('재난문자 3단계 분류 (v2 기준)', fontsize=13)

axes[0].pie([counts['긴급'], counts['주의'], counts['일반']],
            labels=['긴급', '주의', '일반'], autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[0].set_title('비율')

axes[1].bar(['긴급', '주의', '일반'],
            [counts['긴급'], counts['주의'], counts['일반']], color=colors)
for i, n in enumerate(['긴급', '주의', '일반']):
    axes[1].text(i, counts[n] + 500, f'{counts[n]:,}', ha='center', fontsize=9)
axes[1].set_title('수량')
plt.tight_layout()
plt.savefig('../../outputs/label_distribution_v2.png', dpi=150, bbox_inches='tight')
print("분포 저장: label_distribution_v2.png")

# 이전 레이블과 비교
try:
    old = pd.read_csv('../data/processed/data_train.csv', encoding='utf-8-sig')
    old_full = pd.concat([
        pd.read_csv('../data/processed/data_train.csv', encoding='utf-8-sig'),
        pd.read_csv('../data/processed/data_val.csv',   encoding='utf-8-sig'),
        pd.read_csv('../data/processed/data_test.csv',  encoding='utf-8-sig'),
    ])
    changed = (df['label'].values != old_full.sort_index()['label'].values).sum()
    print(f"\n이전 대비 변경된 레이블: {changed:,}건 ({changed/len(df)*100:.1f}%)")
except Exception:
    pass

# 데이터 분할 (70/15/15 stratified)
cols = ['메시지내용', 'label', 'label_name']
train_df, temp_df = train_test_split(df[cols], test_size=0.30,
                                     stratify=df['label'], random_state=42)
val_df, test_df   = train_test_split(temp_df,  test_size=0.50,
                                     stratify=temp_df['label'], random_state=42)

print(f"\n분할: Train {len(train_df):,} / Val {len(val_df):,} / Test {len(test_df):,}")
for name, split in [('train', train_df), ('val', val_df), ('test', test_df)]:
    split.to_csv(f'../data/processed/data_{name}.csv', index=False, encoding='utf-8-sig')
print("저장 완료: data_train.csv / data_val.csv / data_test.csv")

# 레이블 기준 저장
with open('../../outputs/label_rules_v2.txt', 'w', encoding='utf-8') as f:
    f.write(__doc__)
    f.write(f"\n\n=== 분포 ===\n")
    for n in ['긴급', '주의', '일반']:
        f.write(f"{n}: {counts[n]:,}건 ({counts[n]/len(df)*100:.1f}%)\n")
print("기준 저장: label_rules_v2.txt")
