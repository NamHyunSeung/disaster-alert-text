"""
Step 1: 3단계 중요도 레이블 생성 및 데이터 분할

[분류 기준]
긴급 (0): 즉각적 생명 위협 - 위급재난/긴급재난, 즉시대피·대피명령·지진발생 키워드
주의 (1): 경보·주의보·특보 발효, 화재·산불·홍수 등 재해 발생
일반 (2): 안전 정보·협조 요청·교통 안내 등 비긴급 안내

[레이블 생성 근거]
- 기존 긴급단계명(위급재난/긴급재난/안전안내) + 재해구분명 + 메시지 키워드를 복합 사용
- 메시지 내용의 직접적 행동 지시 여부가 긴급/주의 판단의 핵심
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 레이블 생성 규칙
# ─────────────────────────────────────────────

# 긴급 판정: 즉각 대피·생명 위협 지시 포함
EMERGENCY_KEYWORDS = [
    '즉시 대피', '대피명령', '대피 명령', '긴급대피', '긴급 대피',
    '지진 발생', '민방공 경보', '민방공경보', '쓰나미', '테러 발생',
    '폭발 발생',
]
EMERGENCY_CATEGORIES = frozenset({'지진해일', '테러', '폭발', '민방공'})

# 주의 판정: 경보·주의보·특보 발효 또는 재해 현장 발생 안내
CAUTION_ALERT_KEYWORDS = ['경보', '주의보', '특보', '예비특보']
CAUTION_OCCUR_CATEGORIES = frozenset({
    '화재', '산불', '홍수', '산사태', '지진', '붕괴', '환경오염사고', '폭발',
})


def create_label(row) -> int:
    urgency  = str(row.get('긴급단계명', '') or '')
    category = str(row.get('재해구분명', '') or '')
    msg      = str(row.get('메시지내용', '') or '')

    # ── 긴급 (0) ──────────────────────────────
    if urgency == '위급재난':
        return 0

    if urgency == '긴급재난':
        # 긴급재난은 모두 즉각 위험 상황
        return 0

    # 안전안내 중 긴급 키워드 포함
    if any(kw in msg for kw in EMERGENCY_KEYWORDS):
        return 0

    # 안전안내 중 최고위험 카테고리 + 경보
    if category in EMERGENCY_CATEGORIES and '경보' in msg:
        return 0

    # ── 주의 (1) ──────────────────────────────
    # 경보·주의보·특보 발효
    if any(kw in msg for kw in CAUTION_ALERT_KEYWORDS):
        return 1

    # 주요 재해 발생 안내
    if category in CAUTION_OCCUR_CATEGORIES and '발생' in msg:
        return 1

    # ── 일반 (2) ──────────────────────────────
    return 2


# ─────────────────────────────────────────────
# 데이터 로드 및 레이블 생성
# ─────────────────────────────────────────────

print("데이터 로드 중...")
df = pd.read_excel('재난문자_통합_2011~현재_분류완성.xlsx')
df['메시지내용'] = df['메시지내용'].fillna('')
print(f"전체 행: {len(df):,}")

print("레이블 생성 중...")
df['label'] = df.apply(create_label, axis=1)

LABEL_NAMES = {0: '긴급', 1: '주의', 2: '일반'}
df['label_name'] = df['label'].map(LABEL_NAMES)

# ─────────────────────────────────────────────
# 분포 확인
# ─────────────────────────────────────────────
counts = df['label_name'].value_counts()
ratios = df['label_name'].value_counts(normalize=True)

print("\n=== 레이블 분포 ===")
for name in ['긴급', '주의', '일반']:
    print(f"  {name}: {counts[name]:>7,}건  ({ratios[name]*100:.1f}%)")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('재난문자 3단계 중요도 분류 - 레이블 분포', fontsize=14)

colors = ['#e74c3c', '#f39c12', '#3498db']
axes[0].pie(counts[['긴급', '주의', '일반']],
            labels=['긴급', '주의', '일반'],
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[0].set_title('레이블 비율')

axes[1].bar(['긴급', '주의', '일반'],
            [counts['긴급'], counts['주의'], counts['일반']],
            color=colors)
axes[1].set_title('레이블 수량')
axes[1].set_ylabel('건수')
for i, (name, v) in enumerate(zip(['긴급', '주의', '일반'],
                                   [counts['긴급'], counts['주의'], counts['일반']])):
    axes[1].text(i, v + 500, f'{v:,}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('label_distribution.png', dpi=150, bbox_inches='tight')
print("\n분포 시각화 저장: label_distribution.png")

# ─────────────────────────────────────────────
# 연도별 레이블 분포 (추가 분석)
# ─────────────────────────────────────────────
df['year'] = pd.to_datetime(df['생성일시'], errors='coerce').dt.year
year_label = df.groupby(['year', 'label_name']).size().unstack(fill_value=0)
year_label = year_label.reindex(columns=['긴급', '주의', '일반'], fill_value=0)

fig, ax = plt.subplots(figsize=(14, 5))
year_label.plot(kind='bar', stacked=True, color=colors, ax=ax)
ax.set_title('연도별 레이블 분포')
ax.set_xlabel('연도')
ax.set_ylabel('건수')
ax.legend(['긴급', '주의', '일반'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('label_by_year.png', dpi=150, bbox_inches='tight')
print("연도별 분포 저장: label_by_year.png")

# ─────────────────────────────────────────────
# Train / Val / Test 분할 (70 / 15 / 15, stratified)
# ─────────────────────────────────────────────
cols = ['메시지내용', 'label', 'label_name']

train_df, temp_df = train_test_split(
    df[cols], test_size=0.30, stratify=df['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42
)

print(f"\n데이터 분할:")
print(f"  Train : {len(train_df):>7,}행")
print(f"  Val   : {len(val_df):>7,}행")
print(f"  Test  : {len(test_df):>7,}행")

for name, split in [('train', train_df), ('val', val_df), ('test', test_df)]:
    split.to_csv(f'data_{name}.csv', index=False, encoding='utf-8-sig')

print("\n저장 완료: data_train.csv / data_val.csv / data_test.csv")

# 레이블 규칙 요약 저장
summary = f"""=== 3단계 중요도 분류 기준 ===

[긴급 (0)] - 즉각적 생명 위협
  - 기존 긴급단계명이 '위급재난' 또는 '긴급재난'인 모든 메시지
  - 안전안내 메시지 중 직접 대피 지시 포함:
    키워드: {EMERGENCY_KEYWORDS}
  - 안전안내 + 최고위험 카테고리 + 경보: {sorted(EMERGENCY_CATEGORIES)}

[주의 (1)] - 경보·주의보 발효 및 재해 발생
  - 경보·주의보·특보·예비특보 키워드 포함: {CAUTION_ALERT_KEYWORDS}
  - 주요 재해 카테고리에서 '발생' 포함: {sorted(CAUTION_OCCUR_CATEGORIES)}

[일반 (2)] - 안전 정보 안내
  - 위 조건에 해당하지 않는 모든 안전안내 메시지
  - 예: 실종자 수색, 교통 통제, 코로나 안내, 수도 단수 안내 등

=== 최종 분포 ===
{df['label_name'].value_counts().to_string()}
비율: {df['label_name'].value_counts(normalize=True).round(4).to_string()}
"""
with open('label_rules.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
print("분류 기준 저장: label_rules.txt")
