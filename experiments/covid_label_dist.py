"""
학습 데이터에서 코로나 관련 문자의 레이블 분포 분석
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '완성 모델', 'src'))

from dataset_v2 import load_and_split_v2

DATA_PATH = os.path.join(ROOT, '중요파일', 'data', 'raw',
                         '재난문자_레이블링결과_dedup_v2.xlsx')

train_df, val_df, test_df = load_and_split_v2(DATA_PATH)
all_df = train_df  # 학습 데이터만

KEYWORDS = ['코로나', '코로나19', '확진', '백신', '접종', '격리', '방역']

def is_covid(text):
    return any(kw in text for kw in KEYWORDS)

covid_mask = all_df['text'].apply(is_covid)
covid_df   = all_df[covid_mask]
non_covid  = all_df[~covid_mask]

print(f"학습 데이터 전체:   {len(all_df):,}개")
print(f"코로나 관련 문자:   {len(covid_df):,}개 ({len(covid_df)/len(all_df)*100:.1f}%)")
print(f"코로나 무관 문자:   {len(non_covid):,}개 ({len(non_covid)/len(all_df)*100:.1f}%)")
print()

print("[ 코로나 관련 - 레이블 분포 ]")
covid_label_counts = covid_df['label'].value_counts().sort_index()
for lbl, cnt in covid_label_counts.items():
    pct_of_covid = cnt / len(covid_df) * 100
    pct_of_all   = cnt / len(all_df) * 100
    print(f"  L{lbl}: {cnt:6,}개  (코로나내 {pct_of_covid:5.1f}%  /  전체내 {pct_of_all:4.1f}%)")

non_l0_covid = covid_df[covid_df['label'] != 0]
print(f"\n  → L0 아닌 코로나 문자: {len(non_l0_covid):,}개 ({len(non_l0_covid)/len(covid_df)*100:.1f}%)")

print()
print("[ L0 아닌 코로나 문자 샘플 (최대 20개) ]")
for _, row in non_l0_covid.head(20).iterrows():
    print(f"  L{row['label']}  {row['text'][:80]}")
