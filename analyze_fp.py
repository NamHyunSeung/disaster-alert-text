import pandas as pd

df = pd.concat([
    pd.read_csv('중요파일/data/processed/data_train.csv', encoding='utf-8-sig'),
    pd.read_csv('중요파일/data/processed/data_val.csv',   encoding='utf-8-sig'),
    pd.read_csv('중요파일/data/processed/data_test.csv',  encoding='utf-8-sig'),
])
df['label_name'] = df['label'].map({0:'긴급', 1:'주의', 2:'일반'})

CERT_CAUTION = ['호우경보','호우주의보','태풍경보','태풍주의보','한파경보','한파주의보',
                '폭염경보','폭염주의보','대설경보','대설주의보','강풍경보','강풍주의보',
                '풍랑경보','풍랑주의보']
CERT_GENERAL = ['찾습니다','실종된']

# 주의 키워드 오탐: 키워드 매칭됐는데 실제 레이블이 주의가 아닌 것
print('=== 주의 키워드 오탐 ===')
caution_fp_emerg = 0
caution_fp_normal = 0
for kw in CERT_CAUTION:
    mask = df['메시지내용'].str.contains(kw, na=False)
    sub = df[mask]
    fp_e = (sub['label_name'] == '긴급').sum()
    fp_n = (sub['label_name'] == '일반').sum()
    caution_fp_emerg  += fp_e
    caution_fp_normal += fp_n
    if fp_e > 0 or fp_n > 0:
        print(f'  {kw:<10}  긴급오탐 {fp_e:>4}건  일반오탐 {fp_n:>4}건')

print(f'\n  주의 키워드 오탐 합계: 긴급 {caution_fp_emerg}건 / 일반 {caution_fp_normal}건')
print(f'  (주의로 잘못 분류될 수 있는 긴급: {caution_fp_emerg}건)')

print()
print('=== 일반 키워드 오탐 ===')
general_fp_emerg = 0
general_fp_caution = 0
for kw in CERT_GENERAL:
    mask = df['메시지내용'].str.contains(kw, na=False)
    sub = df[mask]
    fp_e = (sub['label_name'] == '긴급').sum()
    fp_c = (sub['label_name'] == '주의').sum()
    general_fp_emerg   += fp_e
    general_fp_caution += fp_c
    print(f'  {kw:<10}  긴급오탐 {fp_e:>4}건  주의오탐 {fp_c:>4}건')

print(f'\n  일반 키워드 오탐 합계: 긴급 {general_fp_emerg}건 / 주의 {general_fp_caution}건')
