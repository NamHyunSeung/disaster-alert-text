import pandas as pd

df = pd.concat([
    pd.read_csv('중요파일/data/processed/data_train.csv', encoding='utf-8-sig'),
    pd.read_csv('중요파일/data/processed/data_val.csv',   encoding='utf-8-sig'),
    pd.read_csv('중요파일/data/processed/data_test.csv',  encoding='utf-8-sig'),
])
df['label_name'] = df['label'].map({0:'긴급', 1:'주의', 2:'일반'})

CERT_EMERG   = ['즉시 대피','대피명령','대피 명령','긴급대피','긴급 대피','신속히 대피',
                '지진 발생','쓰나미','민방공 경보','민방공경보','테러 발생']
CERT_CAUTION = ['호우경보','호우주의보','태풍경보','태풍주의보','한파경보','한파주의보',
                '폭염경보','폭염주의보','대설경보','대설주의보','강풍경보','강풍주의보',
                '풍랑경보','풍랑주의보']
CERT_GENERAL = ['찾습니다','실종된']

msg = df['메시지내용'].fillna('')

has_emerg   = msg.apply(lambda x: any(kw in x for kw in CERT_EMERG))
has_caution = msg.apply(lambda x: any(kw in x for kw in CERT_CAUTION))
has_general = msg.apply(lambda x: any(kw in x for kw in CERT_GENERAL))

# 실제 오탐: Stage 1 규칙이 잘못 판정하는 케이스
# 주의 오탐: 긴급 레이블인데 긴급 키워드 없고 주의 키워드만 있음
real_caution_fp = df[
    (df['label_name'] == '긴급') &
    (~has_emerg) &
    (has_caution)
]
print(f'주의 키워드 실제 오탐 (긴급→주의): {len(real_caution_fp)}건')
if len(real_caution_fp) > 0:
    print(real_caution_fp['메시지내용'].str[:80].tolist())

# 일반 오탐: 긴급/주의 레이블인데 긴급/주의 키워드 없고 일반 키워드만 있음
real_general_fp = df[
    (df['label_name'].isin(['긴급','주의'])) &
    (~has_emerg) &
    (~has_caution) &
    (has_general)
]
print(f'\n일반 키워드 실제 오탐 (긴급/주의→일반): {len(real_general_fp)}건')
if len(real_general_fp) > 0:
    print(real_general_fp[['label_name','메시지내용']].assign(msg=real_general_fp['메시지내용'].str[:80]).drop(columns='메시지내용').to_string())
