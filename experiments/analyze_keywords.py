import pandas as pd

df = pd.concat([
    pd.read_csv('중요파일/data/processed/data_train.csv', encoding='utf-8-sig'),
    pd.read_csv('중요파일/data/processed/data_val.csv',   encoding='utf-8-sig'),
    pd.read_csv('중요파일/data/processed/data_test.csv',  encoding='utf-8-sig'),
])
LABEL = {0:'긴급', 1:'주의', 2:'일반'}
df['label_name'] = df['label'].map(LABEL)

CERT_EMERG   = ['즉시 대피','대피명령','대피 명령','긴급대피','긴급 대피','신속히 대피','지진 발생','쓰나미','민방공 경보','민방공경보','테러 발생']
CERT_CAUTION = ['호우경보','호우주의보','태풍경보','태풍주의보','한파경보','한파주의보','폭염경보','폭염주의보','대설경보','대설주의보','강풍경보','강풍주의보','풍랑경보','풍랑주의보']
CERT_GENERAL = ['찾습니다','실종된']

def show(title, keywords):
    print(f'\n=== {title} ===')
    print(f'  {"키워드":<12} {"총건수":>6}  {"긴급":>7}  {"주의":>7}  {"일반":>7}')
    print('  ' + '-'*52)
    for kw in keywords:
        mask = df['메시지내용'].str.contains(kw, na=False)
        cnt = mask.sum()
        if cnt == 0:
            continue
        dist = df[mask]['label_name'].value_counts()
        e = dist.get('긴급', 0) / cnt * 100
        c = dist.get('주의', 0) / cnt * 100
        n = dist.get('일반', 0) / cnt * 100
        print(f'  {kw:<12} {cnt:>6}건  {e:6.1f}%  {c:6.1f}%  {n:6.1f}%')

show('긴급 키워드 상관관계', CERT_EMERG)
show('주의 키워드 상관관계', CERT_CAUTION)
show('일반 키워드 상관관계', CERT_GENERAL)
