import pandas as pd

df = pd.read_excel('재난문자_통합_2011~현재_분류완성.xlsx')
df['메시지내용'] = df['메시지내용'].fillna('')

# 긴급단계명 + 재해구분명 조합 확인
with open('data_check.txt', 'w', encoding='utf-8') as f:
    f.write(f"전체 행: {len(df):,}\n\n")
    f.write("=== 긴급단계명 분포 ===\n")
    f.write(str(df['긴급단계명'].value_counts()) + "\n\n")
    f.write("=== 재해구분명 분포 ===\n")
    f.write(str(df['재해구분명'].value_counts()) + "\n\n")

    # 긴급재난 메시지 샘플
    f.write("=== 긴급재난 메시지 샘플 10개 ===\n")
    for msg in df[df['긴급단계명'] == '긴급재난']['메시지내용'].head(10):
        f.write(str(msg)[:120] + "\n")

    # 키워드 등장 빈도 확인
    f.write("\n=== 주요 키워드 등장 빈도 ===\n")
    keywords = ['경보', '주의보', '특보', '예비특보', '즉시 대피', '대피명령',
                '대피 명령', '긴급대피', '쓰나미', '민방공', '지진 발생', '발생', '위험']
    for kw in keywords:
        cnt = df['메시지내용'].str.contains(kw, na=False).sum()
        f.write(f"  '{kw}': {cnt:,}건\n")

print("done → data_check.txt")
