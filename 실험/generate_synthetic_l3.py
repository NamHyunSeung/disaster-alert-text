"""
신종 감염병 L3 합성 데이터 생성.

dedup_v3.xlsx에 500개의 합성 L3 신종 감염병 샘플을 추가해 dedup_v4.xlsx 생성.
목표: 마스킹 후에도 L3 판별 가능한 어휘(원인불명, 사망자, 치명률, 신종)를 포함시켜
     모델이 이 패턴을 L1 보건 공지와 구분하도록 학습.

마스킹 키워드: 발생, 경보, 주의보, 특보, 대피, 화재, 산불, 홍수, 태풍, 침수 등
→ 마스킹 후에도 남는 L3 시그널: 원인불명, 사망자/사망, 치명률, 신종, 봉쇄, 격리 명령
"""

import pandas as pd
import itertools
import random
import os

random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, '중요파일', 'data', 'raw',
                    '재난문자_레이블링결과_dedup_v3.xlsx')
DST  = os.path.join(ROOT, '중요파일', 'data', 'raw',
                    '재난문자_레이블링결과_dedup_v4.xlsx')

# ── 구성 요소 ──────────────────────────────────────────────────────────────

AGENCIES = [
    "[질병관리청]", "[보건복지부]", "[행정안전부]",
    "[서울시]", "[경기도]", "[인천시]", "[부산시]",
    "[중앙방역대책본부]", "[국가방역당국]",
]

DISEASE_TYPES = [
    "원인불명 호흡기 질환",
    "신종 바이러스 감염증",
    "원인불명 출혈열",
    "정체불명 폐렴",
    "신종 감염병",
    "원인불명 급성 호흡기 증후군",
    "신종 바이러스성 폐렴",
    "원인불명 중증 감염증",
    "원인불명 신경계 감염",
    "정체불명 호흡기 감염",
]

AREAS = [
    "서울 전역",
    "수도권 일대",
    "전국",
    "인천·경기 일대",
    "부산·경남 일대",
    "대구·경북 일대",
    "서울·인천·경기",
]

DEATH_COUNTS  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
FATALITY_RATE = [5, 8, 10, 12, 15, 18, 20, 25, 30]

# ── 템플릿 ─────────────────────────────────────────────────────────────────
# 5가지 다른 패턴으로 다양성 확보
# 모든 템플릿에 마스킹 후 남는 L3 시그널 포함 (원인불명/사망자/치명률/신종)

def tmpl_a(agency, disease, area, n):
    """테스트 케이스와 가장 유사한 패턴: 원인불명 + 집단 발생(사망자 N명)"""
    responses = [
        f"{area} 외출 자제, 마스크 착용 필수. 발열·기침 시 즉시 신고 바랍니다.",
        f"{area} 불필요한 외출 삼가고 의심 증상 시 즉시 보건소 신고 바랍니다.",
        f"{area} 마스크 착용 철저히 하고 발열·기침 시 즉시 신고 바랍니다.",
        f"{area} 외출 자제하고 발열·기침·호흡 곤란 시 즉시 신고 바랍니다.",
    ]
    resp = random.choice(responses)
    return f"{agency} {disease} 집단 발생(사망자 {n}명). {resp}"


def tmpl_b(agency, disease, area, n):
    """치명률 포함 패턴"""
    rate = random.choice(FATALITY_RATE)
    intros = [
        f"{agency} {area}에서 {disease} 집단 발생. 사망자 {n}명, 치명률 {rate}% 이상.",
        f"{agency} {disease} 집단 발병으로 사망 {n}명 확인. 치명률 {rate}% 이상.",
    ]
    responses = [
        f" {area} 외출 자제 및 의심 증상 시 즉시 신고 바랍니다.",
        f" {area} 불필요한 이동 자제 및 발열·기침 시 즉시 당국에 신고 바랍니다.",
        f" {area} 마스크 착용 필수, 의심 증상자 즉시 격리 신고 바랍니다.",
    ]
    return random.choice(intros) + random.choice(responses)


def tmpl_c(agency, disease, area, n):
    """사망 확인 + 원인 미확인 패턴"""
    phrases = [
        f"{agency} {disease} 집단 발병으로 사망 {n}명 확인. 원인균 미확인.",
        f"{agency} {area} {disease} 집단 환자 발생, 사망 {n}명. 원인 불명.",
        f"{agency} 원인 불명 {disease} 다수 집단 감염 확인, 사망자 {n}명.",
    ]
    responses = [
        f" {area} 외출 자제 및 발열·기침 시 즉시 신고 바랍니다.",
        f" {area} 이동 자제하고 의심 증상 시 즉시 보건소 신고 바랍니다.",
        f" 전파 경로 불명. {area} 마스크 착용, 외출 자제 바랍니다.",
    ]
    return random.choice(phrases) + random.choice(responses)


def tmpl_d(agency, disease, area, n):
    """광범위 전파 + 봉쇄/격리 명령 패턴"""
    intros = [
        f"{agency} {disease} 광범위 집단 감염 확산. 사망자 {n}명.",
        f"{agency} {area} {disease} 광범위 전파 확인. 사망 {n}명 보고.",
        f"{agency} 신종 {disease} 집단 감염, 사망자 {n}명 이상.",
    ]
    responses = [
        f" 의심 증상자 즉시 격리 신고 바랍니다.",
        f" 외출 자제 및 의심 증상자 즉시 격리 신고 바랍니다.",
        f" 마스크 착용 필수, 발열·기침·호흡 곤란 시 즉시 신고 바랍니다.",
        f" {area} 이동 자제 및 즉시 신고 바랍니다.",
    ]
    return random.choice(intros) + random.choice(responses)


def tmpl_e(agency, disease, area, n):
    """전파 경로 불명 + 위급 대응 패턴"""
    rate = random.choice(FATALITY_RATE)
    intros = [
        f"{agency} 전파 경로 불명의 {disease} 발병. 사망자 {n}명, 치명률 {rate}%.",
        f"{agency} 원인 및 전파 경로 미확인 {disease} 집단 발병, 사망 {n}명.",
        f"{agency} {disease} 집단 발병, 원인 불명. 사망자 {n}명 이상.",
    ]
    responses = [
        f" {area} 즉각 외출 자제 바랍니다.",
        f" {area} 외출 자제, 의심 증상 시 즉시 신고 바랍니다.",
        f" {area} 불필요한 외출 금지, 발열 시 즉시 신고 바랍니다.",
    ]
    return random.choice(intros) + random.choice(responses)


TEMPLATES = [tmpl_a, tmpl_b, tmpl_c, tmpl_d, tmpl_e]


def generate_samples(n_target: int) -> list[str]:
    """n_target개의 다양한 L3 합성 샘플 생성"""
    pool = []
    combos = list(itertools.product(AGENCIES, DISEASE_TYPES, AREAS, DEATH_COUNTS))
    random.shuffle(combos)

    for tmpl in TEMPLATES:
        for agency, disease, area, n in combos:
            text = tmpl(agency, disease, area, n)
            pool.append(text)

    # 중복 제거 후 n_target개 샘플링
    pool = list(dict.fromkeys(pool))   # 순서 유지 중복 제거
    random.shuffle(pool)
    return pool[:n_target]


def main():
    print("dedup_v3.xlsx 로드 중...")
    df = pd.read_excel(SRC)
    orig_len = len(df)
    print(f"  기존: {orig_len}개  (L3={( df['label']==3).sum()}개)")

    samples = generate_samples(500)
    new_df  = pd.DataFrame({'text': samples, 'label': 3})

    df = pd.concat([df, new_df], ignore_index=True)
    df.to_excel(DST, index=False)

    print(f"\n합성 샘플 추가: {len(new_df)}개  (L3 신종 감염병)")
    print(f"최종 행 수: {len(df)}개")
    print("\n최종 레이블 분포:")
    for lbl, cnt in df['label'].value_counts().sort_index().items():
        print(f"  L{lbl}: {cnt:,}개")
    print(f"\n저장 완료: {DST}")

    print("\n합성 샘플 예시 (처음 3개):")
    for s in samples[:3]:
        print(f"  {s}")


if __name__ == '__main__':
    main()
