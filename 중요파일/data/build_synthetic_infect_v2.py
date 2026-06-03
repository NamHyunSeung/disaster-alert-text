"""
추가 감염병 합성 데이터 생성 (v2) — 프로그래매틱 조합 방식
L3 +100건: 다양한 바이러스/시설/지역 조합
L4 +160건: 고사망자·광역확산·병원과부하·변이 바이러스 시나리오
출력: raw/감염병_합성데이터_v2.xlsx
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, 'raw', '감염병_합성데이터_v2.xlsx')


# ── L3 템플릿 재료 ──────────────────────────────────────────────────────────
L3_AGENCIES = [
    "[질병관리청]", "[보건복지부]", "[행정안전부]", "[서울특별시]", "[경기도청]",
    "[부산시청]",   "[인천시청]",   "[대구시청]",   "[광주시청]",   "[대전시청]",
    "[충청남도청]", "[전라북도청]", "[경상남도청]", "[강원도청]",   "[제주도청]",
]
L3_DISEASES = [
    "변이 인플루엔자", "원숭이두창", "홍역 변이 바이러스", "뎅기열",
    "콜레라 변이균", "지카바이러스", "MERS 유사 바이러스", "에볼라 유사 바이러스",
    "결핵 변이균", "수두 변이 바이러스", "라싸열 유사 바이러스", "니파 바이러스",
]
L3_FACILITIES = [
    "유치원", "군부대", "선박", "기숙사", "교정시설",
    "콜센터", "스포츠센터", "국제공항 입국자", "학원 복합시설", "다문화센터",
    "노인요양원", "재활병원",
]
L3_LOCATIONS = [
    "서울 송파구", "경기 안성시", "부산 북구",   "인천 연수구", "대구 동구",
    "경남 통영시", "전북 전주시", "충북 청주시", "강원 원주시", "제주 서귀포시",
    "경기 여주시", "충남 당진시", "전남 무안군", "경북 영주시", "강원 삼척시",
]
L3_CASES  = [6, 8, 10, 12, 14, 16, 18, 21, 24, 27, 30, 33, 36, 42, 48]
L3_DEATHS = [0, 0, 1, 0, 1, 0, 2, 1, 0, 1, 0, 2, 1, 0, 1]
L3_SYMPTOMS = [
    "발열·호흡곤란",   "발열·발진·근육통", "발열·기침·인후통",
    "발열·구토·설사",  "발열·피부 출혈",   "발열·황달·근육통",
    "발열·두통·관절통","발열·전신 무력감",
]
L3_HOTLINES = ["☎1339", "☎119", "☎보건소", "☎1339 또는 ☎119"]

l3_msgs = []
i = 0
for ag in L3_AGENCIES:
    for dis in L3_DISEASES:
        for fac in L3_FACILITIES:
            for loc in L3_LOCATIONS:
                if len(l3_msgs) >= 100:
                    break
                n  = L3_CASES[i % len(L3_CASES)]
                d  = L3_DEATHS[i % len(L3_DEATHS)]
                sy = L3_SYMPTOMS[i % len(L3_SYMPTOMS)]
                h  = L3_HOTLINES[i % len(L3_HOTLINES)]
                if d > 0:
                    msg = (f"{ag} {loc} {fac} 내 {dis} 집단 발생"
                           f"(확진 {n}명, 사망 {d}명). 역학조사 진행 중. "
                           f"{sy} 증상 시 즉시 자가 격리, {h} 신고 바랍니다.")
                else:
                    msg = (f"{ag} {loc} {fac} 내 {dis} 집단 감염 확인"
                           f"(확진 {n}명). 방역당국 현장 대응 중. "
                           f"{sy} 증상 시 외출 자제, {h} 신고 바랍니다.")
                l3_msgs.append(msg)
                i += 1
            if len(l3_msgs) >= 100:
                break
        if len(l3_msgs) >= 100:
            break
    if len(l3_msgs) >= 100:
        break


# ── L4 템플릿 재료 ──────────────────────────────────────────────────────────
L4_AGENCIES = [
    "[질병관리청]", "[보건복지부]", "[행정안전부]", "[질병관리청]", "[보건복지부]",
]
L4_DISEASES = [
    "원인불명 신종 바이러스",   "치명적 신종 감염병",
    "신종 출혈열 바이러스",     "고위험 변이 바이러스",
    "원인불명 집단 감염병",     "신종 공기감염 바이러스",
    "무증상 전파 신종 감염병",  "변이 고위험 바이러스",
]
L4_SPREAD = [
    "수도권", "전국", "수도권·충청권", "전국 동시다발", "수도권·영남권", "광역 동시",
]
L4_DEATHS = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 25, 28, 30, 35, 40, 50]
L4_CASES  = [50, 80, 120, 160, 210, 270, 330, 400, 480, 560, 650, 750, 900, 1100, 1400]
L4_ORDERS = [
    "즉시 외출 금지. 가정 내 격리 이행.",
    "집합 금지 명령 발령. 이동 즉시 중단.",
    "외출 금지 명령. 의료기관 방문 전 ☎1339 필수.",
    "긴급 격리 명령. 귀가 후 자가 격리 실시.",
    "이동 금지 명령 발령. 즉시 귀가 바랍니다.",
]
L4_EXTRA = [
    "",
    "병원 과부하 우려. 경증 자택 대기. ",
    "무증상 전파 확인. 마스크 착용 필수. ",
    "변이 바이러스 출현 경보. ",
    "의료 시스템 포화 위기. 경증 자택 치료. ",
    "치명률 15% 이상 추정. ",
    "2차 감염 확산 경보. ",
    "항바이러스제 부족 경고. ",
]
L4_HOTLINES = ["☎1339", "☎119", "☎1339 또는 ☎119"]

l4_msgs = []
j = 0
for sp in L4_SPREAD:
    for dis in L4_DISEASES:
        for ag in L4_AGENCIES:
            if len(l4_msgs) >= 160:
                break
            dc = L4_DEATHS[j % len(L4_DEATHS)]
            cc = L4_CASES[j % len(L4_CASES)]
            od = L4_ORDERS[j % len(L4_ORDERS)]
            ex = L4_EXTRA[j % len(L4_EXTRA)]
            h  = L4_HOTLINES[j % len(L4_HOTLINES)]
            msg = (f"{ag} {sp} {dis} 급속 확산"
                   f"(확진 {cc}명, 사망 {dc}명). "
                   f"{ex}{od} 발열·호흡기 증상 시 {h} 신고 바랍니다.")
            l4_msgs.append(msg)
            j += 1
        if len(l4_msgs) >= 160:
            break
    if len(l4_msgs) >= 160:
        break


def main():
    records = (
        [{'메시지내용': m, 'label': 3} for m in l3_msgs] +
        [{'메시지내용': m, 'label': 4} for m in l4_msgs]
    )
    df = pd.DataFrame(records)
    df.to_excel(OUT, index=False)
    print(f"저장: {OUT}")
    print(f"총 {len(df)}건  (L3={len(l3_msgs)}, L4={len(l4_msgs)})")


if __name__ == '__main__':
    main()
