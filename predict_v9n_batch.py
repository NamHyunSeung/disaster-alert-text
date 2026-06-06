"""v9n 모델로 재난문자 리스트 일괄 분류"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR     = "model_v9n"
TOKENIZER_DIR = "tokenizer_v9n"
MAX_LENGTH    = 96

LABEL_NAMES = {0: "L0 (정보없음)", 1: "L1 (일반)", 2: "L2 (주의)", 3: "L3 (경보)", 4: "L4 (긴급)"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

messages = [
    "컴퓨터 해킹으로 전력망 마비. 병원 비상전원 가동 중. 인근 주민 대피 권고.",
    "해저 광케이블 절단으로 광역 통신 두절. 비상 연락망 사용 바람.",
    "태양폭풍으로 GPS·통신위성 오작동. 항공·선박 운항 위험. 야외 활동 자제.",
    "반도체 공장 냉각수 유출. 독성물질 검출. 인근 2km 주민 실내 대피.",
    "해수면 급상승으로 해안 저지대 침수 진행 중. 고지대 이동 바람.",
    "○○시 가스관 노후로 인한 누출 감지. 불꽃 사용 삼가고 환기 권장.",
    "열돔 현상 지속. 사흘째 40도 이상. 야외 노동자·노약자 건강 위협.",
    "지하철 전기 합선으로 연기 발생. 해당 노선 운행 중단. 승객 역사 밖으로.",
    "○○강 상류 댐 균열 발견. 하류 주민 고지대 이동 권장.",
    "핵발전소 냉각 계통 이상. 반경 10km 주민 예방 대피 진행 중.",
    "○○ 지역에 대규모 산불 확산 중. 바람 방향 주의 요망. 연기 흡입 피해야.",
    "오늘 밤부터 기온이 영하 20도까지 떨어질 예정. 수도관 관리 각별히 신경 쓰세요.",
    "신종 호흡기 바이러스 지역사회 전파 확인. 마스크 착용·손씻기 생활화.",
    "○○ 바이러스 변종 출현. 기존 백신 효능 불명확. 외출 최소화 권고.",
    "황사 농도 '매우 나쁨'. 미세먼지 동반. 창문 닫고 외출 시 마스크 필수.",
    "○○항 인근 해역 적조 발생. 어패류 채취·섭취 주의.",
    "AI(조류인플루엔자) 의심 사례 발생. 가금류 접촉 자제.",
    "도심 집중호우로 지하차도 침수. 우회 도로 이용 바람.",
    "고압 전선 도로 낙하. 접근하지 마시고 신고 바람.",
    "○○시 식수원 오염 의심. 수돗물 음용 잠정 중단. 생수 배급 예정.",
]

print(f"{'No':<3} {'분류':^12} {'L0':>6} {'L1':>6} {'L2':>6} {'L3':>6} {'L4':>6}  메시지")
print("─" * 110)

for i, msg in enumerate(messages, 1):
    inputs = tokenizer(msg, truncation=True, padding="max_length",
                       max_length=MAX_LENGTH, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0].cpu()
    pred = int(probs.argmax())
    p = [f"{probs[j]*100:.1f}%" for j in range(5)]
    short = msg[:45] + ("…" if len(msg) > 45 else "")
    print(f"{i:<3} {LABEL_NAMES[pred]:^12} {p[0]:>6} {p[1]:>6} {p[2]:>6} {p[3]:>6} {p[4]:>6}  {short}")
