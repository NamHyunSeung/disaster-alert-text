"""
의도적으로 모호하게 설계한 재난문자 모델 테스트
"""
import torch, sys
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR  = "koelectra_model"
MAX_LENGTH = 128
LABELS     = ['긴급', '주의', '일반']
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device).eval()

def predict(text):
    enc = tokenizer(text, truncation=True, padding='max_length',
                    max_length=MAX_LENGTH, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        p = F.softmax(model(**enc).logits, dim=-1)[0].cpu().numpy()
    pred = LABELS[p.argmax()]
    return pred, p

msgs = [
    # ── 긴급 vs 주의 ──────────────────────────────────────
    ("긴급↔주의",
     "[기상청] 오늘 17:23 경북 포항 북북동쪽 11km 해역 규모 3.8 지진 발생. "
     "여진이 발생할 수 있으니 주의하시기 바랍니다."),

    ("긴급↔주의",
     "[보령시] 오천면 일원 산사태 주의보 발령. "
     "산간지역 주민께서는 가까운 마을회관으로 대피를 권고드립니다."),

    ("긴급↔주의",
     "[행정안전부] 태풍 '카눈' 경로상 전남 해안지역 주민은 안전한 내륙으로 "
     "이동을 검토하시고, 저지대 침수 우려 지역은 사전 대피 바랍니다."),

    ("긴급↔주의",
     "[동해시] 오늘 14:05경 묵호항 인근 야산에서 소규모 산불이 발생하였습니다. "
     "현재 진화 중에 있으니 인근 주민은 접근을 삼가 바랍니다."),

    # ── 주의 vs 일반 ──────────────────────────────────────
    ("주의↔일반",
     "[질병관리청] 전국 코로나19 확진자 수가 일일 5만명을 초과하였습니다. "
     "고위험군은 외출을 자제하고 마스크 착용을 철저히 해주시기 바랍니다."),

    ("주의↔일반",
     "[서울시] 오늘 밤 수도권 일부 지역 노후 수도관 정비공사로 단수가 예상됩니다. "
     "사전에 식수를 확보해 두시기 바랍니다."),

    ("주의↔일반",
     "[인천시] 오늘 오후 인천 서구 일대 정전이 발생하였습니다. "
     "현재 한국전력에서 복구 중이며 2시간 내 해결 예정입니다."),

    ("주의↔일반",
     "[부산시] 태풍 북상으로 내일 오전까지 강한 비바람이 예상됩니다. "
     "해안가 및 산간지역 불필요한 외출을 자제하여 주시기 바랍니다."),

    # ── 긴급 vs 일반 ──────────────────────────────────────
    ("긴급↔일반",
     "[합천군] 오늘 22:10경 황강 수위가 급격히 상승하여 범람이 우려됩니다. "
     "하천변 인근 주민께서는 상황을 예의주시하시기 바랍니다."),

    ("긴급↔일반",
     "[고성군] 현재 고성군 전역에 대북전단 살포가 확인되었습니다. "
     "외출을 자제하고 낙하물을 절대 접촉하지 마시고 112에 신고해 주십시오."),

    # ── 표현만 바꿔서 강도 차이 테스트 ──────────────────────
    ("강도 테스트 A - 권고",
     "[남양주시] 북한강 수위 상승으로 하천변 대피를 권고합니다."),

    ("강도 테스트 B - 명령",
     "[남양주시] 북한강 수위 상승으로 하천변 주민은 즉시 대피하십시오."),

    ("강도 테스트 C - 경보",
     "[기상청] 경기 남부 호우경보 발효. 시간당 80mm 폭우가 예상됩니다."),

    ("강도 테스트 D - 주의보",
     "[기상청] 경기 남부 호우주의보 발효. 시간당 30mm 비가 예상됩니다."),
]

out = []
out.append("=" * 68)
out.append("의도적으로 모호하게 설계한 재난문자 분류 결과")
out.append("=" * 68)

for category, msg in msgs:
    pred, p = predict(msg)
    bar_긴급 = "#" * int(p[0] * 20)
    bar_주의 = "#" * int(p[1] * 20)
    bar_일반 = "#" * int(p[2] * 20)
    out.append(f"\n[{category}]")
    out.append(msg[:120])
    out.append(f"-> 예측: {pred}  |  긴급 {p[0]*100:5.1f}%  주의 {p[1]*100:5.1f}%  일반 {p[2]*100:5.1f}%")
    out.append(f"   긴급 [{bar_긴급:<20}]")
    out.append(f"   주의 [{bar_주의:<20}]")
    out.append(f"   일반 [{bar_일반:<20}]")

result = "\n".join(out)
with open("ambiguous_result.txt", "w", encoding="utf-8") as f:
    f.write(result)
print("저장 완료: ambiguous_result.txt")
