"""v9n + kNN OOD 탐지 - 20개 메시지 분류"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '완성 모델', 'src'))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ood_knn import KNNOOD

MODEL_DIR  = "model_v9n"
TOK_DIR    = "tokenizer_v9n"
OOD_STATS  = "실험/knn_ood_stats.pt"
MAX_LEN    = 96

LABEL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4']

# (메시지, 정답 라벨)
messages = [
    ("컴퓨터 해킹으로 전력망 마비. 병원 비상전원 가동 중. 인근 주민 대피 권고.",        4),
    ("해저 광케이블 절단으로 광역 통신 두절. 비상 연락망 사용 바람.",                  3),
    ("태양폭풍으로 GPS통신위성 오작동. 항공선박 운항 위험. 야외 활동 자제.",            3),
    ("반도체 공장 냉각수 유출. 독성물질 검출. 인근 2km 주민 실내 대피.",               4),
    ("해수면 급상승으로 해안 저지대 침수 진행 중. 고지대 이동 바람.",                  4),
    ("가스관 노후로 인한 누출 감지. 불꽃 사용 삼가고 환기 권장.",                     2),
    ("열돔 현상 지속. 사흘째 40도 이상. 야외 노동자 노약자 건강 위협.",                2),
    ("지하철 전기 합선으로 연기 발생. 해당 노선 운행 중단. 승객 역사 밖으로.",          3),
    ("강 상류 댐 균열 발견. 하류 주민 고지대 이동 권장.",                             4),
    ("핵발전소 냉각 계통 이상. 반경 10km 주민 예방 대피 진행 중.",                    4),
    ("대규모 산불 확산 중. 바람 방향 주의 요망. 연기 흡입 피해야.",                   3),
    ("오늘 밤부터 기온이 영하 20도까지 떨어질 예정. 수도관 관리 각별히 신경 쓰세요.",    1),
    ("신종 호흡기 바이러스 지역사회 전파 확인. 마스크 착용 손씻기 생활화.",             1),
    ("바이러스 변종 출현. 기존 백신 효능 불명확. 외출 최소화 권고.",                  2),
    ("황사 농도 매우 나쁨. 미세먼지 동반. 창문 닫고 외출 시 마스크 필수.",             1),
    ("항 인근 해역 적조 발생. 어패류 채취 섭취 주의.",                               1),
    ("AI조류인플루엔자 의심 사례 발생. 가금류 접촉 자제.",                            1),
    ("도심 집중호우로 지하차도 침수. 우회 도로 이용 바람.",                           3),
    ("고압 전선 도로 낙하. 접근하지 마시고 신고 바람.",                               3),
    ("식수원 오염 의심. 수돗물 음용 잠정 중단. 생수 배급 예정.",                      3),
]

# cosine OOD 결과 (비교용)
COSINE_OOD = {
    1:'in', 2:'in', 3:'in', 4:'in', 5:'OOD',
    6:'in', 7:'OOD', 8:'in', 9:'OOD', 10:'OOD',
    11:'in', 12:'OOD', 13:'in', 14:'in', 15:'OOD',
    16:'in', 17:'in', 18:'in', 19:'OOD', 20:'in',
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device); model.eval()

print("kNN OOD 통계 로드 중...")
detector = KNNOOD.load(OOD_STATS)
print(f"  k={detector.k}, 학습 샘플={len(detector.train_embs)}개")
print(f"  thresholds: {[f'L{i}={v:.4f}' for i,v in enumerate(detector.class_thresholds)]}")

lines = []
correct = 0
ood_correct = 0  # OOD로 잡은 것 중 실제로 틀린 예측

header = (f"{'#':>2}  {'정답':^3}  {'예측':^4}  {'신뢰도':>6}  "
          f"{'kNN거리':>7}  {'kNN class':^9}  {'OOD':^5}  {'cos':^5}  {'정답?':^4}  메시지")
lines.append(header)
lines.append("─" * 125)

results = []
for i, (msg, true_label) in enumerate(messages, 1):
    enc = tokenizer(msg, truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        probs = F.softmax(out.logits, dim=-1)[0].cpu()

    pred    = int(probs.argmax())
    cls_emb = out.hidden_states[-1][:, 0, :].squeeze(0).cpu()

    dist, knn_class = detector.score(cls_emb)
    ood = dist > detector._threshold_for(knn_class)

    ok = pred == true_label
    if ok:
        correct += 1

    cos_str = COSINE_OOD.get(i, '?')
    ood_str = "OOD" if ood else "in "
    thr     = detector._threshold_for(knn_class)
    knn_str = f"L{knn_class}({thr:.4f})"
    short   = msg[:42] + ("…" if len(msg) > 42 else "")

    lines.append(
        f"{i:>2}  L{true_label:^3}  {LABEL_NAMES[pred]:^4}  {probs[pred]*100:5.1f}%  "
        f"{dist:.5f}  {knn_str:^13}  {ood_str:^5}  {cos_str:^5}  {'O' if ok else 'X':^4}  {short}"
    )
    results.append({'i': i, 'pred': pred, 'true': true_label, 'dist': dist,
                    'knn_class': knn_class, 'ood': ood, 'ok': ok})

n = len(messages)
n_ood = sum(1 for r in results if r['ood'])
n_ood_wrong = sum(1 for r in results if r['ood'] and not r['ok'])
n_in_wrong  = sum(1 for r in results if not r['ood'] and not r['ok'])

lines.append("─" * 125)
lines.append(f"정확도: {correct}/{n} ({correct/n*100:.1f}%)")
lines.append(f"OOD 탐지: {n_ood}개  "
             f"(탐지된 것 중 실제 오답: {n_ood_wrong}/{n_ood})")
lines.append(f"in-distribution 오답: {n_in_wrong}개  "
             f"(탐지 못한 오류 — cosine OOD와 비교)")
lines.append("")
lines.append("[ kNN threshold ]")
for c, thr in enumerate(detector.class_thresholds):
    lines.append(f"  L{c}: {thr:.4f}")

output = "\n".join(lines)

with open("predict_knn_results.txt", "w", encoding="utf-8") as f:
    f.write(output)

print(f"정확도: {correct}/{n} ({correct/n*100:.1f}%)")
print(f"OOD 탐지: {n_ood}개  (탐지된 것 중 실제 오답: {n_ood_wrong}/{n_ood})")
print(f"in-distribution 오답: {n_in_wrong}개")
print("")
print("[ kNN threshold ]")
for c, thr in enumerate(detector.class_thresholds):
    print(f"  L{c}: {thr:.4f}")
print("")

# 상세 결과
print("# 정답  예측   신뢰도  kNN거리  kNN cls   OOD  cos  ok  메시지")
print("-" * 100)
for r in results:
    msg, _ = messages[r['i']-1]
    thr = detector._threshold_for(r['knn_class'])
    print(f"{r['i']:>2}  L{r['true']}  {LABEL_NAMES[r['pred']]}  "
          f"{r['dist']:.5f}  L{r['knn_class']}({thr:.4f})  "
          f"{'OOD' if r['ood'] else 'in ':3s}  {COSINE_OOD.get(r['i'],'?'):3s}  "
          f"{'O' if r['ok'] else 'X'}  {msg[:40]}")

print("\n-> predict_knn_results.txt 저장 완료")
