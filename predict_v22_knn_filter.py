"""
v22 + KNN-OOD 필터링 파이프라인.

1단계: KNN 거리로 OOD 판정 → 재난문자 아님이면 분류 거부
2단계: in-distribution으로 판정된 것만 모델로 L0~L4 분류

실행: python predict_v22_knn_filter.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '완성 모델', 'src'))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.neighbors import NearestNeighbors

MODEL_DIR  = "model_v22"
TOK_DIR    = "tokenizer_v22"
EMB_FILE   = "실험/knn_ood_v22.npz"
META_FILE  = "실험/knn_ood_v22_meta.pt"
MAX_LEN    = 96

LABEL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4']

# true_label: 0~4 = 재난문자 레이블
messages = [
    ("컴퓨터 해킹으로 전력망 마비. 병원 비상전원 가동 중. 인근 주민 대피 권고.", 4),   # L4
    ("해저 광케이블 절단으로 광역 통신 두절. 비상 연락망 사용 바람.", 2),               # L3→L2
    ("태양폭풍으로 GPS통신위성 오작동. 항공선박 운항 위험. 야외 활동 자제.", 3),        # L3
    ("반도체 공장 냉각수 유출. 독성물질 검출. 인근 2km 주민 실내 대피.", 4),            # L4
    ("해수면 급상승으로 해안 저지대 침수 진행 중. 고지대 이동 바람.", 4),               # L4
    ("가스관 노후로 인한 누출 감지. 불꽃 사용 삼가고 환기 권장.", 2),                   # L2
    ("열돔 현상 지속. 사흘째 40도 이상. 야외 노동자 노약자 건강 위협.", 2),             # L2
    ("지하철 전기 합선으로 연기 발생. 해당 노선 운행 중단. 승객 역사 밖으로.", 4),      # L3→L4
    ("강 상류 댐 균열 발견. 하류 주민 고지대 이동 권장.", 4),                           # L4
    ("핵발전소 냉각 계통 이상. 반경 10km 주민 예방 대피 진행 중.", 4),                  # L4
    ("대규모 산불 확산 중. 바람 방향 주의 요망. 연기 흡입 피해야.", 3),                 # L3
    ("오늘 밤부터 기온이 영하 20도까지 떨어질 예정. 수도관 관리 각별히 신경 쓰세요.", 3), # L1→L3
    ("신종 호흡기 바이러스 지역사회 전파 확인. 마스크 착용 손씻기 생활화.", 2),         # L1→L2
    ("바이러스 변종 출현. 기존 백신 효능 불명확. 외출 최소화 권고.", 2),                # L2
    ("황사 농도 매우 나쁨. 미세먼지 동반. 창문 닫고 외출 시 마스크 필수.", 3),          # L1→L3
    ("항 인근 해역 적조 발생. 어패류 채취 섭취 주의.", 1),                              # L1
    ("AI조류인플루엔자 의심 사례 발생. 가금류 접촉 자제.", 1),                          # L1
    ("도심 집중호우로 지하차도 침수. 우회 도로 이용 바람.", 3),                         # L3
    ("고압 전선 도로 낙하. 접근하지 마시고 신고 바람.", 4),                             # L3→L4
    ("식수원 오염 의심. 수돗물 음용 잠정 중단. 생수 배급 예정.", 2),                    # L3→L2
]

# ── 로드 ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device); model.eval()

print("KNN 인덱스 로드 중...")
data    = np.load(EMB_FILE)
embs_np = data['embs']
meta    = torch.load(META_FILE, weights_only=False)
K                = meta['k']
global_thr       = meta['global_threshold']
class_thresholds = meta['class_thresholds']

nn = NearestNeighbors(n_neighbors=K, algorithm='brute', metric='cosine', n_jobs=-1)
nn.fit(embs_np)

# p99 threshold로 오버라이드 (build_knn_ood_v22.py 출력값)
class_thresholds = {0: 0.018056, 1: 0.015686, 2: 0.014583, 3: 0.007599, 4: 0.003537}
global_thr = 0.018056
print(f"  K={K}, 인덱스={embs_np.shape}, threshold=p99\n")

# ── 예측 ──────────────────────────────────────────────────────────────
lines = []
n_total   = len(messages)
# 통계
ood_count   = 0   # 재난문자인데 OOD로 거부된 수
cls_correct = 0   # in 판정 후 분류까지 맞음
cls_total   = 0   # in 판정된 것 수
base_correct = 0  # 기본 예측 정확도 (비교용)

header = (f"{'#':>2}  {'정답':^3}  {'KNN거리':>9}  {'OOD?':^5}  "
          f"{'예측':^5}  {'기본OK':^4}  {'결과':^4}  메시지")
lines.append(header)
lines.append("─" * 115)

for i, (msg, true_label) in enumerate(messages, 1):
    enc = tokenizer(msg, truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out   = model(**enc, output_hidden_states=True)
        probs = F.softmax(out.logits, dim=-1)[0].cpu()
    base_pred = int(probs.argmax())

    cls_emb   = out.hidden_states[-1][:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
    dist, _   = nn.kneighbors(cls_emb.reshape(1, -1))
    knn_score = float(dist.mean())

    thr    = class_thresholds.get(base_pred, global_thr)
    is_ood = knn_score > thr

    ok_base = base_pred == true_label
    if ok_base:
        base_correct += 1

    if is_ood:
        pred_str   = "거부"
        result_str = "-"   # 분류 거부 = 평가 대상 아님
        ood_count += 1
    else:
        pred_str  = LABEL_NAMES[base_pred]
        cls_total += 1
        ok = base_pred == true_label
        result_str = "O" if ok else "X"
        if ok:
            cls_correct += 1

    ood_str  = "OOD" if is_ood else "in "
    base_str = "O" if ok_base else "X"
    short    = msg[:50] + ("…" if len(msg) > 50 else "")
    lines.append(
        f"{i:>2}  L{true_label:^3}  {knn_score:.6f}  {ood_str:^5}  "
        f"{pred_str:^5}  {base_str:^4}  {result_str:^4}  {short}"
    )

lines.append("─" * 115)
lines.append(f"[기본 예측]   전체 {n_total}개 중 정답: {base_correct}/{n_total} ({base_correct/n_total*100:.1f}%)")
lines.append(f"[OOD 필터]    거부: {ood_count}개  통과: {cls_total}개")
lines.append(f"[분류 성능]   통과 {cls_total}개 중 정답: {cls_correct}/{cls_total}"
             f" ({cls_correct/max(cls_total,1)*100:.1f}%)")

result_path = "predict_v22_knn_filter_p99_criteria_results.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

for line in lines:
    print(line)
print(f"\n결과 저장: {result_path}")
