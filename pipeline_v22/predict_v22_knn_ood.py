"""v22 + KNN-OOD 탐지 - 20개 메시지 분류"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '완성 모델', 'src'))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.neighbors import NearestNeighbors

MODEL_DIR  = "model"
TOK_DIR    = "tokenizer"
EMB_FILE   = "ood/knn_ood_v22.npz"
META_FILE  = "ood/knn_ood_v22_meta.pt"
MAX_LEN    = 96
SAFE_FLOOR = 4   # OOD이면 최소 L4로 상향

LABEL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4']

messages = [
    ("컴퓨터 해킹으로 전력망 마비. 병원 비상전원 가동 중. 인근 주민 대피 권고.", 4),
    ("해저 광케이블 절단으로 광역 통신 두절. 비상 연락망 사용 바람.", 3),
    ("태양폭풍으로 GPS통신위성 오작동. 항공선박 운항 위험. 야외 활동 자제.", 3),
    ("반도체 공장 냉각수 유출. 독성물질 검출. 인근 2km 주민 실내 대피.", 4),
    ("해수면 급상승으로 해안 저지대 침수 진행 중. 고지대 이동 바람.", 4),
    ("가스관 노후로 인한 누출 감지. 불꽃 사용 삼가고 환기 권장.", 2),
    ("열돔 현상 지속. 사흘째 40도 이상. 야외 노동자 노약자 건강 위협.", 2),
    ("지하철 전기 합선으로 연기 발생. 해당 노선 운행 중단. 승객 역사 밖으로.", 3),
    ("강 상류 댐 균열 발견. 하류 주민 고지대 이동 권장.", 4),
    ("핵발전소 냉각 계통 이상. 반경 10km 주민 예방 대피 진행 중.", 4),
    ("대규모 산불 확산 중. 바람 방향 주의 요망. 연기 흡입 피해야.", 3),
    ("오늘 밤부터 기온이 영하 20도까지 떨어질 예정. 수도관 관리 각별히 신경 쓰세요.", 1),
    ("신종 호흡기 바이러스 지역사회 전파 확인. 마스크 착용 손씻기 생활화.", 1),
    ("바이러스 변종 출현. 기존 백신 효능 불명확. 외출 최소화 권고.", 2),
    ("황사 농도 매우 나쁨. 미세먼지 동반. 창문 닫고 외출 시 마스크 필수.", 1),
    ("항 인근 해역 적조 발생. 어패류 채취 섭취 주의.", 1),
    ("AI조류인플루엔자 의심 사례 발생. 가금류 접촉 자제.", 1),
    ("도심 집중호우로 지하차도 침수. 우회 도로 이용 바람.", 3),
    ("고압 전선 도로 낙하. 접근하지 마시고 신고 바람.", 3),
    ("식수원 오염 의심. 수돗물 음용 잠정 중단. 생수 배급 예정.", 3),
]

# ── 로드 ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device); model.eval()

print("KNN 인덱스 로드 중...")
data = np.load(EMB_FILE)
embs_np = data['embs']         # (N, 768) float32
meta    = torch.load(META_FILE, weights_only=False)
K                = meta['k']
global_thr       = meta['global_threshold']
class_thresholds = meta['class_thresholds']

nn = NearestNeighbors(n_neighbors=K, algorithm='brute', metric='cosine', n_jobs=-1)
nn.fit(embs_np)
print(f"  K={K}, 인덱스 크기={embs_np.shape}, 글로벌 thr={global_thr:.6f}")

# ── 예측 ──────────────────────────────────────────────────────────────
lines = []
correct_base = 0
correct_ood  = 0

header = f"{'#':>2}  {'정답':^3}  {'기본예측':^8}  {'KNN보정':^5}  {'KNN거리':>9}  {'OOD?':^5}  {'기본OK':^4}  {'보정OK':^4}  메시지"
lines.append(header)
lines.append("─" * 125)

for i, (msg, true_label) in enumerate(messages, 1):
    enc = tokenizer(msg, truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out   = model(**enc, output_hidden_states=True)
        probs = F.softmax(out.logits, dim=-1)[0].cpu()
    base_pred = int(probs.argmax())

    # CLS 임베딩 → KNN 거리
    cls_emb = out.hidden_states[-1][:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
    dist, _ = nn.kneighbors(cls_emb.reshape(1, -1))
    knn_score = float(dist.mean())

    # OOD 판정: 클래스별 threshold (없으면 글로벌)
    thr = class_thresholds.get(base_pred, global_thr)
    is_ood = knn_score > thr

    final_pred = max(base_pred, SAFE_FLOOR) if is_ood else base_pred

    ok_base = "O" if base_pred  == true_label else "X"
    ok_ood  = "O" if final_pred == true_label else "X"
    if base_pred  == true_label: correct_base += 1
    if final_pred == true_label: correct_ood  += 1

    bumped  = "↑" if (is_ood and final_pred > base_pred) else " "
    ood_str = "OOD" if is_ood else "in"
    short   = msg[:45] + ("…" if len(msg) > 45 else "")
    lines.append(
        f"{i:>2}  L{true_label:^3}  {LABEL_NAMES[base_pred]:^4}({probs[base_pred]*100:4.1f}%)  "
        f"L{final_pred}{bumped:^5}  {knn_score:.6f}  {ood_str:^5}  {ok_base:^4}  {ok_ood:^4}  {short}"
    )

n = len(messages)
lines.append("─" * 125)
lines.append(f"기본 예측 정확도 : {correct_base}/{n} ({correct_base/n*100:.1f}%)")
lines.append(f"KNN OOD 보정 정확도 : {correct_ood}/{n} ({correct_ood/n*100:.1f}%)")
lines.append(f"변화                 : {correct_ood - correct_base:+d}개")

result_path = "predict_v22_knn_ood_results.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

for line in lines:
    print(line)
print(f"\n결과 저장: {result_path}")
