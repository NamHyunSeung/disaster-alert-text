"""
v22 전체 파이프라인: KNN-OOD 거부 → confidence threshold → v22 분류
  - KNN 거리 > p99 threshold  → OOD 거부 (비재난/신종)
  - confidence < CONF_THR     → LLM fallback
  - 그 외                      → v22 직접 분류
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '완성 모델', 'src'))

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
CONF_THR   = 0.85   # confidence threshold

LABEL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4']

messages = [
    ("컴퓨터 해킹으로 전력망 마비. 병원 비상전원 가동 중. 인근 주민 대피 권고.", 4),
    ("해저 광케이블 절단으로 광역 통신 두절. 비상 연락망 사용 바람.", 2),
    ("태양폭풍으로 GPS통신위성 오작동. 항공선박 운항 위험. 야외 활동 자제.", 3),
    ("반도체 공장 냉각수 유출. 독성물질 검출. 인근 2km 주민 실내 대피.", 4),
    ("해수면 급상승으로 해안 저지대 침수 진행 중. 고지대 이동 바람.", 4),
    ("가스관 노후로 인한 누출 감지. 불꽃 사용 삼가고 환기 권장.", 2),
    ("열돔 현상 지속. 사흘째 40도 이상. 야외 노동자 노약자 건강 위협.", 2),
    ("지하철 전기 합선으로 연기 발생. 해당 노선 운행 중단. 승객 역사 밖으로.", 4),
    ("강 상류 댐 균열 발견. 하류 주민 고지대 이동 권장.", 4),
    ("핵발전소 냉각 계통 이상. 반경 10km 주민 예방 대피 진행 중.", 4),
    ("대규모 산불 확산 중. 바람 방향 주의 요망. 연기 흡입 피해야.", 3),
    ("오늘 밤부터 기온이 영하 20도까지 떨어질 예정. 수도관 관리 각별히 신경 쓰세요.", 3),
    ("신종 호흡기 바이러스 지역사회 전파 확인. 마스크 착용 손씻기 생활화.", 2),
    ("바이러스 변종 출현. 기존 백신 효능 불명확. 외출 최소화 권고.", 2),
    ("황사 농도 매우 나쁨. 미세먼지 동반. 창문 닫고 외출 시 마스크 필수.", 3),
    ("항 인근 해역 적조 발생. 어패류 채취 섭취 주의.", 1),
    ("AI조류인플루엔자 의심 사례 발생. 가금류 접촉 자제.", 1),
    ("도심 집중호우로 지하차도 침수. 우회 도로 이용 바람.", 3),
    ("고압 전선 도로 낙하. 접근하지 마시고 신고 바람.", 4),
    ("식수원 오염 의심. 수돗물 음용 잠정 중단. 생수 배급 예정.", 2),
]

# ── 로드 ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device); model.eval()

data    = np.load(EMB_FILE)
embs_np = data['embs']
meta    = torch.load(META_FILE, weights_only=False)
K = meta['k']

nn = NearestNeighbors(n_neighbors=K, algorithm='brute', metric='cosine', n_jobs=-1)
nn.fit(embs_np)

class_thresholds = {0: 0.018056, 1: 0.015686, 2: 0.014583, 3: 0.007599, 4: 0.003537}
global_thr = 0.018056

print(f"K={K}  KNN p99 threshold  confidence threshold={CONF_THR:.0%}\n")

# ── 예측 ──────────────────────────────────────────────────────────────
lines = []
n_total  = len(messages)
ood_cnt  = 0
llm_cnt  = 0
v22_cnt  = 0
v22_corr = 0

header = (f"{'#':>2}  {'정답':^3}  {'KNN거리':>9}  {'OOD':^5}  "
          f"{'Conf':>7}  {'판정':^7}  {'결과':^4}  메시지")
lines.append(header)
lines.append("─" * 120)

for i, (msg, true_label) in enumerate(messages, 1):
    enc = tokenizer(msg, truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out   = model(**enc, output_hidden_states=True)
        probs = F.softmax(out.logits, dim=-1)[0].cpu()

    base_pred = int(probs.argmax())
    conf      = float(probs.max())

    cls_emb   = out.hidden_states[-1][:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
    dist, _   = nn.kneighbors(cls_emb.reshape(1, -1))
    knn_score = float(dist.mean())

    thr    = class_thresholds.get(base_pred, global_thr)
    is_ood = knn_score > thr

    if is_ood:
        판정 = "OOD거부"
        결과 = "-"
        ood_cnt += 1
    elif conf < CONF_THR:
        판정 = "LLM↑"
        결과 = "-"
        llm_cnt += 1
    else:
        판정 = LABEL_NAMES[base_pred]
        v22_cnt += 1
        ok = base_pred == true_label
        결과 = "O" if ok else "X"
        if ok:
            v22_corr += 1

    ood_str = "OOD" if is_ood else "in "
    short   = msg[:50] + ("…" if len(msg) > 50 else "")
    lines.append(
        f"{i:>2}  L{true_label:^3}  {knn_score:.6f}  {ood_str:^5}  "
        f"{conf:6.1%}  {판정:^7}  {결과:^4}  {short}"
    )

lines.append("─" * 120)
lines.append(f"[OOD 거부]     {ood_cnt}개")
lines.append(f"[LLM fallback] {llm_cnt}개  (confidence < {CONF_THR:.0%})")
lines.append(f"[v22 분류]     {v22_cnt}개  →  정답 {v22_corr}/{v22_cnt}"
             f" ({v22_corr/max(v22_cnt,1)*100:.1f}%)")
lines.append(f"[전체 커버]    {ood_cnt+llm_cnt+v22_cnt}/{n_total}")

result_path = "실험/predict_full_pipeline_results.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

for line in lines:
    print(line)
print(f"\n결과 저장: {result_path}")
