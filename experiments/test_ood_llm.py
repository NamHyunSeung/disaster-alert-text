"""
OOD → LLM 라우팅 테스트
파이프라인 변형: KNN OOD 탐지 → LLM fallback (거부 대신)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '완성 모델', 'src'))

import re
import time
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.neighbors import NearestNeighbors
from groq import Groq

MODEL_DIR  = "model_v22"
TOK_DIR    = "tokenizer_v22"
EMB_FILE   = "실험/knn_ood_v22.npz"
META_FILE  = "실험/knn_ood_v22_meta.pt"
MAX_LEN     = 96
CONF_THR    = 0.85
TEMPERATURE = 1.5
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
LABEL_NAMES    = ['L0', 'L1', 'L2', 'L3', 'L4']
class_thresholds = {0: 0.018056, 1: 0.015686, 2: 0.014583, 3: 0.007599, 4: 0.003537}

_SYS = """재난문자를 긴급도 L0~L4로 분류하는 전문가입니다.
키워드가 아닌 상황 심각도·행동 즉시성·생명위협 여부로 판단합니다.

【L0】행동 불필요
  재난 해제·훈련 종료·경보 해제·행사 안내

【L1】주의·정보 전달 (행동 변화 불필요)
  기상 예보, 발생 알림, 주의 권고 없는 정보

【L2】사회 기능 마비·행정 조치 (직접 생명위협 없음)
  기능이 제한·마비되거나 행정 조치가 필요하지만 생명 위협 없음
  예: 사이버공격으로 금융·전력·통신 인프라 마비 + 대안 사용 권고
  예: 수돗물 제한급수·음용 금지, 집합금지·시설폐쇄·방역협조
  예: 바이러스·전염병·해충 매개 감염 확산으로 야외 접근 금지 또는 활동 자제
  ※ 바이러스·전염병·진드기·해충 관련 접근 금지·활동 자제는 항상 L2
  판단 힌트: "현금 사용 권고", "생수 배급", "서비스 중단", "제한급수", "바이러스 확산", "진드기·해충 매개"

【L3】재난 진행 중 + 예방적 자제 권고 (즉각 이동 지시 없음)
  위험이 존재하지만 즉각 이동보다는 자제·주의·대기 수준
  예: 정전으로 신호등 불통 → 이동 자제, 원인불명 사고 현장 → 외부 대기
  예: 태양폭풍·통신 두절 → 이동 자제, 야외 활동 자제, 창문 닫기
  판단 힌트: "이동 자제", "야외 자제", "외부 대기", "접근 금지"(비감염성·물리적 위험)

【L4】즉각 대피·이동 지시 (생명 직접 위협)
  지금 당장 이동·대피하지 않으면 사망 가능한 상황
  예: 쓰나미·해수면 급상승 → 즉시 고지대 이동
  예: 댐 붕괴·가스관 폭발·방사성 물질 누출 → 즉시 대피
  예: 기온 48도 등 생명 직결 극한 환경 → 외출 금지
  판단 힌트: "즉시", "지금 즉시", "즉각", "대피소로", "고지대로", "실내 대피"
  특수 규칙: 방사성·핵·폭발·독성물질 누출은 "접근 금지"만 있어도 L4

【경계 구분 규칙】
  L2 vs L3: L2=기능 마비(생명위협 없음), L3=위험 진행 중 + 신체 자제 권고
  L3 vs L4: L3=자제·주의 권고, L4="즉시/지금 당장" 이동·대피 명령 또는 생명직결 위험 물질

반드시 JSON 형식으로만 답변: {"label": "L2"}"""

_groq = Groq(api_key=GROQ_API_KEY)

_last_call_time = 0.0
RPM_INTERVAL = 2.5  # Groq 30 RPM → 2.5초 간격

def call_llm(text: str) -> int:
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < RPM_INTERVAL:
        time.sleep(RPM_INTERVAL - elapsed)
    try:
        resp = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYS},
                {"role": "user", "content": f"재난문자: {text}"},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        _last_call_time = time.time()
        content = resp.choices[0].message.content or ""
        m = re.search(r'L([0-4])', content)
        if m:
            return int(m.group(1))
    except Exception as e:
        _last_call_time = time.time()
        print(f"  [Groq 오류] {e}")
    return -1

# ── 신종 재난 테스트 케이스 (OOD 유발 목적) ──────────────────────────
messages = [
    # 사이버·디지털 재난
    ("전국 금융망 사이버 공격. ATM 및 인터넷뱅킹 마비. 현금 사용 권고.", 2),
    ("국가 전력망 해킹으로 수도권 정전 지속. 신호등 불통. 이동 자제 바람.", 3),
    # 기후·환경 신종
    ("서울 도심 기온 48도 기록. 아스팔트 변형·화재 위험. 외출 금지.", 4),
    ("극한 가뭄으로 수돗물 공급 제한. 하루 2시간 급수. 저장 바람.", 2),
    ("동해안 해수면 3m 급상승 예측. 해안 저지대 주민 즉시 고지대 이동.", 4),
    # 생물·의료 신종
    ("원인불명 집단 실신 사례 다수 발생. 해당 건물 외부 대기 바람.", 3),
    ("신종 진드기 매개 바이러스 확산. 야외 풀숲 접근 금지.", 2),
    # 우주·지구물리
    ("강력 태양폭풍 도달. 항공기 운항 위험, 통신 두절 가능. 이동 자제.", 3),
    ("동해 해저 지진(규모 6.8). 동해안 지역 쓰나미 경보 발령. 즉시 고지대 이동.", 4),
    # 인프라 복합
    ("지하 고압 가스관 폭발. 반경 500m 건물 유리 파손. 즉시 실내 대피.", 4),
    ("댐 붕괴 위험. 하류 5km 주민 지금 즉시 대피소로 이동하세요.", 4),
    ("방사성 폐기물 운반 차량 전복. 인근 1km 통제. 접근 금지.", 4),
]

# ── 모델 로드 ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device); model.eval()

data    = np.load(EMB_FILE)
embs_np = data['embs']
meta    = torch.load(META_FILE, weights_only=False)
K       = meta['k']

nn_idx = NearestNeighbors(n_neighbors=K, algorithm='brute', metric='cosine', n_jobs=-1)
nn_idx.fit(embs_np)

print(f"K={K} | conf threshold={CONF_THR:.0%} | OOD→LLM 활성화 ({GROQ_MODEL})\n")

lines = []
ood_llm_cnt = 0; ood_llm_corr = 0
conf_llm_cnt = 0; conf_llm_corr = 0
v22_cnt = 0; v22_corr = 0

header = (f"{'#':>2}  {'정답':^3}  {'KNN거리':>9}  {'OOD':^5}  "
          f"{'Conf':>7}  {'경로':^14}  {'결과':^4}  메시지")
lines.append(header)
lines.append("─" * 130)

for i, (msg, true_label) in enumerate(messages, 1):
    enc = tokenizer(msg, truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out   = model(**enc, output_hidden_states=True)
        probs = F.softmax(out.logits / TEMPERATURE, dim=-1)[0].cpu()

    base_pred = int(probs.argmax())
    conf      = float(probs.max())
    cls_emb   = out.hidden_states[-1][:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
    dist, _   = nn_idx.kneighbors(cls_emb.reshape(1, -1))
    knn_score = float(dist.mean())
    thr       = class_thresholds.get(base_pred, 0.018056)
    is_ood    = knn_score > thr

    if is_ood:
        llm_pred = call_llm(msg)
        if llm_pred >= 0:
            경로 = f"OOD→LLM:{LABEL_NAMES[llm_pred]}"
            ok   = llm_pred == true_label
            결과 = "O" if ok else "X"
            ood_llm_cnt += 1
            if ok:
                ood_llm_corr += 1
        else:
            경로 = "OOD→LLM실패→거부"
            결과 = "-"
    elif conf < CONF_THR:
        llm_pred = call_llm(msg)
        if llm_pred >= 0:
            경로 = f"저신뢰→LLM:{LABEL_NAMES[llm_pred]}"
            ok   = llm_pred == true_label
            결과 = "O" if ok else "X"
            conf_llm_cnt += 1
            if ok:
                conf_llm_corr += 1
        else:
            경로 = f"LLM실패→{LABEL_NAMES[base_pred]}"
            ok   = base_pred == true_label
            결과 = "O" if ok else "X"
            v22_cnt += 1
            if ok:
                v22_corr += 1
    else:
        경로 = f"v22:{LABEL_NAMES[base_pred]}"
        ok   = base_pred == true_label
        결과 = "O" if ok else "X"
        v22_cnt += 1
        if ok:
            v22_corr += 1

    ood_str = "OOD" if is_ood else "in "
    short   = msg[:50] + ("…" if len(msg) > 50 else "")
    lines.append(
        f"{i:>2}  L{true_label:^3}  {knn_score:.6f}  {ood_str:^5}  "
        f"{conf:6.1%}  {경로:^14}  {결과:^4}  {short}"
    )

lines.append("─" * 130)
total_llm  = ood_llm_cnt + conf_llm_cnt
total_corr = ood_llm_corr + conf_llm_corr + v22_corr
total_cls  = total_llm + v22_cnt
lines.append(f"[OOD→LLM]    {ood_llm_cnt}개  →  정답 {ood_llm_corr}/{max(ood_llm_cnt,1)} ({ood_llm_corr/max(ood_llm_cnt,1)*100:.1f}%)")
lines.append(f"[저신뢰→LLM] {conf_llm_cnt}개  →  정답 {conf_llm_corr}/{max(conf_llm_cnt,1)} ({conf_llm_corr/max(conf_llm_cnt,1)*100:.1f}%)")
lines.append(f"[v22 직접]   {v22_cnt}개  →  정답 {v22_corr}/{max(v22_cnt,1)} ({v22_corr/max(v22_cnt,1)*100:.1f}%)")
lines.append(f"[전체]       {total_cls}개  →  정답 {total_corr}/{max(total_cls,1)} ({total_corr/max(total_cls,1)*100:.1f}%)")

result_path = "실험/test_ood_llm_results.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

for line in lines:
    print(line)
print(f"\n결과 저장: {result_path}")
