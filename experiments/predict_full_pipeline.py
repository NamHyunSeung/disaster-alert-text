"""
v22 전체 파이프라인: KNN-OOD 거부 → confidence threshold → v22 분류
  - KNN 거리 > p99 threshold  → OOD 거부 (비재난/신종)
  - confidence < CONF_THR     → Groq LLM fallback
  - 그 외                      → v22 직접 분류

실행:
  cd "프로젝트 루트"
  python 실험/predict_full_pipeline.py
의존:
  pip install groq
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

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    print("[경고] groq 미설치 → pip install groq")
    _GROQ_AVAILABLE = False

# ── 경로 (프로젝트 루트 기준) ─────────────────────────────────────────
MODEL_DIR  = "model_v22"
TOK_DIR    = "tokenizer_v22"
EMB_FILE   = "실험/knn_ood_v22.npz"
META_FILE  = "실험/knn_ood_v22_meta.pt"

# ── 파라미터 ──────────────────────────────────────────────────────────
MAX_LEN      = 96
CONF_THR     = 0.85
TEMPERATURE  = 1.5
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

LABEL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4']
class_thresholds = {0: 0.018056, 1: 0.015686, 2: 0.014583, 3: 0.007599, 4: 0.003537}

# ── Gemini 시스템 프롬프트 ────────────────────────────────────────────
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

# ── Groq 클라이언트 초기화 ────────────────────────────────────────────
_groq = Groq(api_key=GROQ_API_KEY) if _GROQ_AVAILABLE else None
_last_call_time = 0.0
RPM_INTERVAL = 2.5

def call_llm(text: str) -> int:
    """Groq LLM fallback. 성공 시 0~4, 실패 시 -1 반환."""
    global _last_call_time
    if not _GROQ_AVAILABLE or _groq is None:
        return -1
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

# ── 테스트 케이스 ─────────────────────────────────────────────────────
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

print(f"K={K} | confidence threshold={CONF_THR:.0%} | LLM={GROQ_MODEL}\n")

# ── 추론 루프 ─────────────────────────────────────────────────────────
lines = []
n_total  = len(messages)
ood_cnt  = 0
llm_cnt  = 0; llm_corr  = 0
v22_cnt  = 0; v22_corr  = 0

header = (f"{'#':>2}  {'정답':^3}  {'KNN거리':>9}  {'OOD':^5}  "
          f"{'Conf':>7}  {'판정':^11}  {'결과':^4}  메시지")
lines.append(header)
lines.append("─" * 125)

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

    thr    = class_thresholds.get(base_pred, 0.018056)
    is_ood = knn_score > thr

    if is_ood:
        판정 = "OOD거부"
        결과 = "-"
        ood_cnt += 1
    elif conf < CONF_THR:
        llm_pred = call_llm(msg)
        if llm_pred >= 0:
            판정 = f"LLM→{LABEL_NAMES[llm_pred]}"
            ok   = llm_pred == true_label
            결과 = "O" if ok else "X"
            llm_cnt += 1
            if ok:
                llm_corr += 1
        else:
            판정 = f"LLM실패→{LABEL_NAMES[base_pred]}"
            ok   = base_pred == true_label
            결과 = "O" if ok else "X"
            v22_cnt += 1
            if ok:
                v22_corr += 1
    else:
        판정 = LABEL_NAMES[base_pred]
        ok   = base_pred == true_label
        결과 = "O" if ok else "X"
        v22_cnt += 1
        if ok:
            v22_corr += 1

    ood_str = "OOD" if is_ood else "in "
    short   = msg[:50] + ("…" if len(msg) > 50 else "")
    lines.append(
        f"{i:>2}  L{true_label:^3}  {knn_score:.6f}  {ood_str:^5}  "
        f"{conf:6.1%}  {판정:^11}  {결과:^4}  {short}"
    )

lines.append("─" * 125)
total_cls  = llm_cnt + v22_cnt
total_corr = llm_corr + v22_corr
lines.append(f"[OOD 거부]   {ood_cnt}개")
lines.append(f"[LLM 분류]   {llm_cnt}개  →  정답 {llm_corr}/{max(llm_cnt,1)} ({llm_corr/max(llm_cnt,1)*100:.1f}%)")
lines.append(f"[v22 분류]   {v22_cnt}개  →  정답 {v22_corr}/{max(v22_cnt,1)} ({v22_corr/max(v22_cnt,1)*100:.1f}%)")
lines.append(f"[분류 합계]  {total_cls}개  →  정답 {total_corr}/{max(total_cls,1)} ({total_corr/max(total_cls,1)*100:.1f}%)")

result_path = "실험/predict_full_pipeline_results.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

for line in lines:
    print(line)
print(f"\n결과 저장: {result_path}")
