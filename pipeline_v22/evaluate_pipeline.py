"""
전체 파이프라인 성능 평가 (원본 테스트셋)
  KNN-OOD 탐지 + Temperature Scaling (T=1.5) + LLM fallback
  LLM 순서: 1순위 Gemini 2.5 Flash → 2순위 Groq → 3순위 Cerebras (모두 할당량 소진 시 순차 전환)
  LLM 결과 캐시: 실험/llm_cache.json (재실행 시 API 호출 절약)

실행:
  python 실험/evaluate_pipeline.py --skip_llm    # LLM 없이 라우팅 분포만 확인
  python 실험/evaluate_pipeline.py               # 전체 파이프라인 (LLM 포함)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '완성 모델', 'src'))

import re, time, argparse, json
import requests
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

try:
    from google import genai as _genai_module
    from google.genai import types as _genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    print("[경고] google-genai 미설치 → pip install google-genai")

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    print("[경고] groq 미설치 → pip install groq")

from dataset_v2 import load_and_split_v2

# ── 경로 ──────────────────────────────────────────────────────────────
MODEL_DIR  = "model"
TOK_DIR    = "tokenizer"
EMB_FILE   = "ood/knn_ood_v22.npz"
META_FILE  = "ood/knn_ood_v22_meta.pt"
DATA_PATH  = "../중요파일/data/raw/재난문자_레이블링결과_dedup_v7.xlsx"

# ── 파라미터 ──────────────────────────────────────────────────────────
MAX_LEN      = 96
BATCH_SIZE   = 64
CONF_THR     = 0.70
TEMPERATURE  = 1.5

# Gemini 2.5 Flash (1순위) — Google AI Studio: aistudio.google.com
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-2.5-flash"
GEMINI_INTERVAL = 4.0   # 15 RPM → 4초 간격

# Groq (2순위 fallback)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
RPM_INTERVAL = 2.5

# Cerebras (3순위 fallback) — cloud.cerebras.ai, OpenAI 호환 REST
CEREBRAS_API_KEY    = os.environ.get("CEREBRAS_API_KEY", "")
CEREBRAS_MODEL      = "gpt-oss-120b"
CEREBRAS_URL        = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_INTERVAL   = 2.0

# OpenRouter (임시 1순위 — Gemini/Groq/Cerebras 모두 소진/장애 상태라 우선 사용) — openrouter.ai, OpenAI 호환 REST
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = "openai/gpt-oss-120b:free"
OPENROUTER_URL      = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_INTERVAL = 3.0

CACHE_FILE       = "llm_cache.json"
BATCH_SIZE_LLM   = 20        # 배치 분류 크기 (토큰 절약)

LABEL_NAMES      = ['L0', 'L1', 'L2', 'L3', 'L4']
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

【주의: 같은 재난 소재라도 "단계어"에 따라 L0~L4 전 구간에 분포】
  산불·산사태·폭염·한파·호우·강풍·감염병 등은 표현이 비슷해도 경보 단계에 따라 등급이 갈립니다.
  문장에 등장하는 단계어를 우선 단서로 삼으세요.
  예: "관심"/"해제"/단순 통계·신청·동선 안내 → L0
  예: "주의"·"주의보"·일반 예방 권고("~하세요","~바랍니다") → L1~L2
  예: "경보"·"위기경보 경계 단계" + 구체적 금지·자제 행동요령 → L2~L3
  예: "위기경보 심각 단계" 또는 "지금 즉시 대피"형 명령 → L3~L4
  ※ 감염병(코로나19 등)도 예외 아님: 확진자 수·검사 안내처럼 평범해 보여도
     "심각 단계", "이동·모임 강력 자제" 등 고강도 표현이 동반되면 L2 이상(드물게 L4)일 수 있음

【주의: 위협 표현 없는 행정·정보성 공지는 기본 L0】
  순환정전 시행 지역 안내, 확진자 이동경로(동선) 안내, 경보·통제 해제,
  운행 중단/재개, 지원금·신청 안내, 실종자 수배 등은
  특별한 위협·행동지시 표현이 없으면 L0으로 분류

【경계 구분 규칙】
  L2 vs L3: L2=기능 마비(생명위협 없음), L3=위험 진행 중 + 신체 자제 권고
  L3 vs L4: L3=자제·주의 권고, L4="즉시/지금 당장" 이동·대피 명령 또는 생명직결 위험 물질

반드시 JSON 형식으로만 답변: {"label": "L2"}"""

_gemini_model     = None
_gemini_exhausted = False
_gemini_last_call = 0.0

_groq            = None
_groq_last_call  = 0.0
_groq_exhausted  = False

_cerebras_last_call = 0.0
_cerebras_exhausted = False

_openrouter_last_call = 0.0
_openrouter_exhausted = False

_llm_cache: dict = {}


def load_cache() -> int:
    global _llm_cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _llm_cache = json.load(f)
        print(f"[캐시] {len(_llm_cache):,}건 로드 ({CACHE_FILE})")
        return len(_llm_cache)
    return 0


def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_llm_cache, f, ensure_ascii=False)
    print(f"[캐시] {len(_llm_cache):,}건 저장 ({CACHE_FILE})")


def init_llm():
    global _gemini_model, _groq
    if _GEMINI_AVAILABLE and GEMINI_API_KEY:
        try:
            _gemini_model = _genai_module.Client(api_key=GEMINI_API_KEY)
            print(f"[LLM] Gemini {GEMINI_MODEL} 초기화 완료 (1순위)")
        except Exception as e:
            print(f"[경고] Gemini 초기화 실패: {e}")
    if _GROQ_AVAILABLE and GROQ_API_KEY:
        _groq = Groq(api_key=GROQ_API_KEY)
        print(f"[LLM] Groq {GROQ_MODEL} 초기화 완료 (2순위 fallback)")
    if CEREBRAS_API_KEY:
        print(f"[LLM] Cerebras {CEREBRAS_MODEL} 준비 완료 (3순위 fallback)")
    if OPENROUTER_API_KEY:
        print(f"[LLM] OpenRouter {OPENROUTER_MODEL} 준비 완료 (임시 1순위)")


def _build_batch_user_msg(texts: list) -> str:
    n = len(texts)
    items = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    return (
        f"다음 재난문자 {n}건을 각각 분류하세요.\n\n"
        f"{items}\n\n"
        "반드시 JSON 배열로만 답변 (다른 텍스트 없이):\n"
        f'[{{"id":1,"label":"L?"}},{{"id":2,"label":"L?"}},...,{{"id":{n},"label":"L?"}}]'
    )


def _parse_batch_response(content: str, n: int) -> list:
    result = [-1] * n
    try:
        m = re.search(r'\[.+\]', content, re.DOTALL)
        if m:
            for item in json.loads(m.group()):
                id_ = int(item.get('id', 0)) - 1
                lbl = re.search(r'[Ll]([0-4])', str(item.get('label', '')))
                if lbl and 0 <= id_ < n:
                    result[id_] = int(lbl.group(1))
            return result
    except Exception:
        pass
    for m2 in re.finditer(r'"id"\s*:\s*(\d+)[^}]*?"label"\s*:\s*"[Ll]([0-4])"', content, re.DOTALL):
        id_ = int(m2.group(1)) - 1
        if 0 <= id_ < n:
            result[id_] = int(m2.group(2))
    return result


def call_gemini_batch(texts: list) -> list:
    global _gemini_last_call, _gemini_exhausted
    n = len(texts)
    elapsed = time.time() - _gemini_last_call
    if elapsed < GEMINI_INTERVAL:
        time.sleep(GEMINI_INTERVAL - elapsed)
    try:
        resp = _gemini_model.models.generate_content(
            model=GEMINI_MODEL,
            contents=_build_batch_user_msg(texts),
            config=_genai_types.GenerateContentConfig(
                system_instruction=_SYS,
                temperature=0.0,
                max_output_tokens=n * 25 + 50,
            ),
        )
        _gemini_last_call = time.time()
        return _parse_batch_response(resp.text or "", n)
    except Exception as e:
        _gemini_last_call = time.time()
        err = str(e).lower()
        if "429" in err or "quota" in err or "resource_exhausted" in err or "resource exhausted" in err:
            print(f"\n  [Gemini 할당량 소진] Groq로 전환합니다.")
            _gemini_exhausted = True
        else:
            print(f"  [Gemini 오류] {e}")
        return [-1] * n


def call_groq_batch(texts: list) -> list:
    global _groq_last_call, _groq_exhausted
    if _groq is None:
        return [-1] * len(texts)
    n = len(texts)
    elapsed = time.time() - _groq_last_call
    if elapsed < RPM_INTERVAL:
        time.sleep(RPM_INTERVAL - elapsed)
    try:
        resp = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYS},
                {"role": "user",   "content": _build_batch_user_msg(texts)},
            ],
            temperature=0.0,
            max_tokens=n * 25 + 50,
        )
        _groq_last_call = time.time()
        return _parse_batch_response(resp.choices[0].message.content or "", n)
    except Exception as e:
        _groq_last_call = time.time()
        err = str(e).lower()
        if "429" in err or "quota" in err or "rate_limit" in err:
            print(f"  [Groq 할당량 소진] 이후 배치 Groq 호출 건너뜁니다.")
            _groq_exhausted = True
        else:
            print(f"  [Groq 오류] {e}")
        return [-1] * n


def call_cerebras_batch(texts: list) -> list:
    global _cerebras_last_call, _cerebras_exhausted
    if not CEREBRAS_API_KEY:
        return [-1] * len(texts)
    n = len(texts)
    elapsed = time.time() - _cerebras_last_call
    if elapsed < CEREBRAS_INTERVAL:
        time.sleep(CEREBRAS_INTERVAL - elapsed)
    try:
        resp = requests.post(
            CEREBRAS_URL,
            headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": CEREBRAS_MODEL,
                "messages": [
                    {"role": "system", "content": _SYS},
                    {"role": "user",   "content": _build_batch_user_msg(texts)},
                ],
                "temperature": 0.0,
                # gpt-oss-120b는 추론 모델 — reasoning에 토큰을 소모하므로 effort를 낮추고 여유 토큰을 둠
                "reasoning_effort": "low",
                "max_tokens": n * 30 + 1000,
            },
            timeout=60,
        )
        _cerebras_last_call = time.time()
        if resp.status_code == 429:
            print(f"  [Cerebras 할당량 소진] 이후 배치 Cerebras 호출 건너뜁니다.")
            _cerebras_exhausted = True
            return [-1] * n
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""
        return _parse_batch_response(content, n)
    except Exception as e:
        _cerebras_last_call = time.time()
        err = str(e).lower()
        if "429" in err or "quota" in err or "rate_limit" in err:
            print(f"  [Cerebras 할당량 소진] 이후 배치 Cerebras 호출 건너뜁니다.")
            _cerebras_exhausted = True
        else:
            print(f"  [Cerebras 오류] {e}")
        return [-1] * n


def call_openrouter_batch(texts: list) -> list:
    global _openrouter_last_call, _openrouter_exhausted
    if not OPENROUTER_API_KEY:
        return [-1] * len(texts)
    n = len(texts)
    elapsed = time.time() - _openrouter_last_call
    if elapsed < OPENROUTER_INTERVAL:
        time.sleep(OPENROUTER_INTERVAL - elapsed)
    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://github.com/disaster-classifier",
                     "X-Title": "disaster-msg-classifier"},
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": _SYS},
                    {"role": "user",   "content": _build_batch_user_msg(texts)},
                ],
                "temperature": 0.0,
                # gpt-oss-120b는 추론 모델 — reasoning에 토큰을 소모하므로 effort를 낮추고 여유 토큰을 둠
                "reasoning_effort": "low",
                "max_tokens": n * 30 + 500,
            },
            timeout=60,
        )
        _openrouter_last_call = time.time()
        if resp.status_code == 429:
            print(f"  [OpenRouter 할당량 소진] 이후 배치 OpenRouter 호출 건너뜁니다.")
            _openrouter_exhausted = True
            return [-1] * n
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""
        return _parse_batch_response(content, n)
    except Exception as e:
        _openrouter_last_call = time.time()
        err = str(e).lower()
        if "429" in err or "quota" in err or "rate_limit" in err or "rate-limit" in err:
            print(f"  [OpenRouter 할당량 소진] 이후 배치 OpenRouter 호출 건너뜁니다.")
            _openrouter_exhausted = True
        else:
            print(f"  [OpenRouter 오류] {e}")
        return [-1] * n


def call_llm_batch(texts: list) -> list:
    n = len(texts)
    result = [-1] * n

    # 임시 1순위: Gemini/Groq/Cerebras가 모두 소진/장애 상태이므로 OpenRouter를 먼저 시도
    if OPENROUTER_API_KEY and not _openrouter_exhausted:
        result = call_openrouter_batch(texts)

    failed = [i for i, r in enumerate(result) if r == -1]
    if failed and _gemini_model is not None and not _gemini_exhausted:
        gem_res = call_gemini_batch([texts[i] for i in failed])
        for j, i in enumerate(failed):
            if gem_res[j] >= 0:
                result[i] = gem_res[j]

    failed = [i for i, r in enumerate(result) if r == -1]
    if failed and not _groq_exhausted:
        groq_res = call_groq_batch([texts[i] for i in failed])
        for j, i in enumerate(failed):
            if groq_res[j] >= 0:
                result[i] = groq_res[j]

    failed = [i for i, r in enumerate(result) if r == -1]
    if failed and CEREBRAS_API_KEY and not _cerebras_exhausted:
        cb_res = call_cerebras_batch([texts[i] for i in failed])
        for j, i in enumerate(failed):
            if cb_res[j] >= 0:
                result[i] = cb_res[j]

    return result


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.enc = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )

    def __len__(self):
        return len(self.enc['input_ids'])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_llm', action='store_true',
                        help='LLM 호출 없이 라우팅 분포 확인 (빠른 모드)')
    parser.add_argument('--max_calls', type=int, default=0,
                        help='이번 실행에서 수행할 신규 API 호출 건수 제한 (0=무제한, 끊어서 실행할 때 사용)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── 데이터 로드 ─────────────────────────────────────────────────────
    print(f"데이터 로드: {DATA_PATH}")
    _, _, test_df = load_and_split_v2(DATA_PATH, seed=42)
    texts   = test_df["text"].astype(str).tolist()
    labels  = np.array(test_df["label"].tolist())
    n_total = len(texts)
    print(f"테스트 샘플: {n_total:,}건")

    # ── 모델 로드 ─────────────────────────────────────────────────────
    print("모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device); model.eval()

    # ── KNN 로드 ──────────────────────────────────────────────────────
    data    = np.load(EMB_FILE)
    embs_np = data['embs']
    meta    = torch.load(META_FILE, weights_only=False)
    K       = meta['k']
    nn_idx  = NearestNeighbors(n_neighbors=K, algorithm='brute', metric='cosine', n_jobs=-1)
    nn_idx.fit(embs_np)
    print(f"KNN 로드 완료 (K={K}, 학습 임베딩 {len(embs_np):,}개)")

    # ── 배치 추론 (hidden states 포함) ──────────────────────────────────
    dataset = TextDataset(texts, tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs_list = []
    all_embs_list  = []

    print("\n배치 추론 중...")
    with torch.no_grad():
        for batch in tqdm(loader, ncols=90):
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch, output_hidden_states=True)
            probs = F.softmax(out.logits / TEMPERATURE, dim=-1).cpu()
            cls_embs = out.hidden_states[-1][:, 0, :].cpu().numpy().astype(np.float32)
            all_probs_list.append(probs)
            all_embs_list.append(cls_embs)

    all_probs  = torch.cat(all_probs_list, dim=0)
    all_preds  = all_probs.argmax(dim=-1).numpy()
    all_confs  = all_probs.max(dim=-1).values.numpy()
    all_embs   = np.vstack(all_embs_list)

    # ── 배치 KNN 거리 계산 ─────────────────────────────────────────────
    print("KNN 거리 계산 중...")
    dists, _   = nn_idx.kneighbors(all_embs)
    knn_scores = dists.mean(axis=1)

    # ── 라우팅 결정 ────────────────────────────────────────────────────
    is_ood_arr = np.array([
        knn_scores[i] > class_thresholds.get(int(all_preds[i]), 0.018056)
        for i in range(n_total)
    ])
    low_conf_arr = (~is_ood_arr) & (all_confs < CONF_THR)
    need_llm     = is_ood_arr | low_conf_arr
    v22_direct   = ~need_llm

    n_v22   = int(v22_direct.sum())
    n_ood   = int(is_ood_arr.sum())
    n_lconf = int(low_conf_arr.sum())
    n_llm   = int(need_llm.sum())

    print(f"\n[라우팅 분포]  (T={TEMPERATURE}, conf_thr={CONF_THR:.0%})")
    print(f"  v22 직접      : {n_v22:>6,}건  ({n_v22/n_total*100:5.1f}%)")
    print(f"  저신뢰 → LLM  : {n_lconf:>6,}건  ({n_lconf/n_total*100:5.1f}%)")
    print(f"  OOD   → LLM   : {n_ood:>6,}건  ({n_ood/n_total*100:5.1f}%)")
    print(f"  LLM 총 대상   : {n_llm:>6,}건  ({n_llm/n_total*100:5.1f}%)")

    # ── v22 단독 성능 (참고용) ─────────────────────────────────────────
    base_acc = accuracy_score(labels, all_preds) * 100
    base_f1  = f1_score(labels, all_preds, average='macro', zero_division=0) * 100
    print(f"\n[v22 단독] Acc={base_acc:.2f}%  MacroF1={base_f1:.2f}%  (T={TEMPERATURE} 적용)")

    if args.skip_llm:
        print("\n[--skip_llm] LLM 호출 없이 종료합니다.")
        n_batches_est = (n_llm + BATCH_SIZE_LLM - 1) // BATCH_SIZE_LLM
        print(f"LLM 호출 시 예상 소요: ~{n_batches_est * GEMINI_INTERVAL:.0f}초 ({n_batches_est}배치, Gemini 기준)")
        return

    # ── LLM 초기화 및 호출 ────────────────────────────────────────────
    load_cache()
    init_llm()

    llm_indices      = np.where(need_llm)[0].tolist()
    cached_indices   = [i for i in llm_indices if texts[i] in _llm_cache]
    uncached_indices = [i for i in llm_indices if texts[i] not in _llm_cache]
    cache_hits       = len(cached_indices)
    api_needed       = len(uncached_indices)
    n_batches        = (api_needed + BATCH_SIZE_LLM - 1) // BATCH_SIZE_LLM if api_needed > 0 else 0

    print(f"\nLLM 대상 {n_llm:,}건 | 캐시 히트 {cache_hits:,}건 | API 호출 필요 {api_needed:,}건")
    if api_needed > 0:
        print(f"  배치 {BATCH_SIZE_LLM}건 × {n_batches}배치 → 예상 소요: ~{n_batches * GEMINI_INTERVAL:.0f}초 (Gemini 기준)")

    final_preds = all_preds.copy().tolist()
    llm_fail    = 0

    for idx in cached_indices:
        final_preds[idx] = _llm_cache[texts[idx]]

    if args.max_calls > 0 and api_needed > args.max_calls:
        uncached_indices = uncached_indices[:args.max_calls]
        print(f"  [--max_calls] 이번 실행은 {args.max_calls}건만 호출 (남은 {api_needed - args.max_calls:,}건은 다음 실행에서 캐시 이어받기)")

    batches = [uncached_indices[i:i + BATCH_SIZE_LLM]
               for i in range(0, len(uncached_indices), BATCH_SIZE_LLM)]

    n_calls_done = 0
    for b_idx, batch_indices in enumerate(tqdm(batches, ncols=90, desc="배치 LLM")):
        batch_texts = [texts[i] for i in batch_indices]
        preds = call_llm_batch(batch_texts)
        for idx, pred in zip(batch_indices, preds):
            if pred >= 0:
                final_preds[idx] = pred
                _llm_cache[texts[idx]] = pred
            else:
                llm_fail += 1
        n_calls_done += len(batch_indices)
        if (b_idx + 1) % 5 == 0:
            save_cache()

    save_cache()
    if args.max_calls > 0:
        remaining = api_needed - n_calls_done
        print(f"\n[이번 실행 요약] 신규 호출 {n_calls_done:,}건 처리, 남은 미처리 {max(remaining, 0):,}건")
        if remaining > 0:
            print(f"  → 동일 명령에 --max_calls 옵션으로 다시 실행하면 캐시를 이어받아 계속 진행합니다.")
    if llm_fail > 0:
        print(f"  [LLM 실패] {llm_fail}건 → v22 예측 그대로 사용")

    # ── 성능 계산 ─────────────────────────────────────────────────────
    final_preds_arr = np.array(final_preds)
    correct = (final_preds_arr == labels)

    acc_v22_part = correct[v22_direct].mean() * 100 if n_v22 > 0 else 0.0
    acc_llm_part = correct[need_llm].mean() * 100    if n_llm > 0 else 0.0
    acc_total    = correct.mean() * 100
    f1_total     = f1_score(labels, final_preds_arr, average='macro', zero_division=0) * 100

    sep = "=" * 60
    lines = [
        sep,
        f"[전체 파이프라인 성능]  T={TEMPERATURE}, conf_thr={CONF_THR:.0%}",
        sep,
        f"  테스트 샘플     : {n_total:,}건",
        f"  v22 직접({n_v22:,}건) : Acc={acc_v22_part:.2f}%",
        f"  LLM 대상({n_llm:,}건) : Acc={acc_llm_part:.2f}%  (실패 {llm_fail}건)",
        f"  전체 Accuracy   : {acc_total:.2f}%",
        f"  전체 MacroF1    : {f1_total:.2f}%",
        sep,
        "",
        "[참고] v22 단독 (T=1.5, LLM 없음)",
        f"  Acc={base_acc:.2f}%  MacroF1={base_f1:.2f}%",
        "",
        "[클래스별 상세]",
        classification_report(labels, final_preds_arr,
                               target_names=LABEL_NAMES, digits=4, zero_division=0),
    ]

    output = "\n".join(lines)
    print("\n" + output)

    result_path = "results/evaluate_pipeline_results.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\n결과 저장: {result_path}")


if __name__ == '__main__':
    main()
