"""
재난문자 중요도 분류 - 실시간 예측 (2단계 파이프라인)

Stage 1: 규칙 기반 (고확신 케이스 즉시 판정)
Stage 2: KoELECTRA-v2 (나머지 케이스)

사용법:
  python predict.py              # 대화형 입력 모드
  python predict.py --demo       # 예시 메시지 자동 테스트
  python predict.py --demo --model-only  # 규칙 없이 모델만 사용
"""

import re
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── 설정 ──────────────────────────────────────
MODEL_DIR   = "koelectra_v3_model"
MAX_LENGTH  = 96
LABEL_NAMES = ['긴급', '주의', '일반']
COLORS      = {'긴급': '[긴급]', '주의': '[주의]', '일반': '[일반]'}

# 불확실 경고 임계값 (모델 예측 시만 적용)
UNCERTAIN_THRESH = {'긴급': 0.60, '주의': 0.70, '일반': 0.70}
EMERG_THRESH     = 0.10  # 긴급 분류 임계값 (최적화된 값)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 발신기관명 마스킹 ─────────────────────────
_ORG_PATTERN = re.compile(r'\[[^\]]{1,20}\]')

def normalize_text(text: str) -> str:
    return _ORG_PATTERN.sub('[기관]', text)


# ── Stage 1: 규칙 기반 분류기 ─────────────────
# 오분류 위험이 거의 없는 고확신 패턴만 사용
_CERT_EMERG = [
    '즉시 대피', '대피명령', '대피 명령', '긴급대피', '긴급 대피', '신속히 대피',
    '지진 발생', '쓰나미', '민방공 경보', '민방공경보', '테러 발생',
]
_CERT_CAUTION = [
    '호우경보', '호우주의보', '태풍경보', '태풍주의보',
    '한파경보', '한파주의보', '폭염경보', '폭염주의보',
    '대설경보', '대설주의보', '강풍경보', '강풍주의보',
    '풍랑경보', '풍랑주의보',
]
_CERT_GENERAL = ['찾습니다', '실종된']   # 실종자 수색 (인물 묘사 동반)


def rule_classify(msg: str):
    """
    고확신 케이스만 판정. 불명확하면 None 반환 → 모델에 위임.
    반환: 0(긴급) / 1(주의) / 2(일반) / None(모델 위임)
    """
    has_emerg   = any(kw in msg for kw in _CERT_EMERG)
    has_caution = any(kw in msg for kw in _CERT_CAUTION)
    has_general = any(kw in msg for kw in _CERT_GENERAL) and 'cm' in msg

    if has_emerg and not has_caution:                        return 0
    if has_caution and not has_emerg:                        return 1
    if has_general and not has_emerg and not has_caution:    return 2
    return None


# ── 모델 로드 ─────────────────────────────────
print(f"모델 로드 중... ({device})", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()
print("로드 완료\n", flush=True)


# ── Stage 2: 모델 예측 ────────────────────────
def model_predict(text: str) -> dict:
    inputs = tokenizer(
        text, truncation=True, padding='max_length',
        max_length=MAX_LENGTH, return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0]
    pred_idx = 0 if probs[0].item() >= EMERG_THRESH else probs.argmax().item()
    return {
        'label':      LABEL_NAMES[pred_idx],
        'confidence': probs[pred_idx].item(),
        'probs':      {LABEL_NAMES[i]: probs[i].item() for i in range(3)},
        'stage':      'model',
    }


# ── 파이프라인 예측 ───────────────────────────
def predict(text: str, use_rules: bool = True) -> dict:
    """
    use_rules=True : 규칙 1단계 → 모델 2단계 파이프라인
    use_rules=False: 모델만 사용
    """
    text = normalize_text(text)   # 발신기관명 마스킹
    if use_rules:
        rule = rule_classify(text)
        if rule is not None:
            label = LABEL_NAMES[rule]
            return {
                'label':      label,
                'confidence': 1.0,
                'probs':      {n: (1.0 if n == label else 0.0) for n in LABEL_NAMES},
                'stage':      'rule',
            }
    return model_predict(text)


# ── 출력 ──────────────────────────────────────
def print_result(text: str, result: dict):
    label  = result['label']
    conf   = result['confidence']
    stage  = result['stage']
    stage_str = '[규칙]' if stage == 'rule' else '[모델]'

    print(f"\n{'─'*58}")
    print(f"입력: {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"{'─'*58}")
    uncertain = (stage == 'model') and (conf < UNCERTAIN_THRESH[label])
    warn_str  = '  [수동 확인 권고]' if uncertain else ''
    print(f"분류: {COLORS[label]} {label}  확신도 {conf*100:.1f}%  {stage_str}{warn_str}")
    if stage == 'model':
        print(f"  긴급 {result['probs']['긴급']*100:5.1f}%  "
              f"주의 {result['probs']['주의']*100:5.1f}%  "
              f"일반 {result['probs']['일반']*100:5.1f}%")
    print(f"{'─'*58}\n")


# ── 예시 메시지 ───────────────────────────────
DEMO_MESSAGES = [
    # 긴급
    ("[창녕군청] 현재 창녕군 영산면 산불확산 우려, 인근 주민들은 안전한 곳으로 즉시 대피바랍니다.", "긴급"),
    ("[행정안전부] 지진 발생(규모4.5), 실내에서는 탁자 아래로 대피, 야외에서는 건물 밖으로 이동하세요.", "긴급"),
    ("[경기도] 민방공 경보 발령, 인근 지하 대피소로 즉시 대피하시기 바랍니다.", "긴급"),
    # 주의
    ("[행정안전부] 오늘 경남(거제) 호우경보 발효. 산사태·침수 위험지역 대피, 외출 자제 바랍니다.", "주의"),
    ("[서울시] 내일까지 대설주의보 발효. 빙판길 낙상 주의, 대중교통 이용 바랍니다.", "주의"),
    ("[목포시] 오늘 11:13분경 목포시 옥암동 재활용센터에서 화재 발생으로 많은 연기가 발생 중입니다.", "주의"),
    # 일반
    ("[전북경찰청] 전주시에서 실종된 민도식씨(남,70세)를 찾습니다. 160cm, 회색줄무늬티 vo.la/TXWtJ", "일반"),
    ("[세종시청] 조치원읍 신안리 일원 상수도관 누수로 현재 단수 중, 12시경 복구 완료 예정입니다.", "일반"),
    ("[경기도청] 코로나19 예방접종 사전예약이 시작됩니다. 예약은 코로나19 예방접종 누리집을 이용하세요.", "일반"),
]

# ── 메인 ──────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--demo',       action='store_true', help='예시 메시지 자동 테스트')
parser.add_argument('--model-only', action='store_true', help='규칙 없이 모델만 사용')
args = parser.parse_args()

USE_RULES = not args.model_only

if args.demo:
    mode_str = "모델 단독" if args.model_only else "규칙+모델 파이프라인"
    print(f"=== 예시 메시지 테스트 ({mode_str}) ===\n")
    correct, rule_cnt, model_cnt = 0, 0, 0

    for text, true_label in DEMO_MESSAGES:
        result = predict(text, use_rules=USE_RULES)
        pred   = result['label']
        ok     = "O" if pred == true_label else "X"
        if pred == true_label:
            correct += 1
        if result['stage'] == 'rule':
            rule_cnt += 1
        else:
            model_cnt += 1
        print_result(text, result)
        print(f"  정답: {true_label}  예측: {pred}  [{ok}]")

    print(f"\n정확도: {correct}/{len(DEMO_MESSAGES)} ({correct/len(DEMO_MESSAGES)*100:.0f}%)")
    if USE_RULES:
        print(f"규칙 판정: {rule_cnt}건  /  모델 판정: {model_cnt}건")

else:
    print("재난문자를 입력하면 긴급/주의/일반을 분류합니다.")
    print(f"모드: {'규칙+모델 파이프라인' if USE_RULES else '모델 단독'}")
    print("종료: 'q' 입력\n")
    while True:
        text = input("재난문자 입력 > ").strip()
        if text.lower() == 'q':
            break
        if not text:
            continue
        result = predict(text, use_rules=USE_RULES)
        print_result(text, result)
