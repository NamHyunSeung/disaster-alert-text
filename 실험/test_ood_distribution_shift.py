"""
Distribution Shift 테스트 - 학습 데이터에 없는 새로운 형태의 재난문자
마스킹 적용 / 미적용 결과 비교 + OOD 탐지 보수적 예측 비교

사전 준비:
  python 실험/build_ood_detector.py   # ood_stats.pt 생성

실행:
  python 실험/test_ood_distribution_shift.py
"""

import re
import sys
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '완성 모델', 'src'))

MODEL_DIR    = os.path.join(ROOT, 'model_v9n')
TOK_DIR      = os.path.join(ROOT, 'tokenizer_v9n')
OOD_STATS    = os.path.join(ROOT, '실험', 'ood_stats.pt')
MAX_LEN      = 128

LABEL_NAMES = ['L0 긴급아님', 'L1 낮음', 'L2 중간', 'L3 높음', 'L4 매우높음']

_MASK_KEYWORDS = sorted([
    '즉시 대피', '대피명령', '대피 명령', '긴급대피', '긴급 대피', '신속히 대피',
    '지진 발생', '쓰나미', '민방공', '테러',
    '경보', '주의보', '특보', '예비특보',
    '대피', '발생', '화재', '산불', '홍수', '태풍', '침수', '범람',
    '해제', '종료', '완료',
], key=len, reverse=True)

_SPACES = re.compile(r'\s+')


def mask_text(text: str) -> str:
    for kw in _MASK_KEYWORDS:
        text = text.replace(kw, ' ')
    return _SPACES.sub(' ', text).strip()


TEST_CASES = [
    {
        'id': 1,
        'label': 4,
        'name': '태양 흑점·지자기 폭풍',
        'text': '[행정안전부] 태양 흑점 폭발로 강력한 지자기 폭풍 발생. 전력망·GPS·항공 통신 마비 가능. 의료기기 의존 환자는 즉시 병원 대피 바랍니다.',
    },
    {
        'id': 2,
        'label': 4,
        'name': '사이버 공격·대규모 정전',
        'text': '[과학기술정보통신부] 전국 전력 제어망 사이버 공격으로 대규모 정전 임박. 엘리베이터·신호등 작동 중단 예상. 고층 건물 즉시 하층으로 이동 바랍니다.',
    },
    {
        'id': 3,
        'label': 4,
        'name': '드론 위협',
        'text': '[경찰청] 서울 중구 일대 정체불명 드론 다수 출몰, 위험물질 탑재 여부 확인 중. 해당 지역 주민은 실내 대피 후 창문 닫고 대기 바랍니다.',
    },
    {
        'id': 4,
        'label': 4,
        'name': '산불+유해가스 복합 재난',
        'text': '[환경부] 인천 서구 화학공단 인근 산불 확산으로 유해가스 누출 우려. 마스크 착용 후 바람 반대 방향으로 즉시 대피 바랍니다.',
    },
    {
        'id': 5,
        'label': 3,
        'name': '신종 감염병',
        'text': '[질병관리청] 원인불명 호흡기 질환 집단 발생(사망자 3명). 서울 전역 외출 자제, 마스크 착용 필수. 발열·기침 시 즉시 신고 바랍니다.',
    },
    {
        'id': 6,
        'label': 4,
        'name': '해저화산·쓰나미',
        'text': '[기상청] 동해 해저화산 폭발 감지. 강원·경북 동해안 지역 쓰나미 도달 예상시간 23분. 즉시 해안에서 500m 이상 내륙 또는 고지대로 대피 바랍니다.',
    },
    {
        'id': 7,
        'label': 4,
        'name': '소행성 파편 낙하',
        'text': '[행정안전부] 소행성 파편 경북 북부 지역 낙하 예상. 낙하 예상 반경 10km 주민은 즉시 지하 대피시설로 이동 바랍니다.',
    },
]


def predict(model, tokenizer, text, device):
    enc = tokenizer(
        text, truncation=True, padding='max_length',
        max_length=MAX_LEN, return_tensors='pt'
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    probs  = F.softmax(out.logits, dim=-1)[0].cpu().tolist()
    pred   = int(torch.tensor(probs).argmax())
    cls_emb = out.hidden_states[-1][:, 0, :].squeeze(0).cpu()
    return pred, probs, cls_emb


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=5)
    model.to(device)
    model.eval()

    # OOD 탐지기 (없으면 기본 모드로만 실행)
    ood_detector = None
    if Path(OOD_STATS).exists():
        from ood_detector import MahalanobisOOD
        ood_detector = MahalanobisOOD.load(OOD_STATS)
        print(f"OOD 탐지기 로드 완료 (threshold={ood_detector.threshold:.2f})\n")
    else:
        print("[경고] ood_stats.pt 없음 — OOD 보정 없이 실행합니다.")
        print("       먼저 python 실험/build_ood_detector.py 를 실행하세요.\n")

    correct_masked   = 0
    correct_unmasked = 0
    correct_ood_corr = 0  # OOD 보정 후 정확도

    use_ood = ood_detector is not None

    if use_ood:
        header = (f"{'#':<3} {'유형':<20} {'정답':<5} "
                  f"{'마스킹':<12} {'원문':<12} "
                  f"{'OOD보정':<12} {'dist':>7}  {'OOD?'}")
    else:
        header = (f"{'#':<3} {'유형':<20} {'정답':<5} "
                  f"{'마스킹예측':<12} {'원문예측':<12} "
                  f"{'마스킹OK':<8} {'원문OK'}")

    print(header)
    print('─' * (100 if use_ood else 80))

    for case in TEST_CASES:
        text        = case['text']
        masked_text = mask_text(text)
        true_label  = case['label']

        pred_m, probs_m, emb_m = predict(model, tokenizer, masked_text, device)
        pred_u, probs_u, _     = predict(model, tokenizer, text,         device)

        ok_m = 'O' if pred_m == true_label else 'X'
        ok_u = 'O' if pred_u == true_label else 'X'
        if pred_m == true_label: correct_masked   += 1
        if pred_u == true_label: correct_unmasked += 1

        if use_ood:
            final_pred, is_ood, dist = ood_detector.conservative_predict(emb_m, pred_m)
            ok_ood = 'O' if final_pred == true_label else 'X'
            if final_pred == true_label: correct_ood_corr += 1

            bumped = ' ↑' if (is_ood and pred_m < final_pred) else '  '
            print(f"{case['id']:<3} {case['name']:<20} L{true_label:<4} "
                  f"L{pred_m}({probs_m[pred_m]*100:4.1f}%){ok_m}   "
                  f"L{pred_u}({probs_u[pred_u]*100:4.1f}%){ok_u}   "
                  f"L{final_pred}{bumped}{ok_ood:<8}  "
                  f"{dist:7.2f}  {'OOD' if is_ood else 'in ':3}")
        else:
            print(f"{case['id']:<3} {case['name']:<20} L{true_label:<5} "
                  f"{LABEL_NAMES[pred_m]:<12} {LABEL_NAMES[pred_u]:<12} "
                  f"{ok_m:<8} {ok_u}")

        print(f"    [마스킹] {' '.join(f'L{i}:{p*100:4.1f}%' for i,p in enumerate(probs_m))}")
        print(f"    [원  문] {' '.join(f'L{i}:{p*100:4.1f}%' for i,p in enumerate(probs_u))}")
        print(f"    마스킹 텍스트: {masked_text[:80]}{'...' if len(masked_text)>80 else ''}")
        print()

    n = len(TEST_CASES)
    print('=' * (100 if use_ood else 80))
    print(f"마스킹 정확도     : {correct_masked}/{n}  ({correct_masked/n*100:.1f}%)")
    print(f"원문   정확도     : {correct_unmasked}/{n}  ({correct_unmasked/n*100:.1f}%)")
    if use_ood:
        print(f"OOD보정 정확도   : {correct_ood_corr}/{n}  ({correct_ood_corr/n*100:.1f}%)")
        delta = correct_ood_corr - correct_masked
        sign  = '+' if delta >= 0 else ''
        print(f"  → 마스킹 대비 {sign}{delta}개 ({sign}{delta/n*100:.1f}%p) 변화")


if __name__ == '__main__':
    main()
