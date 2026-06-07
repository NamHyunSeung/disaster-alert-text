"""
3단계: threshold 완화 실험 (PCT 값을 여러 개로 바꿔가며 재보정)

임베딩/LOO score/val score/20개 테스트 메시지 임베딩을 "한 번만" 계산해두고,
PCT 후보값(95~99)마다 새 threshold를 산출해
  (a) 학습 데이터 OOD 거부율
  (b) 20개 테스트 메시지(노벨 시나리오 포함)의 OOD 탐지 패턴
이 어떻게 같이 변하는지 비교한다.

실행: python pipeline_v22/scripts/experiment_threshold_pct.py
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # pipeline_v22
PROJ = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(PROJ, '완성 모델', 'src'))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.neighbors import NearestNeighbors

from dataset_v2 import load_and_split_v2, DisasterDatasetAug

MODEL_DIR  = os.path.join(ROOT, 'model')
TOK_DIR    = os.path.join(ROOT, 'tokenizer')
DATA_PATH  = os.path.join(PROJ, '중요파일', 'data', 'raw',
                          '재난문자_레이블링결과_dedup_v7.xlsx')
OUT_TXT    = os.path.join(ROOT, 'results', 'experiment_threshold_pct.txt')
BATCH_SIZE = 64
MAX_LEN    = 128
K          = 20
SAFE_FLOOR = 4
PCT_LIST   = [95, 96, 97, 98, 99]
LABEL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4']

# predict_v22_knn_ood.py 와 동일한 20개 테스트 메시지 (노벨 시나리오 포함)
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
# 모델이 학습한 5개 재난 카테고리 범위를 벗어나는 "노벨" 시나리오로 보이는 메시지 번호
# (해킹/광케이블/태양폭풍/반도체/해수면/댐/핵발전소 = 기존 학습 데이터에 없을 법한 신종 재난유형)
NOVEL_IDXS = [1, 2, 3, 4, 5, 9, 10]


def extract(model, loader, device):
    embs, labels, preds, confs = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            ttids = batch.get('token_type_ids')
            kwargs = dict(input_ids=ids, attention_mask=attn, output_hidden_states=True)
            if ttids is not None:
                kwargs['token_type_ids'] = ttids.to(device)
            out = model(**kwargs)
            probs = F.softmax(out.logits, dim=-1)
            embs.append(out.hidden_states[-1][:, 0, :].cpu())
            labels.append(batch['label'])
            preds.append(probs.argmax(dim=-1).cpu())
            confs.append(probs.max(dim=-1).values.cpu())
    return (torch.cat(embs).numpy().astype(np.float32),
            torch.cat(labels).numpy().astype(np.int32),
            torch.cat(preds).numpy().astype(np.int32),
            torch.cat(confs).numpy().astype(np.float32))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("데이터 로드 중...")
    train_df, val_df, _ = load_and_split_v2(DATA_PATH)
    print(f"  train={len(train_df)}  val={len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device).eval()

    train_loader = DataLoader(DisasterDatasetAug(train_df, tokenizer, MAX_LEN, augment=False),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader   = DataLoader(DisasterDatasetAug(val_df, tokenizer, MAX_LEN, augment=False),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("train CLS 임베딩/예측 추출 중...")
    train_embs, train_labels, train_preds, train_confs = extract(model, train_loader, device)
    print(f"  shape: {train_embs.shape}")
    print("val CLS 임베딩 추출 중...")
    val_embs, val_labels, _, _ = extract(model, val_loader, device)
    print(f"  shape: {val_embs.shape}")

    print(f"KNN 인덱스 fit + LOO score 계산 (K={K})...")
    nn_loo = NearestNeighbors(n_neighbors=K + 1, algorithm='brute', metric='cosine', n_jobs=-1)
    nn_loo.fit(train_embs)
    loo_dists, _ = nn_loo.kneighbors(train_embs)
    train_scores = loo_dists[:, 1:].mean(axis=1)   # 자기 자신 제외 (self-match 버그 수정)

    nn_k = NearestNeighbors(n_neighbors=K, algorithm='brute', metric='cosine', n_jobs=-1)
    nn_k.fit(train_embs)
    val_dists, _ = nn_k.kneighbors(val_embs)
    val_scores = val_dists.mean(axis=1)

    print("20개 테스트 메시지 임베딩/기본예측/KNN score 계산 중...")
    msg_embs, msg_base_preds = [], []
    for msg, _ in messages:
        enc = tokenizer(msg, truncation=True, padding='max_length',
                        max_length=MAX_LEN, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            probs = F.softmax(out.logits, dim=-1)[0].cpu()
        msg_base_preds.append(int(probs.argmax()))
        msg_embs.append(out.hidden_states[-1][:, 0, :].squeeze(0).cpu().numpy().astype(np.float32))
    msg_embs = np.stack(msg_embs).astype(np.float32)
    msg_dists, _ = nn_k.kneighbors(msg_embs)
    msg_scores = msg_dists.mean(axis=1)

    misclassified = train_preds != train_labels

    # ── PCT 후보별 실험 ──────────────────────────────────────────────
    exp = []
    for pct in PCT_LIST:
        class_thr  = {c: float(np.percentile(val_scores[val_labels == c], pct)) for c in range(5)}
        global_thr = float(np.percentile(val_scores, pct))

        thr_arr = np.array([class_thr.get(int(p), global_thr) for p in train_preds])
        is_ood_train = train_scores > thr_arr
        n_ood = int(is_ood_train.sum())
        n_ood_mis = int((is_ood_train & misclassified).sum())
        n_ood_correct = n_ood - n_ood_mis

        msg_rows = []
        n_ood_msgs = 0
        n_novel_detected = 0
        for i, ((msg, true_label), base_pred, score) in enumerate(zip(messages, msg_base_preds, msg_scores), 1):
            thr = class_thr.get(base_pred, global_thr)
            is_ood = bool(score > thr)
            final_pred = max(base_pred, SAFE_FLOOR) if is_ood else base_pred
            msg_rows.append(dict(idx=i, true=true_label, base=base_pred, final=final_pred,
                                 score=float(score), thr=thr, is_ood=is_ood))
            if is_ood:
                n_ood_msgs += 1
            if is_ood and i in NOVEL_IDXS:
                n_novel_detected += 1

        exp.append(dict(pct=pct, class_thr=class_thr, global_thr=global_thr,
                        n_ood=n_ood, n_ood_mis=n_ood_mis, n_ood_correct=n_ood_correct,
                        msg_rows=msg_rows, n_ood_msgs=n_ood_msgs,
                        n_novel_detected=n_novel_detected))

    # ── 리포트 작성 ──────────────────────────────────────────────────
    n_train = len(train_df)
    lines = []
    lines.append("=" * 100)
    lines.append("[3단계] threshold 완화 실험: PCT 값에 따른 학습 데이터 거부율 vs 노벨 시나리오 탐지율")
    lines.append("=" * 100)
    lines.append(f"학습 샘플 수: {n_train}   |   노벨 시나리오로 간주한 테스트 메시지 번호: {NOVEL_IDXS}"
                 f" (총 {len(NOVEL_IDXS)}개 - 해킹/광케이블/태양폭풍/반도체/해수면/댐/핵발전소 등"
                 " 기존 5개 재난유형 범주를 벗어나는 신종 시나리오)")
    lines.append("")
    lines.append(f"{'PCT':>4} | {'학습거부':>9} {'거부율':>7} | {'정분류였는데 거부':>14} {'오분류라서 거부':>12}"
                 f" | {'20개중 OOD개수':>12} | {'노벨 7개중 탐지':>12}")
    lines.append("-" * 100)
    for e in exp:
        lines.append(
            f"{e['pct']:>4} | {e['n_ood']:>9} {e['n_ood']/n_train*100:>6.2f}% "
            f"| {e['n_ood_correct']:>14} {e['n_ood_mis']:>12} "
            f"| {e['n_ood_msgs']:>12} | {e['n_novel_detected']:>9}/{len(NOVEL_IDXS)}"
        )
    lines.append("")

    # PCT=95(현재) 대비 어떤 메시지의 OOD 판정이 바뀌는지 추적
    base_exp = exp[0]
    lines.append(f"[기준 PCT={base_exp['pct']} 대비, 20개 메시지 OOD 판정이 바뀌는 지점]")
    for e in exp[1:]:
        flips_to_in  = []
        flips_to_ood = []
        for r0, r1 in zip(base_exp['msg_rows'], e['msg_rows']):
            if r0['is_ood'] and not r1['is_ood']:
                flips_to_in.append(r1['idx'])
            elif not r0['is_ood'] and r1['is_ood']:
                flips_to_ood.append(r1['idx'])
        novel_lost = [i for i in flips_to_in if i in NOVEL_IDXS]
        lines.append(f"  PCT {base_exp['pct']}→{e['pct']}: "
                     f"OOD→in 전환 {flips_to_in if flips_to_in else '없음'} "
                     f"(이 중 노벨 시나리오: {novel_lost if novel_lost else '없음'})  |  "
                     f"in→OOD 전환 {flips_to_ood if flips_to_ood else '없음'}")
    lines.append("")

    # PCT별 클래스 threshold 값
    lines.append("[PCT별 클래스 threshold (val score의 p{PCT})]")
    for e in exp:
        thr_str = "  ".join(f"L{c}={e['class_thr'][c]:.6f}" for c in range(5))
        lines.append(f"  PCT={e['pct']:>3}: {thr_str}")
    lines.append("")

    # 각 PCT의 20개 메시지 상세 (마지막 PCT는 전체, 나머지는 OOD 판정 컬럼만 비교용으로 표시)
    lines.append("=" * 100)
    lines.append("[PCT별 20개 메시지 OOD 판정 상세]")
    lines.append("=" * 100)
    header = f"{'#':>2} {'정답':^4} {'기본예측':^8} " + " ".join(f"PCT{p:>2}" for p in PCT_LIST) + "   메시지"
    lines.append(header)
    lines.append("-" * 100)
    for j in range(len(messages)):
        idx = exp[0]['msg_rows'][j]['idx']
        true_label = exp[0]['msg_rows'][j]['true']
        base_pred = exp[0]['msg_rows'][j]['base']
        flags = "  ".join((" OOD" if exp[k]['msg_rows'][j]['is_ood'] else "  in") for k in range(len(exp)))
        novel_mark = "★" if idx in NOVEL_IDXS else " "
        short = messages[j][0][:40] + ("…" if len(messages[j][0]) > 40 else "")
        lines.append(f"{idx:>2}{novel_mark} L{true_label:^3}  {LABEL_NAMES[base_pred]:^6}  {flags}   {short}")
    lines.append("")
    lines.append("★ = 노벨 시나리오로 간주한 메시지")
    lines.append("")
    lines.append("[해석 가이드]")
    lines.append("- '학습거부' 행이 줄어들수록 → 멀쩡한 학습 데이터를 OOD로 억울하게 거부하는 비중이 줄어듦 (방향 A 효과)")
    lines.append("- '노벨 7개중 탐지' 가 줄어들면 → threshold를 너무 완화해서 진짜 노벨 시나리오 탐지력이 손실되는 트레이드오프 발생")
    lines.append("- 두 지표를 모두 만족하는 PCT(거부율은 충분히 낮아지고, 노벨 탐지는 유지되는 지점)가 최적 후보")

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    for line in lines:
        print(line)
    print(f"\n결과 저장: {OUT_TXT}")


if __name__ == '__main__':
    main()
