"""
9단계: Mahalanobis 적용 + threshold(PCT) 후보 통합 비교 실험

8단계에서 mahalanobis 거리 기반 score가 raw(KNN 평균거리) 대비
'정분류였는데 거부'를 134건 줄이면서 노벨탐지(3/7)를 그대로 유지하는
순개선임을 확인했다 (experiment_mahalanobis.txt).

3순위 진단(experiment_unjust_rejection_diagnosis.txt)에서는 억울한 거부의
margin(raw_score - threshold) 하위 25%(720건/2878건)가 threshold를 '아주
근소하게' 넘긴 경계선 케이스임을 확인했다 -> threshold(PCT)를 살짝
완화(=더 큰 percentile)하면 이 경계선 케이스들을 추가로 구제할 여지가 있다.

이번 실험은 두 개선책을 하나로 묶어 한 번의 임베딩/거리 계산으로 비교한다:
  (a) mahalanobis score 채택 (이미 검증됨, raw 대비 비등방적 분포 보정)
  (b) PCT 후보 3종 비교: 96(기존), 97, 98
      PCT를 높이면 threshold(percentile 기준값)가 커져서 score > thr 인
      샘플이 줄어듦 -> 학습거부가 감소하는 방향. 단, 너무 높이면 노벨
      탐지에 필요한 진짜 이상치까지 놓칠 위험이 있으므로 노벨 7개 탐지율을
      반드시 함께 확인한다.

가장 좋은 조합(정분류 거부를 가장 많이 줄이면서 노벨탐지 3/7을 유지하는
PCT)을 채택 후보로 제시한다.

실행: python pipeline_v22/scripts/experiment_mahalanobis_threshold_combo.py
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
OUT_TXT    = os.path.join(ROOT, 'results', 'experiment_mahalanobis_threshold_combo.txt')
BATCH_SIZE = 64
MAX_LEN    = 128
K          = 20     # raw(KNN) 비교용으로 동일하게 고정
SAFE_FLOOR = 4
SHRINK_RATIO = 1e-3
LABEL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4']
PCT_CANDIDATES = [96, 97, 98]   # 96=기존, 97/98=완화 후보

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
NOVEL_IDXS = [1, 2, 3, 4, 5, 9, 10]


def extract(model, loader, device):
    embs, labels, preds = [], [], []
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
    return (torch.cat(embs).numpy().astype(np.float32),
            torch.cat(labels).numpy().astype(np.int32),
            torch.cat(preds).numpy().astype(np.int32))


def mahalanobis_score(embs, target_labels, class_means, cov_inv):
    mu = class_means[target_labels]
    diff = embs.astype(np.float64) - mu
    left = diff @ cov_inv
    d2 = np.einsum('ij,ij->i', left, diff)
    return np.sqrt(np.maximum(d2, 0.0))


def eval_at_pct(pct, train_score, val_score, msg_score,
                train_preds, val_labels, msg_base_preds, misclassified, n_train):
    class_thr  = {c: float(np.percentile(val_score[val_labels == c], pct)) for c in range(5)}
    global_thr = float(np.percentile(val_score, pct))

    thr_arr = np.array([class_thr.get(int(p), global_thr) for p in train_preds])
    is_ood_train = train_score > thr_arr
    n_ood = int(is_ood_train.sum())
    n_ood_mis = int((is_ood_train & misclassified).sum())
    n_ood_correct = n_ood - n_ood_mis

    n_ood_msgs = 0
    n_novel_detected = 0
    msg_is_ood = []
    for i, (base_pred, score) in enumerate(zip(msg_base_preds, msg_score), 1):
        thr = class_thr.get(base_pred, global_thr)
        is_ood = bool(score > thr)
        msg_is_ood.append(is_ood)
        if is_ood:
            n_ood_msgs += 1
        if is_ood and i in NOVEL_IDXS:
            n_novel_detected += 1

    return dict(pct=pct, class_thr=class_thr, global_thr=global_thr,
                n_ood=n_ood, n_ood_mis=n_ood_mis, n_ood_correct=n_ood_correct,
                is_ood_train=is_ood_train, msg_is_ood=msg_is_ood,
                n_ood_msgs=n_ood_msgs, n_novel_detected=n_novel_detected,
                rate=n_ood / n_train * 100)


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
    train_embs, train_labels, train_preds = extract(model, train_loader, device)
    print(f"  shape: {train_embs.shape}")
    print("val CLS 임베딩/예측 추출 중...")
    val_embs, val_labels, val_preds = extract(model, val_loader, device)
    print(f"  shape: {val_embs.shape}")

    print("20개 테스트 메시지 임베딩/기본예측 계산 중...")
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
    msg_base_preds_arr = np.array(msg_base_preds, dtype=np.int32)

    misclassified = train_preds != train_labels
    n_train = len(train_df)

    # ── raw(기존) : KNN 평균거리 — PCT=96 한 번만 (참고 기준선) ──────────
    print(f"raw(KNN) 거리 계산 중 (K={K})...")
    nn_loo = NearestNeighbors(n_neighbors=K + 1, algorithm='brute', metric='cosine', n_jobs=-1)
    nn_loo.fit(train_embs)
    loo_dists, _ = nn_loo.kneighbors(train_embs)
    train_raw_score = loo_dists[:, 1:].mean(axis=1)

    nn_k = NearestNeighbors(n_neighbors=K, algorithm='brute', metric='cosine', n_jobs=-1)
    nn_k.fit(train_embs)
    val_dists, _ = nn_k.kneighbors(val_embs)
    msg_dists, _ = nn_k.kneighbors(msg_embs)
    val_raw_score = val_dists.mean(axis=1)
    msg_raw_score = msg_dists.mean(axis=1)
    raw_96 = eval_at_pct(96, train_raw_score, val_raw_score, msg_raw_score,
                         train_preds, val_labels, msg_base_preds_arr, misclassified, n_train)

    # ── Mahalanobis score 계산 (1회만) ──────────────────────────────
    print("클래스별 평균 벡터 / pooled 공분산행렬 계산 중...")
    dim = train_embs.shape[1]
    class_means = np.zeros((5, dim), dtype=np.float64)
    for c in range(5):
        class_means[c] = train_embs[train_labels == c].astype(np.float64).mean(axis=0)

    centered = train_embs.astype(np.float64) - class_means[train_labels]
    cov = (centered.T @ centered) / len(centered)
    mean_var = float(np.mean(np.diag(cov)))
    shrinkage = mean_var * SHRINK_RATIO
    cov_reg = cov + np.eye(dim) * shrinkage
    print(f"  공분산행렬 shape={cov.shape}  평균분산={mean_var:.6f}  shrinkage={shrinkage:.8f}")
    cov_inv = np.linalg.inv(cov_reg)

    print("Mahalanobis 거리 채점 중 (1회 계산 후 PCT 후보별로 재사용)...")
    train_maha_score = mahalanobis_score(train_embs, train_preds, class_means, cov_inv)
    val_maha_score   = mahalanobis_score(val_embs,   val_preds,   class_means, cov_inv)
    msg_maha_score   = mahalanobis_score(msg_embs,   msg_base_preds_arr, class_means, cov_inv)

    print(f"PCT 후보 {PCT_CANDIDATES} 에 대해 mahalanobis 채점 비교 중...")
    maha_results = {}
    for pct in PCT_CANDIDATES:
        maha_results[pct] = eval_at_pct(pct, train_maha_score, val_maha_score, msg_maha_score,
                                         train_preds, val_labels, msg_base_preds_arr,
                                         misclassified, n_train)

    # ── 리포트 작성 ──────────────────────────────────────────────────
    lines = []
    lines.append("=" * 100)
    lines.append("[9단계] Mahalanobis 적용 + PCT(threshold) 후보 통합 비교 실험")
    lines.append("=" * 100)
    lines.append(f"raw(기존)   : score = mean(KNN거리) (등방적 거리), PCT=96")
    lines.append(f"mahalanobis : score = sqrt((x-mu_c)^T * Sigma^-1 * (x-mu_c)) (분포 모양 반영)")
    lines.append(f"PCT 후보: {PCT_CANDIDATES}  (96=기존 기준, 97/98=threshold를 '완화'하여 학습거부 추가 감소 시도)")
    lines.append(f"  -> PCT를 높이면 class별 threshold(percentile 기준값)가 커져서 score>thr 인 샘플이 줄어듦")
    lines.append(f"     (단, 너무 높이면 진짜 이상치(노벨)까지 통과시킬 위험 -> 노벨 7개 탐지율로 안전선 확인)")
    lines.append(f"학습 샘플 수: {n_train}   |   노벨 시나리오 메시지 번호: {NOVEL_IDXS} (총 {len(NOVEL_IDXS)}개)")
    lines.append("")
    lines.append(f"{'방식':^18} | {'학습거부':>9} {'거부율':>7} | {'정분류였는데 거부':>14} {'오분류라서 거부':>12}"
                 f" | {'20개중 OOD':>9} | {'노벨 7개중 탐지':>12}")
    lines.append("-" * 100)
    lines.append(
        f"{'raw(기존,PCT=96)':^18} | {raw_96['n_ood']:>9} {raw_96['rate']:>6.2f}% "
        f"| {raw_96['n_ood_correct']:>14} {raw_96['n_ood_mis']:>12} "
        f"| {raw_96['n_ood_msgs']:>9} | {raw_96['n_novel_detected']:>9}/{len(NOVEL_IDXS)}"
    )
    for pct in PCT_CANDIDATES:
        e = maha_results[pct]
        tag = f"maha(PCT={pct})" + (" 기존동일" if pct == 96 else "")
        lines.append(
            f"{tag:^18} | {e['n_ood']:>9} {e['rate']:>6.2f}% "
            f"| {e['n_ood_correct']:>14} {e['n_ood_mis']:>12} "
            f"| {e['n_ood_msgs']:>9} | {e['n_novel_detected']:>9}/{len(NOVEL_IDXS)}"
        )
    lines.append("")

    # raw 대비 변화량 + 채택 가능 여부 판정
    lines.append("[raw(PCT=96) 대비 변화량 및 채택 가능성 판정]")
    best_pct = None
    best_reduction = -10**9
    for pct in PCT_CANDIDATES:
        e = maha_results[pct]
        d_total = e['n_ood'] - raw_96['n_ood']
        d_correct = e['n_ood_correct'] - raw_96['n_ood_correct']
        d_mis = e['n_ood_mis'] - raw_96['n_ood_mis']
        novel_ok = e['n_novel_detected'] >= raw_96['n_novel_detected']
        verdict = "채택 가능 (노벨탐지 유지/개선)" if novel_ok else "위험 (노벨탐지 하락 -> 기각)"
        lines.append(f"  PCT={pct}: 학습거부 {d_total:+d}건, 정분류였는데 거부 {d_correct:+d}건, "
                     f"오분류라서 거부 {d_mis:+d}건, 노벨탐지 {e['n_novel_detected']}/7  -> {verdict}")
        if novel_ok and -d_correct > best_reduction:
            best_reduction = -d_correct
            best_pct = pct
    lines.append("")
    if best_pct is not None:
        e = best_pct and maha_results[best_pct]
        lines.append(f"[권고] 노벨탐지를 유지하면서 '정분류였는데 거부'를 가장 많이 줄이는 조합: "
                     f"mahalanobis + PCT={best_pct} "
                     f"(raw 대비 정분류 거부 {-(e['n_ood_correct'] - raw_96['n_ood_correct']):+d}건 추가/총 개선)")
    else:
        lines.append("[권고] 모든 PCT 후보에서 노벨탐지가 하락함 -> threshold 조정 없이 mahalanobis(PCT=96)만 채택 권고")
    lines.append("")

    lines.append("=" * 100)
    lines.append("[20개 메시지 OOD 판정 상세: raw(PCT=96) vs mahalanobis(PCT 후보별)]")
    lines.append("=" * 100)
    header = f"{'#':>2} {'정답':^5} {'기본예측':^7} {'maha_score':^11}   {'raw96':^6}"
    for pct in PCT_CANDIDATES:
        header += f" {'m'+str(pct):^6}"
    header += "   메시지"
    lines.append(header)
    lines.append("-" * 100)
    for j in range(len(messages)):
        idx = j + 1
        true_label = messages[j][1]
        base_pred = int(msg_base_preds_arr[j])
        ms = msg_maha_score[j]
        r_flag = "OOD" if raw_96['msg_is_ood'][j] else " in"
        novel_mark = "*" if idx in NOVEL_IDXS else " "
        short = messages[j][0][:34] + ("..." if len(messages[j][0]) > 34 else "")
        row = f"{idx:>2}{novel_mark} L{true_label:^3}  {LABEL_NAMES[base_pred]:^6} {ms:>9.3f}     {r_flag:^6}"
        for pct in PCT_CANDIDATES:
            m_flag = "OOD" if maha_results[pct]['msg_is_ood'][j] else " in"
            row += f" {m_flag:^6}"
        row += f"   {short}"
        lines.append(row)
    lines.append("")
    lines.append("* = 노벨 시나리오로 간주한 메시지")
    lines.append("")

    lines.append("[해석 가이드]")
    lines.append("- '학습거부'/'정분류였는데 거부'가 raw(PCT=96) 대비 줄었는지가 1차 개선 지표")
    lines.append("- '노벨 7개중 탐지'가 raw의 3/7 미만으로 떨어지면 그 PCT 후보는 기각 (OOD 탐지 본래 목적 훼손)")
    lines.append("- 두 조건(정분류 거부 감소 + 노벨탐지 유지)을 모두 만족하는 PCT 중 가장 많이 줄인 값이 최종 채택 후보")
    lines.append("- 위 [권고] 항목이 비어있다면 mahalanobis(PCT=96)만 적용하는 것이 가장 안전한 선택")

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    for line in lines:
        print(line)
    print(f"\n결과 저장: {OUT_TXT}")


if __name__ == '__main__':
    main()
