"""
7단계: 이웃 라벨 일치도(Neighbor Label Agreement/Purity) 검증 실험

지금까지 실패한 세 가지 방법(K/PCT 조정, LOF 로컬 밀도 정규화, confidence 결합)은
모두 '신호의 크기(거리 크기, confidence 크기)'에 의존했고 셋 다 변별력을 만들지 못했다.
confidence는 정분류/오분류 가리지 않고 85~95%에 쏠려 있어 독립 신호가 되지 못했다.

이번에는 크기가 아니라 '이웃 집단의 정체성 일관성'이라는 완전히 다른 종류의
신호를 시도한다:

  purity = (K개 최근접 이웃 중, 그들의 '진짜 라벨'이 내 '예측 라벨'과 일치하는 비율)
  combo_score = raw_score * (1 - purity)

  - purity 높음 (이웃들의 실제 라벨이 내 예측과 일관되게 일치)
        -> (1-purity) 작아짐 -> score 깎여서 살아남음 (구제)
  - purity 낮음 (이웃 라벨이 뒤섞이거나 내 예측과 불일치)
        -> (1-purity) 커짐 -> score 거의 유지 (계속 거부)

직관: "거리는 멀어도 그 동네에 실제로 내가 예측된 클래스의 샘플들이
일관되게 모여 있다면" 그 영역은 해당 클래스의 변두리일 뿐 이상치가 아닐
가능성이 높고, 반대로 "이웃 라벨이 뒤섞여 있다면" 클래스 경계나 노벨
영역에 있을 가능성이 높다.

confidence(0.85~0.95 좁은 구간에 쏠림)와 달리 purity는 이웃 구성에 따라
0~1 전체 구간에 걸쳐 넓게 분포할 것으로 기대되어, 실제 변별력 있는
독립 신호가 될 가능성이 있다. (이미 계산된 KNN 이웃 인덱스에서 라벨만
조회하면 되므로 추가 연산비용도 거의 없음)

새 threshold 파라미터를 추가하지 않고 기존 PCT=96 percentile 방식을
그대로 재사용 (score 분포가 바뀌면 percentile이 새 분포에 맞춰
threshold를 알아서 재산출)

실행: python pipeline_v22/scripts/experiment_neighbor_purity.py
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
OUT_TXT    = os.path.join(ROOT, 'results', 'experiment_neighbor_purity.txt')
BATCH_SIZE = 64
MAX_LEN    = 128
PCT        = 96     # 기존과 동일하게 고정
K          = 20     # 기존과 동일
SAFE_FLOOR = 4
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


def eval_method(name, train_score, val_score, msg_score,
                train_labels, train_preds, val_labels, msg_base_preds, misclassified):
    class_thr  = {c: float(np.percentile(val_score[val_labels == c], PCT)) for c in range(5)}
    global_thr = float(np.percentile(val_score, PCT))

    thr_arr = np.array([class_thr.get(int(p), global_thr) for p in train_preds])
    is_ood_train = train_score > thr_arr
    n_ood = int(is_ood_train.sum())
    n_ood_mis = int((is_ood_train & misclassified).sum())
    n_ood_correct = n_ood - n_ood_mis

    msg_rows = []
    n_ood_msgs = 0
    n_novel_detected = 0
    for i, (base_pred, score) in enumerate(zip(msg_base_preds, msg_score), 1):
        thr = class_thr.get(base_pred, global_thr)
        is_ood = bool(score > thr)
        msg_rows.append(dict(idx=i, base=base_pred, score=float(score), thr=thr, is_ood=is_ood))
        if is_ood:
            n_ood_msgs += 1
        if is_ood and i in NOVEL_IDXS:
            n_novel_detected += 1

    return dict(name=name, class_thr=class_thr, global_thr=global_thr,
                n_ood=n_ood, n_ood_mis=n_ood_mis, n_ood_correct=n_ood_correct,
                is_ood_train=is_ood_train, msg_rows=msg_rows,
                n_ood_msgs=n_ood_msgs, n_novel_detected=n_novel_detected)


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

    # ── KNN 계산 (이번엔 인덱스도 함께 받아서 이웃 라벨 조회에 사용) ──
    print(f"KNN 인덱스 fit + 거리/이웃인덱스 계산 중 (K={K}, 1회만 수행)...")
    nn_loo = NearestNeighbors(n_neighbors=K + 1, algorithm='brute', metric='cosine', n_jobs=-1)
    nn_loo.fit(train_embs)
    loo_dists, loo_idxs = nn_loo.kneighbors(train_embs)
    train_raw_score = loo_dists[:, 1:].mean(axis=1)            # 자기 자신 제외
    train_neighbor_idxs = loo_idxs[:, 1:]                       # 자기 자신 제외 (N, K)

    nn_k = NearestNeighbors(n_neighbors=K, algorithm='brute', metric='cosine', n_jobs=-1)
    nn_k.fit(train_embs)
    val_dists, val_idxs = nn_k.kneighbors(val_embs)
    msg_dists, msg_idxs = nn_k.kneighbors(msg_embs)
    val_raw_score = val_dists.mean(axis=1)
    msg_raw_score = msg_dists.mean(axis=1)

    # ── 이웃 라벨 일치도(purity) = 이웃들의 '진짜 라벨'이 '내 예측 라벨'과 일치하는 비율 ──
    train_neighbor_true = train_labels[train_neighbor_idxs]               # (N, K)
    train_purity = (train_neighbor_true == train_preds[:, None]).mean(axis=1)

    val_neighbor_true = train_labels[val_idxs]
    val_purity = (val_neighbor_true == val_preds[:, None]).mean(axis=1)

    msg_neighbor_true = train_labels[msg_idxs]
    msg_purity = (msg_neighbor_true == msg_base_preds_arr[:, None]).mean(axis=1)

    print(f"  train purity 분포: mean={train_purity.mean():.3f}  std={train_purity.std():.3f}"
          f"  min={train_purity.min():.3f}  max={train_purity.max():.3f}")

    # ── combo_score = raw_score * (1 - purity) ──────────────────────
    train_combo_score = train_raw_score * (1.0 - train_purity)
    val_combo_score   = val_raw_score   * (1.0 - val_purity)
    msg_combo_score   = msg_raw_score   * (1.0 - msg_purity)

    print("PCT=96 기준으로 raw / combo 두 방식 채점 중...")
    raw = eval_method('raw(기존)', train_raw_score, val_raw_score, msg_raw_score,
                      train_labels, train_preds, val_labels, msg_base_preds, misclassified)
    combo = eval_method('combo(purity)', train_combo_score, val_combo_score, msg_combo_score,
                        train_labels, train_preds, val_labels, msg_base_preds, misclassified)

    # ── 리포트 작성 ──────────────────────────────────────────────────
    n_train = len(train_df)
    lines = []
    lines.append("=" * 100)
    lines.append(f"[7단계] 이웃 라벨 일치도(purity) 결합 검증 실험 (K={K}, PCT={PCT} 고정)")
    lines.append("=" * 100)
    lines.append("raw(기존)    : score = mean(KNN거리)")
    lines.append("combo(purity): score = mean(KNN거리) * (1 - purity)")
    lines.append("               purity = K개 이웃 중 '진짜 라벨'이 '내 예측 라벨'과 일치하는 비율")
    lines.append("               (이웃들이 내 예측과 일관되면 score 깎임->구제, 뒤섞이면 score 유지->계속 거부)")
    lines.append(f"train purity 분포: 평균={train_purity.mean():.3f}  표준편차={train_purity.std():.3f}"
                 f"  최소={train_purity.min():.3f}  최대={train_purity.max():.3f}"
                 f"   (참고: confidence는 0.85~0.95 좁은 구간에 쏠려 변별력이 없었음)")
    lines.append(f"학습 샘플 수: {n_train}   |   노벨 시나리오 메시지 번호: {NOVEL_IDXS}"
                 f" (총 {len(NOVEL_IDXS)}개 - 해킹/광케이블/태양폭풍/반도체/해수면/댐/핵발전소 등"
                 " 기존 5개 재난유형 범주를 벗어나는 신종 시나리오)")
    lines.append("")
    lines.append(f"{'방식':^14} | {'학습거부':>9} {'거부율':>7} | {'정분류였는데 거부':>14} {'오분류라서 거부':>12}"
                 f" | {'20개중 OOD':>9} | {'노벨 7개중 탐지':>12}")
    lines.append("-" * 100)
    for e in (raw, combo):
        lines.append(
            f"{e['name']:^14} | {e['n_ood']:>9} {e['n_ood']/n_train*100:>6.2f}% "
            f"| {e['n_ood_correct']:>14} {e['n_ood_mis']:>12} "
            f"| {e['n_ood_msgs']:>9} | {e['n_novel_detected']:>9}/{len(NOVEL_IDXS)}"
        )
    lines.append("")
    diff_total = combo['n_ood'] - raw['n_ood']
    diff_correct = combo['n_ood_correct'] - raw['n_ood_correct']
    lines.append(f"[raw -> combo 변화] 학습거부 {raw['n_ood']} -> {combo['n_ood']} ({diff_total:+d}),  "
                 f"정분류였는데 거부 {raw['n_ood_correct']} -> {combo['n_ood_correct']} ({diff_correct:+d}),  "
                 f"노벨탐지 {raw['n_novel_detected']}/7 -> {combo['n_novel_detected']}/7")
    lines.append("")

    raw_ood = raw['is_ood_train']
    combo_ood = combo['is_ood_train']
    rescued_correct = int((raw_ood & ~combo_ood & ~misclassified).sum())
    rescued_mis     = int((raw_ood & ~combo_ood & misclassified).sum())
    newly_correct   = int((~raw_ood & combo_ood & ~misclassified).sum())
    newly_mis       = int((~raw_ood & combo_ood & misclassified).sum())
    lines.append("[학습 데이터 차원 raw -> combo 전환 분석]")
    lines.append(f"  - 억울한 거부에서 구제됨 (정분류인데 raw=OOD -> combo=in)      : {rescued_correct:>6} 건  (방향 B 개선)")
    lines.append(f"  - 오분류 거부에서 풀려남 (오분류인데 raw=OOD -> combo=in)       : {rescued_mis:>6} 건  (놓친 오분류, 부작용)")
    lines.append(f"  - 새로 억울하게 거부됨 (정분류인데 raw=in  -> combo=OOD)       : {newly_correct:>6} 건  (새 부작용)")
    lines.append(f"  - 새로 오분류 거부 포착 (오분류인데 raw=in  -> combo=OOD)       : {newly_mis:>6} 건  (방향 A 개선)")
    lines.append("")

    flips_to_in, flips_to_ood = [], []
    for r0, r1 in zip(raw['msg_rows'], combo['msg_rows']):
        if r0['is_ood'] and not r1['is_ood']:
            flips_to_in.append(r1['idx'])
        elif not r0['is_ood'] and r1['is_ood']:
            flips_to_ood.append(r1['idx'])
    novel_lost = [i for i in flips_to_in if i in NOVEL_IDXS]
    novel_gained = [i for i in flips_to_ood if i in NOVEL_IDXS]
    lines.append("[기준 raw 대비, 20개 메시지 OOD 판정이 바뀌는 지점 (combo 적용 시)]")
    lines.append(f"  OOD->in 전환 {flips_to_in if flips_to_in else '없음'} "
                 f"(이 중 노벨 손실: {novel_lost if novel_lost else '없음'})")
    lines.append(f"  in->OOD 전환 {flips_to_ood if flips_to_ood else '없음'} "
                 f"(이 중 노벨 추가: {novel_gained if novel_gained else '없음'})")
    lines.append("")

    lines.append("=" * 100)
    lines.append("[20개 메시지 OOD 판정 상세: raw vs combo (purity 포함)]")
    lines.append("=" * 100)
    lines.append(f"{'#':>2} {'정답':^5} {'기본예측':^7} {'purity':^7}   {'raw':^5} {'combo':^5}   메시지")
    lines.append("-" * 100)
    for j in range(len(messages)):
        idx = raw['msg_rows'][j]['idx']
        true_label = messages[j][1]
        base_pred = raw['msg_rows'][j]['base']
        pur = msg_purity[j]
        r_flag = "OOD" if raw['msg_rows'][j]['is_ood'] else " in"
        c_flag = "OOD" if combo['msg_rows'][j]['is_ood'] else " in"
        novel_mark = "*" if idx in NOVEL_IDXS else " "
        short = messages[j][0][:38] + ("..." if len(messages[j][0]) > 38 else "")
        lines.append(f"{idx:>2}{novel_mark} L{true_label:^3}  {LABEL_NAMES[base_pred]:^6} {pur*100:>6.1f}%   {r_flag:^5} {c_flag:^5}   {short}")
    lines.append("")
    lines.append("* = 노벨 시나리오로 간주한 메시지")
    lines.append("")

    rescued_idx = np.where(raw_ood & ~combo_ood & ~misclassified)[0]
    lines.append(f"[참고: 억울한 거부에서 구제된 샘플 예시 (총 {len(rescued_idx)}건 중 최대 8건 샘플)]")
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(rescued_idx, size=min(8, len(rescued_idx)), replace=False) if len(rescued_idx) else []
    for i in sample_idx:
        text = str(train_df.iloc[int(i)]['text'])[:55]
        lines.append(f"  - 정답=L{train_labels[i]} 예측=L{train_preds[i]}(purity={train_purity[i]*100:.1f}%) "
                     f"raw_score={train_raw_score[i]:.6f} combo_score={train_combo_score[i]:.6f} | {text}")
    lines.append("")

    newly_idx = np.where(~raw_ood & combo_ood & ~misclassified)[0]
    lines.append(f"[참고: combo 적용 후 새로 억울하게 거부된 샘플 예시 (총 {len(newly_idx)}건 중 최대 8건 샘플)]")
    sample_idx2 = rng.choice(newly_idx, size=min(8, len(newly_idx)), replace=False) if len(newly_idx) else []
    for i in sample_idx2:
        text = str(train_df.iloc[int(i)]['text'])[:55]
        lines.append(f"  - 정답=L{train_labels[i]} 예측=L{train_preds[i]}(purity={train_purity[i]*100:.1f}%) "
                     f"raw_score={train_raw_score[i]:.6f} combo_score={train_combo_score[i]:.6f} | {text}")
    lines.append("")

    lines.append("[해석 가이드]")
    lines.append("- '정분류였는데 거부'(방향 B)가 raw 대비 combo에서 줄었다면 -> 이웃 라벨 일치도가 실제로 변별력 있는 신호라는 근거")
    lines.append("- '오분류라서 거부'(방향 A 포착력)가 combo에서도 유지/개선됐다면 -> 부작용 없이 보정된 것")
    lines.append("- '노벨 7개중 탐지'가 combo에서 줄었다면 -> 노벨 샘플 주변에도 우연히 같은 라벨 이웃이 몰려있어 구제되어 버리는 트레이드오프")
    lines.append("- train purity의 표준편차가 confidence보다 훨씬 크다면 -> 이 신호가 실제로 폭넓게 분포하며 변별 정보를 담고 있다는 뜻")

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    for line in lines:
        print(line)
    print(f"\n결과 저장: {OUT_TXT}")


if __name__ == '__main__':
    main()
