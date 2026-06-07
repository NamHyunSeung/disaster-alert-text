"""
9단계: '억울한 거부' 샘플 패턴 진단 (순수 분석, 신규 score 실험 아님)

지금까지 raw(KNN거리) 기준 OOD 판정에서 '정분류인데 거부당한' 샘플이
2878건(전체의 약 3.0%) 존재한다는 사실은 확인했지만, "이들이 왜 거부되는지"
"이들에게 공통된 특징이 있는지"는 아직 분석한 적이 없다.

이 스크립트는 새로운 score 방법을 제안하지 않고, raw 기준
'정분류인데 거부됨'(A=억울한 거부) vs '정분류이고 정상 통과'(B=기준 비교군)
두 그룹을 다음 축으로 비교해 패턴을 진단한다:

  1) 진짜 라벨(클래스) 분포  - 특정 클래스에 쏠려 있는가?
  2) 텍스트 길이            - 유난히 짧거나 긴가?
  3) 모델 confidence        - 확신도가 낮은 경계선 샘플들인가?
  4) raw_score(거부 정도)   - threshold를 살짝 넘기는 수준인가, 한참 넘기는가?
  5) 빈출 단어              - A에서 B 대비 과대표집된 단어(템플릿/표현)가 있는가?

실행: python pipeline_v22/scripts/experiment_unjust_rejection_diagnosis.py
"""

import sys, os, re
from collections import Counter

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
OUT_TXT    = os.path.join(ROOT, 'results', 'experiment_unjust_rejection_diagnosis.txt')
BATCH_SIZE = 64
MAX_LEN    = 128
PCT        = 96
K          = 20
LABEL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4']

STOPWORDS = set("""
이 그 저 것 등 및 의 에 를 을 가 은 는 이며 으로 로 에서 에게 하는 한 합니다 바랍니다
주시기 주시길 주시고 주세요 권고 안내 발생 관련 인근 지역 시간 오늘 현재 기준 위해
""".split())


def tokenize(text):
    words = re.findall(r'[가-힣]{2,}', str(text))
    return [w for w in words if w not in STOPWORDS]


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


def pct_dist(arr):
    return dict(mean=float(np.mean(arr)), median=float(np.median(arr)),
                std=float(np.std(arr)),
                p25=float(np.percentile(arr, 25)), p75=float(np.percentile(arr, 75)))


def fmt_dist(d, unit=""):
    return (f"평균={d['mean']:.2f}{unit}  중앙값={d['median']:.2f}{unit}  "
            f"표준편차={d['std']:.2f}{unit}  IQR=[{d['p25']:.2f}, {d['p75']:.2f}]{unit}")


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

    print("train CLS 임베딩/예측/confidence 추출 중...")
    train_embs, train_labels, train_preds, train_confs = extract(model, train_loader, device)
    print("val CLS 임베딩/예측/confidence 추출 중...")
    val_embs, val_labels, val_preds, val_confs = extract(model, val_loader, device)

    print(f"raw(KNN) 거리 계산 중 (K={K})...")
    nn_loo = NearestNeighbors(n_neighbors=K + 1, algorithm='brute', metric='cosine', n_jobs=-1)
    nn_loo.fit(train_embs)
    loo_dists, _ = nn_loo.kneighbors(train_embs)
    train_raw_score = loo_dists[:, 1:].mean(axis=1)

    nn_k = NearestNeighbors(n_neighbors=K, algorithm='brute', metric='cosine', n_jobs=-1)
    nn_k.fit(train_embs)
    val_dists, _ = nn_k.kneighbors(val_embs)
    val_raw_score = val_dists.mean(axis=1)

    class_thr  = {c: float(np.percentile(val_raw_score[val_labels == c], PCT)) for c in range(5)}
    global_thr = float(np.percentile(val_raw_score, PCT))
    thr_arr = np.array([class_thr.get(int(p), global_thr) for p in train_preds])
    is_ood = train_raw_score > thr_arr
    correct = train_preds == train_labels

    # A = 억울한 거부 (정분류인데 OOD), B = 정상 통과 (정분류이고 in)
    A_mask = correct & is_ood
    B_mask = correct & ~is_ood
    A_idx = np.where(A_mask)[0]
    B_idx = np.where(B_mask)[0]
    print(f"  A(억울한 거부)={len(A_idx)}건   B(정상 통과 기준군)={len(B_idx)}건")

    texts_A = train_df.iloc[A_idx]['text'].astype(str).tolist()
    texts_B = train_df.iloc[B_idx]['text'].astype(str).tolist()
    lens_A = np.array([len(t) for t in texts_A])
    lens_B = np.array([len(t) for t in texts_B])

    margin_A = train_raw_score[A_idx] - thr_arr[A_idx]   # threshold를 얼마나 넘었는가

    # ── 1) 라벨 분포 ──────────────────────────────────────────────
    labelcnt_A = Counter(train_labels[A_idx].tolist())
    labelcnt_B = Counter(train_labels[B_idx].tolist())
    total_A, total_B = len(A_idx), len(B_idx)

    # ── 5) 빈출 단어 (A에서 B 대비 과대표집) ─────────────────────
    cnt_A = Counter()
    for t in texts_A:
        cnt_A.update(set(tokenize(t)))   # 문서당 1회만 카운트 (도큐먼트 빈도)
    cnt_B = Counter()
    for t in texts_B:
        cnt_B.update(set(tokenize(t)))

    MIN_DF = 15
    overrep = []
    for w, ca in cnt_A.items():
        if ca < MIN_DF:
            continue
        cb = cnt_B.get(w, 0)
        rate_a = ca / total_A
        rate_b = (cb + 1) / total_B   # 라플라스 스무딩
        ratio = rate_a / rate_b
        overrep.append((w, ca, cb, rate_a * 100, rate_b * 100, ratio))
    overrep.sort(key=lambda x: -x[5])

    # ── 리포트 작성 ──────────────────────────────────────────────
    lines = []
    lines.append("=" * 100)
    lines.append("[9단계] 억울한 거부 샘플 패턴 진단 (raw 기준 OOD 판정, K=20 PCT=96)")
    lines.append("=" * 100)
    lines.append("A = 억울한 거부 (정분류인데 raw=OOD)")
    lines.append("B = 정상 통과 기준군 (정분류이고 raw=in)")
    lines.append(f"A={total_A}건 ({total_A/len(train_df)*100:.2f}% of 전체 {len(train_df)})   "
                 f"B={total_B}건 ({total_B/len(train_df)*100:.2f}% of 전체)")
    lines.append("")

    lines.append("-" * 100)
    lines.append("[1] 진짜 라벨(클래스) 분포 비교  -  특정 클래스에 쏠려 있는가?")
    lines.append("-" * 100)
    lines.append(f"{'라벨':^6} | {'A(억울한 거부) 건수':>18} {'A 비율':>8} | {'B(정상 통과) 건수':>16} {'B 비율':>8} | {'쏠림(A/B 비율)':>12}")
    for c in range(5):
        a, b = labelcnt_A.get(c, 0), labelcnt_B.get(c, 0)
        ra, rb = a / total_A * 100, b / total_B * 100
        skew = (ra / rb) if rb > 0 else float('inf')
        lines.append(f"{LABEL_NAMES[c]:^6} | {a:>18} {ra:>7.2f}% | {b:>16} {rb:>7.2f}% | {skew:>11.2f}x")
    lines.append("(쏠림 1.0x = 두 그룹에서 비율이 동일. 1.0x보다 크면 그 클래스가 억울한 거부에 과대표집된 것)")
    lines.append("")

    lines.append("-" * 100)
    lines.append("[2] 텍스트 길이(글자 수) 분포 비교  -  유난히 짧거나 긴가?")
    lines.append("-" * 100)
    lines.append(f"  A(억울한 거부) : {fmt_dist(pct_dist(lens_A), '자')}")
    lines.append(f"  B(정상 통과)   : {fmt_dist(pct_dist(lens_B), '자')}")
    lines.append("")

    lines.append("-" * 100)
    lines.append("[3] 모델 confidence(확신도) 분포 비교  -  확신 낮은 경계선 샘플들인가?")
    lines.append("-" * 100)
    lines.append(f"  A(억울한 거부) : {fmt_dist(pct_dist(train_confs[A_idx] * 100), '%')}")
    lines.append(f"  B(정상 통과)   : {fmt_dist(pct_dist(train_confs[B_idx] * 100), '%')}")
    lines.append("")

    lines.append("-" * 100)
    lines.append("[4] raw_score가 threshold를 넘은 정도(margin)  -  살짝 넘었나, 한참 넘었나?")
    lines.append("-" * 100)
    lines.append(f"  margin = raw_score - 그 샘플의 클래스 threshold")
    lines.append(f"  A(억울한 거부) margin 분포 : {fmt_dist(pct_dist(margin_A))}")
    near_thr = int((margin_A < np.percentile(margin_A, 25)).sum())
    lines.append(f"  -> margin이 하위 25% 구간(threshold를 '살짝'만 넘긴 경계선 케이스): {near_thr}건 / {total_A}건"
                 f" ({near_thr/total_A*100:.1f}%)")
    lines.append("")

    lines.append("-" * 100)
    lines.append("[5] 억울한 거부(A)에서 정상군(B) 대비 과대표집된 단어 TOP 20")
    lines.append(f"    (도큐먼트 빈도 기준, A에서 최소 {MIN_DF}건 이상 등장한 단어만 집계)")
    lines.append("-" * 100)
    lines.append(f"{'단어':<12} | {'A 등장':>7} {'A 비율':>8} | {'B 등장':>7} {'B 비율':>8} | {'과대표집 배율':>10}")
    for w, ca, cb, ra, rb, ratio in overrep[:20]:
        lines.append(f"{w:<12} | {ca:>7} {ra:>7.2f}% | {cb:>7} {rb:>7.2f}% | {ratio:>9.2f}x")
    lines.append("")

    lines.append("-" * 100)
    lines.append("[참고] 억울한 거부(A) 샘플 랜덤 12건 (margin이 큰 순으로 정렬한 뒤 상위/하위 섞어 표본)")
    lines.append("-" * 100)
    order = np.argsort(-margin_A)
    pick = list(order[:6]) + list(order[-6:])
    for k in pick:
        i = A_idx[k]
        text = str(train_df.iloc[int(i)]['text'])[:70]
        lines.append(f"  - 라벨={LABEL_NAMES[train_labels[i]]} conf={train_confs[i]*100:.1f}% "
                     f"margin={margin_A[k]:+.5f} 길이={len(str(train_df.iloc[int(i)]['text']))}자 | {text}")
    lines.append("")

    lines.append("[해석 가이드]")
    lines.append("- [1]에서 특정 클래스(특히 L0처럼 표현이 정형화된 코로나/공지성 클래스)의 쏠림 배율이 1.5x 이상이면")
    lines.append("  -> '그 클래스 특유의 표현 패턴이 학습 임베딩 분포에서 이질적으로 군집되어 있다'는 구조적 신호")
    lines.append("- [2]에서 A가 B보다 유의미하게 짧거나 길면 -> 길이 자체가 임베딩을 일반 군집에서 밀어내는 요인일 수 있음")
    lines.append("- [3]에서 A의 confidence가 B와 큰 차이 없이 높다면 -> '모델은 확신하는데 임베딩만 이상치로 찍히는' 모순 -> OOD score가 모델 확신도와 독립적인 별개 신호라는 근거")
    lines.append("- [4]에서 margin 하위 25% 비율이 높다면 -> threshold를 살짝 조정(소폭 완화)하는 것만으로 상당수 구제 가능 (저위험 튜닝 여지)")
    lines.append("- [5]에서 특정 템플릿성 단어가 과대표집되면 -> 그 표현을 포함한 메시지들이 임베딩 공간에서 별도 군집을 이뤄 OOD로 오판되는 원인일 수 있음")

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    for line in lines:
        print(line)
    print(f"\n결과 저장: {OUT_TXT}")


if __name__ == '__main__':
    main()
