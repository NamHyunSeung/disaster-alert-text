"""
KNN-OOD threshold 재보정 (self-match 버그 수정 + held-out val 보정).

문제 1: build_knn_ood_v22.py가 train으로 fit한 인덱스에 train 자신을 질의해서
        각 샘플의 1등 이웃이 자기 자신(거리 0)이 되어버림 → score가 인위적으로 낮게 나옴
        → leave-one-out (k+1 질의 후 자기 자신 제외)으로 수정

문제 2: threshold를 train score의 percentile로 잡으면 정의상 학습 데이터의 일정 비율이
        항상 자기 threshold를 넘게 됨 → held-out val set의 score로 threshold를 보정

실행: python pipeline_v22/scripts/recalibrate_knn_ood_v22.py
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
OLD_META   = os.path.join(ROOT, 'ood', 'knn_ood_v22_meta.pt')
NEW_META   = os.path.join(ROOT, 'ood', 'knn_ood_v22_meta_recalibrated.pt')
BATCH_SIZE = 64
MAX_LEN    = 128
K          = 20
PCT        = 95


def extract(model, loader, device):
    """CLS 임베딩 + 정답 레이블 + 모델 예측 레이블 추출"""
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
            embs.append(out.hidden_states[-1][:, 0, :].cpu())
            labels.append(batch['label'])
            preds.append(out.logits.argmax(dim=-1).cpu())
    return (torch.cat(embs).numpy().astype(np.float32),
            torch.cat(labels).numpy().astype(np.int32),
            torch.cat(preds).numpy().astype(np.int32))


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

    print("train CLS 임베딩 추출 중...")
    train_embs, train_labels, train_preds = extract(model, train_loader, device)
    print(f"  shape: {train_embs.shape}")

    print("val CLS 임베딩 추출 중...")
    val_embs, val_labels, val_preds = extract(model, val_loader, device)
    print(f"  shape: {val_embs.shape}")

    print(f"\nKNN 인덱스 fit (train, K={K}, metric=cosine)...")
    nn = NearestNeighbors(n_neighbors=K + 1, algorithm='brute', metric='cosine', n_jobs=-1)
    nn.fit(train_embs)

    # ── [수정 1] train: leave-one-out (자기 자신 제외) ──────────────────
    print("train score 계산 중 (leave-one-out, 자기 자신 제외)...")
    dists, idxs = nn.kneighbors(train_embs)              # (N, K+1) — 1등은 자기 자신(거리 0)
    self_is_first = (idxs[:, 0] == np.arange(len(train_embs)))
    print(f"  자기 자신이 1등 이웃인 샘플: {self_is_first.sum()}/{len(train_embs)} "
          f"({self_is_first.mean()*100:.1f}%)")
    train_scores_loo = dists[:, 1:].mean(axis=1)         # 자기 자신을 빼고 K개 평균

    # (참고) 기존 방식: 자기 자신 포함 점수
    train_scores_buggy = dists[:, :K].mean(axis=1)

    # ── [수정 2] val: 보지 못한 in-distribution 데이터로 threshold 보정 ──
    print("val score 계산 중 (train 인덱스에 질의, self-match 없음)...")
    nn_k = NearestNeighbors(n_neighbors=K, algorithm='brute', metric='cosine', n_jobs=-1)
    nn_k.fit(train_embs)
    val_dists, _ = nn_k.kneighbors(val_embs)
    val_scores = val_dists.mean(axis=1)

    # ── 새 threshold (val score의 percentile, 클래스=정답 레이블 기준) ──
    new_global_thr = float(np.percentile(val_scores, PCT))
    new_class_thr = {}
    print(f"\n[새 threshold] (val 기준 p{PCT}, 클래스=정답 레이블)")
    for c in range(5):
        mask = val_labels == c
        if mask.sum() == 0:
            new_class_thr[c] = new_global_thr
            continue
        thr_c = float(np.percentile(val_scores[mask], PCT))
        new_class_thr[c] = thr_c
        print(f"  L{c}: n={mask.sum():>4}  val p50={np.percentile(val_scores[mask],50):.6f}"
              f"  val p{PCT}={thr_c:.6f}")

    # ── 기존(버그) threshold 로드 ──────────────────────────────────────
    old_meta = torch.load(OLD_META, weights_only=False)
    old_global_thr = old_meta['global_threshold']
    old_class_thr  = old_meta['class_thresholds']
    print(f"\n[기존 threshold] (train 자기-매칭 포함, p{PCT})")
    for c in range(5):
        print(f"  L{c}: {old_class_thr.get(c, old_global_thr):.6f}")

    # ── 비교: 학습 데이터 중 OOD로 분류되는 건수 ─────────────────────────
    def count_ood(scores, preds, class_thr, global_thr):
        thr_arr = np.array([class_thr.get(int(p), global_thr) for p in preds])
        flags = scores > thr_arr
        return int(flags.sum()), flags

    old_n, old_flags   = count_ood(train_scores_buggy, train_preds, old_class_thr, old_global_thr)
    new_n_loo, new_flags_loo = count_ood(train_scores_loo, train_preds, new_class_thr, new_global_thr)
    # (참고) self-match 안 고치고 새 threshold만 적용했을 때
    mix_n, _ = count_ood(train_scores_buggy, train_preds, new_class_thr, new_global_thr)
    # (참고) self-match는 고쳤지만 기존(버그) threshold를 적용했을 때
    loo_old_n, _ = count_ood(train_scores_loo, train_preds, old_class_thr, old_global_thr)

    n_train = len(train_embs)
    print("\n" + "=" * 70)
    print("학습 데이터(train) 중 OOD로 분류되는 건수")
    print("=" * 70)
    print(f"  [기존 방식]  train score(자기매칭 포함) + 기존 threshold(자기매칭 train로 보정)")
    print(f"             → {old_n}/{n_train} ({old_n/n_train*100:.2f}%)")
    print(f"  [중간 1]    train score(자기매칭 포함) + 새 threshold(val로 보정)")
    print(f"             → {mix_n}/{n_train} ({mix_n/n_train*100:.2f}%)")
    print(f"  [중간 2]    train score(LOO, self-match 제거) + 기존 threshold")
    print(f"             → {loo_old_n}/{n_train} ({loo_old_n/n_train*100:.2f}%)")
    print(f"  [수정 후]   train score(LOO) + 새 threshold(val로 보정)  ← 최종")
    print(f"             → {new_n_loo}/{n_train} ({new_n_loo/n_train*100:.2f}%)")

    # 검증: val 자체는 약 (100-PCT)% 정도가 OOD로 잡혀야 정상 (정의상)
    val_n, _ = count_ood(val_scores, val_preds, new_class_thr, new_global_thr)
    print(f"\n  [검증] val 자체 중 OOD 분류  → {val_n}/{len(val_embs)} "
          f"({val_n/len(val_embs)*100:.2f}%)  (p{PCT} 보정이므로 이론상 ~{100-PCT}% 근처가 정상)")

    print(f"\n새 메타데이터 저장: {NEW_META}")
    torch.save({
        'k': K,
        'global_threshold': new_global_thr,
        'class_thresholds': new_class_thr,
        'note': f'self-match 제거(LOO) + val(p{PCT}) 보정',
    }, NEW_META)
    print("완료.")


if __name__ == '__main__':
    main()
