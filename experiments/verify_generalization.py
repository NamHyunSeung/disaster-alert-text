"""
일반화 능력 검증: TF-IDF 키워드 분석 + Train-Test 유사도 + 키워드 마스킹 테스트

결과: results/verify_generalization.txt
"""

import re
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dataset_clean import load_and_split_clean, DisasterDataset
from model import load_model
from utils import compute_metrics

# dataset_v2와 동일한 마스킹 키워드 사용 (일관성 유지)
from dataset_v2 import _MASK_KEYWORDS as MASK_KEYWORDS

LABEL_NAMES = {0: '긴급아님', 1: '낮음', 2: '중간', 3: '높음', 4: '매우높음'}

output_lines = []


def log(msg=''):
    print(msg)
    output_lines.append(msg)


def mask_text(text: str) -> str:
    for kw in sorted(MASK_KEYWORDS, key=len, reverse=True):  # 긴 키워드 먼저
        text = text.replace(kw, ' ')
    return re.sub(r'\s+', ' ', text).strip()


def evaluate_loader(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttids = batch['token_type_ids'].to(device)
            lbls  = batch['label']
            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())
    return all_labels, all_preds


def part1_tfidf_keywords(train_df):
    log("=" * 60)
    log("Part 1: TF-IDF 클래스별 상위 키워드 (One-vs-Rest)")
    log("=" * 60)

    all_texts = train_df['text'].tolist()
    all_labels = train_df['label'].tolist()

    for lbl in range(5):
        binary = [1 if l == lbl else 0 for l in all_labels]
        vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                              max_features=50000, min_df=5)
        X = vec.fit_transform(all_texts)
        X_class = X[np.array(binary) == 1]
        mean_tfidf = np.asarray(X_class.mean(axis=0)).flatten()
        top_idx = mean_tfidf.argsort()[::-1][:15]
        feature_names = vec.get_feature_names_out()
        top_words = [feature_names[i] for i in top_idx]
        n = sum(binary)
        log(f"\nLabel {lbl} ({LABEL_NAMES[lbl]}, n={n:,}):")
        log(f"  {', '.join(top_words)}")


def part2_overlap(train_df, test_df, n_test_sample=500, n_train_sample=10000):
    log("\n" + "=" * 60)
    log(f"Part 2: Train-Test 유사도 (test {n_test_sample}개 × train {n_train_sample}개 샘플)")
    log("=" * 60)

    np.random.seed(42)
    test_sample = test_df.sample(n=min(n_test_sample, len(test_df)), random_state=42)
    train_sample = train_df.sample(n=min(n_train_sample, len(train_df)), random_state=42)

    # Exact match
    train_set = set(train_df['text'].tolist())
    exact = sum(1 for t in test_sample['text'] if t in train_set)
    log(f"\n  Exact match (test → train): {exact}건 / {len(test_sample)}건")

    # TF-IDF Cosine similarity (sparse, 메모리 효율적)
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3),
                          max_features=30000, min_df=2)
    all_texts = train_sample['text'].tolist() + test_sample['text'].tolist()
    vec.fit(all_texts)

    X_train = vec.transform(train_sample['text'].tolist())
    X_test  = vec.transform(test_sample['text'].tolist())

    # 배치 처리로 메모리 절약
    batch_size = 50
    max_sims = []
    for i in range(0, len(test_sample), batch_size):
        sims = cosine_similarity(X_test[i:i+batch_size], X_train)
        max_sims.extend(sims.max(axis=1).tolist())

    scores = np.array(max_sims)
    log(f"\n  TF-IDF Cosine 유사도 (최근접 train 문장):")
    for t in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:
        pct = (scores >= t).mean() * 100
        log(f"    ≥ {t:.2f}: {pct:5.1f}%")
    log(f"\n  평균: {scores.mean():.3f}  중앙값: {np.median(scores):.3f}  최대: {scores.max():.3f}")

    # 상위 5개 유사 쌍 출력
    top5_idx = np.argsort(scores)[::-1][:5]
    test_texts = test_sample['text'].tolist()
    train_texts = train_sample['text'].tolist()
    log(f"\n  [가장 유사한 test-train 쌍 Top 5]")
    for rank, i in enumerate(top5_idx, 1):
        batch_i = i // batch_size
        within_batch = i % batch_size
        sims_row = cosine_similarity(X_test[i:i+1], X_train).flatten()
        best_j = sims_row.argmax()
        log(f"  #{rank} sim={scores[i]:.3f}")
        log(f"    test : {test_texts[i][:80]}")
        log(f"    train: {train_texts[best_j][:80]}")


def part3_keyword_masking(test_df, tokenizer, model, device):
    log("\n" + "=" * 60)
    log("Part 3: 키워드 마스킹 후 성능 변화")
    log("=" * 60)
    log(f"마스킹 키워드 {len(MASK_KEYWORDS)}개: {', '.join(MASK_KEYWORDS[:8])} ...")

    # 원본 평가
    test_ds_orig = DisasterDataset(test_df, tokenizer, max_length=128)
    test_loader_orig = DataLoader(test_ds_orig, batch_size=64, shuffle=False, num_workers=0)
    y_true, y_pred_orig = evaluate_loader(model, test_loader_orig, device)
    acc_orig, f1_orig, prec_orig, rec_orig, f1_per_orig = compute_metrics(y_true, y_pred_orig)

    # 마스킹 후 평가
    test_df_masked = test_df.copy()
    test_df_masked['text'] = test_df_masked['text'].apply(mask_text)
    masked_count = (test_df_masked['text'] != test_df['text']).sum()
    masked_ratio = masked_count / len(test_df) * 100

    test_ds_masked = DisasterDataset(test_df_masked, tokenizer, max_length=128)
    test_loader_masked = DataLoader(test_ds_masked, batch_size=64, shuffle=False, num_workers=0)
    _, y_pred_masked = evaluate_loader(model, test_loader_masked, device)
    acc_masked, f1_masked, prec_masked, rec_masked, f1_per_masked = compute_metrics(y_true, y_pred_masked)

    log(f"\n  마스킹된 문장 비율: {masked_ratio:.1f}% ({masked_count:,}/{len(test_df):,})")
    log(f"\n  {'지표':<12} {'원본':>10} {'마스킹 후':>10} {'변화':>9}")
    log(f"  {'-'*44}")
    log(f"  {'Accuracy':<12} {acc_orig*100:>9.2f}% {acc_masked*100:>9.2f}% {(acc_masked-acc_orig)*100:>+8.2f}%")
    log(f"  {'Macro F1':<12} {f1_orig*100:>9.2f}% {f1_masked*100:>9.2f}% {(f1_masked-f1_orig)*100:>+8.2f}%")

    log(f"\n  클래스별 F1 / Recall 변화:")
    for lbl in range(5):
        f1_b = f1_per_orig[lbl]
        f1_a = f1_per_masked[lbl]
        rec_b = rec_orig[lbl]
        rec_a = rec_masked[lbl]
        bar = '▼' if f1_a < f1_b else '▲' if f1_a > f1_b else '='
        log(f"    Label {lbl} ({LABEL_NAMES[lbl]:6s}): F1 {f1_b*100:6.2f}%→{f1_a*100:6.2f}%  Recall {rec_b*100:6.2f}%→{rec_a*100:6.2f}%  {bar}{abs(f1_a-f1_b)*100:+.2f}%")

    # 마스킹 후 오분류 패턴
    log(f"\n  마스킹 후 오분류 증가 (클래스별):")
    y_true_arr = np.array(y_true)
    y_orig_arr = np.array(y_pred_orig)
    y_masked_arr = np.array(y_pred_masked)
    for lbl in range(5):
        mask_lbl = y_true_arr == lbl
        orig_wrong  = (y_orig_arr[mask_lbl] != y_true_arr[mask_lbl]).sum()
        masked_wrong = (y_masked_arr[mask_lbl] != y_true_arr[mask_lbl]).sum()
        n = mask_lbl.sum()
        log(f"    Label {lbl} ({LABEL_NAMES[lbl]:6s}): {orig_wrong}건 → {masked_wrong}건  (+{masked_wrong-orig_wrong}건) / {n:,}건")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v2', action='store_true', help='model_v2 + dedup_v2 데이터 사용')
    parser.add_argument('--v3', action='store_true', help='model_v3 + dedup_v3 데이터 사용')
    parser.add_argument('--v4', action='store_true', help='model_v4 + dedup_v3 데이터 사용')
    parser.add_argument('--v5', action='store_true', help='model_v5 + dedup_v3 데이터 사용 (KoELECTRA-small)')
    parser.add_argument('--v6', action='store_true', help='model_v6 + dedup_v3 데이터 사용 (KoELECTRA-small + 확장 마스킹)')
    parser.add_argument('--v7', action='store_true', help='model_v7 + dedup_v3 데이터 사용 (KoELECTRA-small + 비대칭 증강)')
    parser.add_argument('--v8', action='store_true', help='model_v8 + dedup_v3 데이터 사용 (KoELECTRA-small + masked FT)')
    parser.add_argument('--v9', action='store_true', help='model_v9 + dedup_v3 데이터 사용 (KoELECTRA-small + KL Consistency)')
    parser.add_argument('--model_dir', type=str, default=None, help='커스텀 모델 경로 (--v9 override)')
    parser.add_argument('--tok_dir',   type=str, default=None, help='커스텀 토크나이저 경로 (--v9 override)')
    parser.add_argument('--out',       type=str, default=None, help='결과 파일 경로 (--v9 override)')
    parser.add_argument('--show_errors', action='store_true', help='마스킹 후 신규 오분류 샘플 상세 출력')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")

    log("\n데이터 로드 중...")
    if args.v9:
        from dataset_v2 import load_and_split_v2, DisasterDatasetAug
        train_df, val_df, test_df = load_and_split_v2('중요파일/data/raw/재난문자_레이블링결과_dedup_v3.xlsx')
        _ds_cls = lambda df, tok: DisasterDatasetAug(df, tok, max_length=128, augment=False)
        model_dir = args.model_dir or 'model_v9'
        tok_dir   = args.tok_dir   or 'tokenizer_v9'
        tag       = model_dir.replace('/', '_').replace('\\', '_')
        out_path  = args.out or f'results/verify_generalization_{tag}.txt'
        log(f"모드: v9 (model={model_dir}, tok={tok_dir}, KoELECTRA-small + KL Consistency)")
    elif args.v8:
        from dataset_v2 import load_and_split_v2, DisasterDatasetAug
        train_df, val_df, test_df = load_and_split_v2('중요파일/data/raw/재난문자_레이블링결과_dedup_v3.xlsx')
        _ds_cls = lambda df, tok: DisasterDatasetAug(df, tok, max_length=128, augment=False)
        model_dir = 'model_v8'
        tok_dir   = 'tokenizer_v7'
        out_path  = 'results/verify_generalization_v8.txt'
        log(f"모드: v8 (model_v8 + dedup_v3, KoELECTRA-small + masked FT)")
    elif args.v7:
        from dataset_v2 import load_and_split_v2, DisasterDatasetAug
        train_df, val_df, test_df = load_and_split_v2('중요파일/data/raw/재난문자_레이블링결과_dedup_v3.xlsx')
        _ds_cls = lambda df, tok: DisasterDatasetAug(df, tok, max_length=128, augment=False)
        model_dir = 'model_v7'
        tok_dir   = 'tokenizer_v7'
        out_path  = 'results/verify_generalization_v7.txt'
        log(f"모드: v7 (model_v7 + dedup_v3, KoELECTRA-small + 비대칭 증강)")
    elif args.v6:
        from dataset_v2 import load_and_split_v2, DisasterDatasetAug
        train_df, val_df, test_df = load_and_split_v2('중요파일/data/raw/재난문자_레이블링결과_dedup_v3.xlsx')
        _ds_cls = lambda df, tok: DisasterDatasetAug(df, tok, max_length=128, augment=False)
        model_dir = 'model_v6'
        tok_dir   = 'tokenizer_v6'
        out_path  = 'results/verify_generalization_v6.txt'
        log(f"모드: v6 (model_v6 + dedup_v3, KoELECTRA-small + 확장 마스킹 38개)")
    elif args.v5:
        from dataset_v2 import load_and_split_v2, DisasterDatasetAug
        train_df, val_df, test_df = load_and_split_v2('중요파일/data/raw/재난문자_레이블링결과_dedup_v3.xlsx')
        _ds_cls = lambda df, tok: DisasterDatasetAug(df, tok, max_length=128, augment=False)
        model_dir = 'model_v5'
        tok_dir   = 'tokenizer_v5'
        out_path  = 'results/verify_generalization_v5.txt'
        log(f"모드: v5 (model_v5 + dedup_v3, KoELECTRA-small)")
    elif args.v4:
        from dataset_v2 import load_and_split_v2, DisasterDatasetAug
        train_df, val_df, test_df = load_and_split_v2('중요파일/data/raw/재난문자_레이블링결과_dedup_v3.xlsx')
        _ds_cls = lambda df, tok: DisasterDatasetAug(df, tok, max_length=128, augment=False)
        model_dir = 'model_v4'
        tok_dir   = 'tokenizer'
        out_path  = 'results/verify_generalization_v4.txt'
        log(f"모드: v4 (model_v4 + dedup_v3)")
    elif args.v3:
        from dataset_v2 import load_and_split_v2, DisasterDatasetAug
        train_df, val_df, test_df = load_and_split_v2('중요파일/data/raw/재난문자_레이블링결과_dedup_v3.xlsx')
        _ds_cls = lambda df, tok: DisasterDatasetAug(df, tok, max_length=128, augment=False)
        model_dir = 'model_v3'
        tok_dir   = 'tokenizer'
        out_path  = 'results/verify_generalization_v3.txt'
        log(f"모드: v3 (model_v3 + dedup_v3)")
    elif args.v2:
        from dataset_v2 import load_and_split_v2, DisasterDatasetAug
        train_df, val_df, test_df = load_and_split_v2()
        _ds_cls = lambda df, tok: DisasterDatasetAug(df, tok, max_length=128, augment=False)
        model_dir = 'model_v2'
        tok_dir   = 'tokenizer'
        out_path  = 'results/verify_generalization_v2.txt'
        log(f"모드: v2 (model_v2 + dedup_v2)")
    else:
        _ds_cls = lambda df, tok: DisasterDataset(df, tok, max_length=128)
        train_df, val_df, test_df = load_and_split_clean()
        model_dir = 'model_clean'
        tok_dir   = 'tokenizer'
        out_path  = 'results/verify_generalization.txt'
        log(f"모드: v1 (model_clean)")
    log(f"Train: {len(train_df):,} / Val: {len(val_df):,} / Test: {len(test_df):,}")

    part1_tfidf_keywords(train_df)
    part2_overlap(train_df, test_df)

    log("\n모델·토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    model = load_model(model_dir, num_labels=5).to(device)

    # part3에서 ds_cls 사용하도록 내부 함수 재정의
    def _part3(test_df, tokenizer, model, device):
        log("\n" + "=" * 60)
        log("Part 3: 키워드 마스킹 후 성능 변화")
        log("=" * 60)
        log(f"마스킹 키워드 {len(MASK_KEYWORDS)}개: {', '.join(MASK_KEYWORDS[:8])} ...")

        test_ds_orig = _ds_cls(test_df, tokenizer)
        test_loader_orig = DataLoader(test_ds_orig, batch_size=64, shuffle=False, num_workers=0)
        y_true, y_pred_orig = evaluate_loader(model, test_loader_orig, device)
        acc_orig, f1_orig, prec_orig, rec_orig, f1_per_orig = compute_metrics(y_true, y_pred_orig)

        test_df_masked = test_df.copy()
        test_df_masked['text'] = test_df_masked['text'].apply(mask_text)
        masked_count = (test_df_masked['text'] != test_df['text']).sum()
        masked_ratio = masked_count / len(test_df) * 100

        test_ds_masked = _ds_cls(test_df_masked, tokenizer)
        test_loader_masked = DataLoader(test_ds_masked, batch_size=64, shuffle=False, num_workers=0)
        _, y_pred_masked = evaluate_loader(model, test_loader_masked, device)
        acc_masked, f1_masked, prec_masked, rec_masked, f1_per_masked = compute_metrics(y_true, y_pred_masked)

        log(f"\n  마스킹된 문장 비율: {masked_ratio:.1f}% ({masked_count:,}/{len(test_df):,})")
        log(f"\n  {'지표':<12} {'원본':>10} {'마스킹 후':>10} {'변화':>9}")
        log(f"  {'-'*44}")
        log(f"  {'Accuracy':<12} {acc_orig*100:>9.2f}% {acc_masked*100:>9.2f}% {(acc_masked-acc_orig)*100:>+8.2f}%")
        log(f"  {'Macro F1':<12} {f1_orig*100:>9.2f}% {f1_masked*100:>9.2f}% {(f1_masked-f1_orig)*100:>+8.2f}%")

        log(f"\n  클래스별 F1 / Recall 변화:")
        for lbl in range(5):
            f1_b = f1_per_orig[lbl]
            f1_a = f1_per_masked[lbl]
            rec_b = rec_orig[lbl]
            rec_a = rec_masked[lbl]
            bar = '▼' if f1_a < f1_b else '▲' if f1_a > f1_b else '='
            log(f"    Label {lbl} ({LABEL_NAMES[lbl]:6s}): F1 {f1_b*100:6.2f}%→{f1_a*100:6.2f}%  Recall {rec_b*100:6.2f}%→{rec_a*100:6.2f}%  {bar}{abs(f1_a-f1_b)*100:+.2f}%")

        log(f"\n  마스킹 후 오분류 증가 (클래스별):")
        y_true_arr = np.array(y_true)
        y_orig_arr = np.array(y_pred_orig)
        y_masked_arr = np.array(y_pred_masked)
        for lbl in range(5):
            mask_lbl = y_true_arr == lbl
            orig_wrong   = (y_orig_arr[mask_lbl]   != y_true_arr[mask_lbl]).sum()
            masked_wrong = (y_masked_arr[mask_lbl]  != y_true_arr[mask_lbl]).sum()
            n = mask_lbl.sum()
            log(f"    Label {lbl} ({LABEL_NAMES[lbl]:6s}): {orig_wrong}건 → {masked_wrong}건  (+{masked_wrong-orig_wrong}건) / {n:,}건")

        # 신규 오분류 샘플 상세 출력 (L2/L3/L4)
        if args.show_errors:
            texts_orig_list   = test_df['text'].tolist()
            texts_masked_list = test_df_masked['text'].tolist()
            log(f"\n  [신규 오분류 샘플: 원본 정답 → 마스킹 후 틀림 (L2/L3/L4)]")
            for lbl in [2, 3, 4]:
                newly_wrong = (
                    (y_orig_arr == y_true_arr) &
                    (y_masked_arr != y_true_arr) &
                    (y_true_arr == lbl)
                )
                indices = np.where(newly_wrong)[0]
                log(f"\n  Label {lbl} ({LABEL_NAMES[lbl]}) 신규 오분류 {len(indices)}건 (최대 15건 출력):")
                for i in indices[:15]:
                    log(f"    원본  : {texts_orig_list[i][:90]}")
                    log(f"    마스킹: {texts_masked_list[i][:90]}")
                    log(f"    정답:{lbl} → 예측:{y_masked_arr[i]} ({LABEL_NAMES[y_masked_arr[i]]})")
                    log("")

    _part3(test_df, tokenizer, model, device)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    log(f"\n결과 저장: {out_path}")


if __name__ == '__main__':
    main()
