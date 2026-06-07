"""
v9n 모델 R2/F1/Recall/Accuracy 계산
- 원본 테스트셋: confusion_matrix_model_v9n.png 에서 읽은 정확한 confusion matrix 사용
- 마스킹 테스트셋: verify_generalization_model_v9n.txt 의 F1/Recall 데이터 사용
"""
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, f1_score, recall_score, precision_score

LABEL_NAMES = ['긴급아님(L0)', '낮음(L1)', '중간(L2)', '높음(L3)', '매우높음(L4)']
N_LABELS = 5

# ── 원본 테스트셋: confusion matrix PNG에서 읽은 값 ──────────────────────────
# rows = True Label(0~4), cols = Predicted Label(0~4)
cm = np.array([
    [14698,    3,    9,    0,    2],  # True L0
    [    1, 6007,    4,   17,    1],  # True L1
    [    0,    0, 1838,    2,    6],  # True L2
    [    0,    0,    5, 1648,    0],  # True L3
    [    0,    0,    0,    0, 1071],  # True L4
])

# (true, pred) 쌍 구성
labels = []
preds  = []
for tc in range(N_LABELS):
    for pc in range(N_LABELS):
        n = cm[tc, pc]
        labels.extend([tc] * n)
        preds.extend([pc] * n)
labels = np.array(labels)
preds  = np.array(preds)

def compute_and_print(tag, labels, preds, n_per_class):
    N = len(labels)
    acc    = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average='macro', zero_division=0)
    r2_ord = r2_score(labels, preds)

    print(f"\n{'='*70}")
    print(f"  {tag}")
    print(f"{'='*70}")
    print(f"\n  {'레벨':<16} {'F1':>8} {'Recall':>8} {'R2(이진)':>10}")
    print(f"  {'-'*46}")

    for c in range(N_LABELS):
        tp = cm[c, c] if tag == '원본 테스트셋' else None  # masked는 별도 처리
        f1_c  = f1_score(labels, preds, labels=[c], average='macro', zero_division=0)
        rec_c = recall_score(labels, preds, labels=[c], average='macro', zero_division=0)
        y_bin = (labels == c).astype(int)
        p_bin = (preds  == c).astype(int)
        r2_c  = r2_score(y_bin, p_bin)
        print(f"  L{c} {LABEL_NAMES[c]:<14} {f1_c*100:>7.2f}%  {rec_c*100:>7.2f}%  {r2_c:>9.4f}")

    print(f"  {'-'*46}")
    print(f"  {'전체 (Macro)':<16} {f1_mac*100:>7.2f}%              {r2_ord:>9.4f}  ← R2 ordinal")
    print(f"\n  Accuracy: {acc*100:.4f}%   |   총 {N:,}건")

# ── 원본 테스트셋 출력 ────────────────────────────────────────────────────────
compute_and_print('원본 테스트셋', labels, preds, cm.sum(axis=1))

# ── 마스킹 테스트셋: verify_generalization_model_v9n.txt 기반 ─────────────────
# F1 / Recall per class (verify script output에서)
masked_stats = {
    # class: (F1, Recall, n_total, n_misclassified)
    0: (0.9991, 0.9988, 14711, 18),
    1: (0.9942, 0.9915,  6029, 51),
    2: (0.9800, 0.9800,  1852, 37),
    3: (0.9783, 0.9860,  1648, 23),
    4: (0.9857, 0.9935,  1072,  7),
}
masked_acc    = 0.9946
masked_f1_mac = 0.9875
N_masked = 25312

print(f"\n{'='*70}")
print(f"  마스킹 테스트셋  (verify_generalization_model_v9n.txt 기반)")
print(f"{'='*70}")
print(f"\n  {'레벨':<16} {'F1':>8} {'Recall':>8} {'R2(이진)':>10}")
print(f"  {'-'*46}")

for c, (f1_c, rec_c, n_c, fn) in masked_stats.items():
    # Precision = F1*R / (2R - F1)
    denom = 2 * rec_c - f1_c
    prec_c = f1_c * rec_c / denom if denom > 1e-9 else 1.0
    tp = n_c - fn
    fp = max(0, round(tp * (1 / prec_c - 1)))

    # binary R2: SS_res = FN + FP, SS_tot = n_c*(N-n_c)/N
    ss_tot = n_c * (N_masked - n_c) / N_masked
    ss_res = fn + fp
    r2_c   = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print(f"  L{c} {LABEL_NAMES[c]:<14} {f1_c*100:>7.2f}%  {rec_c*100:>7.2f}%  {r2_c:>9.4f}"
          f"  (TP={tp}, FP~{fp})")

print(f"  {'-'*46}")

# 마스킹 ordinal R2 추정
# 원본 SS_res=184 (50개 오류), 마스킹은 136개 오류
# 추가된 86개 오류는 키워드 마스킹 후 주로 인접 클래스로 감 (1칸 오류)
ss_res_orig = 184  # 원본 confusion matrix에서 정확히 계산한 값
extra_errors = sum(v[3] for v in masked_stats.values()) - 50  # 86개 추가 오류
ss_res_masked_est = ss_res_orig + extra_errors * 1  # 추가 오류는 인접(1칸) 가정

n_per_class = {c: v[2] for c, v in masked_stats.items()}
y_mean = sum(c * n for c, n in n_per_class.items()) / N_masked
ss_tot_ord = sum(n_per_class[c] * (c - y_mean)**2 for c in range(N_LABELS))

r2_masked_est = 1 - ss_res_masked_est / ss_tot_ord

print(f"  {'전체 (Macro)':<16} {masked_f1_mac*100:>7.2f}%              ~{r2_masked_est:.4f}  (R2 ordinal 추정)")
print(f"  (마스킹 CM 없음: 추가 86개 오류 인접 클래스 가정)")
print(f"\n  Accuracy: {masked_acc*100:.2f}%   |   총 {N_masked:,}건")

r2_ord_orig = r2_score(labels, preds)
print(f"\n{'='*70}")
print("  [요약]")
print(f"{'='*70}")
print(f"  지표             원본 테스트셋    마스킹 테스트셋")
print(f"  Accuracy          99.80%          99.46%")
print(f"  Macro F1          99.58%          98.75%")
print(f"  R2 ordinal        {r2_ord_orig:.4f}          ~{r2_masked_est:.4f} (추정)")
