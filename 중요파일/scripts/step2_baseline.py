"""
Step 2: TF-IDF + Logistic Regression 베이스라인 모델

- 특징 추출: TF-IDF (문자 n-gram, 2~4)
- 분류기: Logistic Regression (class_weight='balanced')
- 평가: Accuracy, Macro F1, Weighted F1, Confusion Matrix
"""

import pandas as pd
import numpy as np
import joblib
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

LABEL_NAMES = ['긴급', '주의', '일반']

# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
print("데이터 로드 중...")
train_df = pd.read_csv('data_train.csv', encoding='utf-8-sig')
val_df   = pd.read_csv('data_val.csv',   encoding='utf-8-sig')
test_df  = pd.read_csv('data_test.csv',  encoding='utf-8-sig')

X_train, y_train = train_df['메시지내용'].fillna(''), train_df['label']
X_val,   y_val   = val_df['메시지내용'].fillna(''),   val_df['label']
X_test,  y_test  = test_df['메시지내용'].fillna(''),  test_df['label']

print(f"Train {len(X_train):,}  /  Val {len(X_val):,}  /  Test {len(X_test):,}")
print(f"클래스 분포 (train):\n{train_df['label_name'].value_counts().to_string()}\n")

# ─────────────────────────────────────────────
# 모델 정의 및 학습
# ─────────────────────────────────────────────
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='char_wb',     # 문자 단위 n-gram (한국어 형태소 분리 없이 효과적)
        ngram_range=(2, 4),
        max_features=100_000,
        sublinear_tf=True,      # log(1+tf) 변환으로 고빈도 단어 가중치 완화
        min_df=2,
    )),
    ('clf', LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',  # 클래스 불균형 자동 보정
        solver='lbfgs',
        random_state=42,
    )),
])

print("모델 학습 중...")
t0 = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - t0
print(f"학습 완료: {train_time:.1f}초")

# ─────────────────────────────────────────────
# 평가
# ─────────────────────────────────────────────
def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    acc   = accuracy_score(y, preds)
    f1_macro    = f1_score(y, preds, average='macro',    zero_division=0)
    f1_weighted = f1_score(y, preds, average='weighted', zero_division=0)

    print(f"\n[{split_name}]")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Macro F1        : {f1_macro:.4f}")
    print(f"  Weighted F1     : {f1_weighted:.4f}")
    print(classification_report(y, preds, target_names=LABEL_NAMES, zero_division=0))
    return preds, {'acc': acc, 'macro_f1': f1_macro, 'weighted_f1': f1_weighted}

val_preds,  val_metrics  = evaluate(pipeline, X_val,  y_val,  'Validation')
test_preds, test_metrics = evaluate(pipeline, X_test, y_test, 'Test')

# ─────────────────────────────────────────────
# 혼동 행렬 시각화
# ─────────────────────────────────────────────
def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13)

    for ax, data, fmt, subtitle in zip(
        axes, [cm, cm_norm], ['d', '.2f'], ['절대값', '비율 (행 기준)']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
        ax.set_xlabel('예측')
        ax.set_ylabel('실제')
        ax.set_title(subtitle)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"혼동 행렬 저장: {filename}")

plot_confusion(y_test, test_preds,
               'TF-IDF + LR 베이스라인 - 혼동 행렬 (Test)',
               'baseline_confusion_matrix.png')

# ─────────────────────────────────────────────
# 클래스별 주요 특성 시각화 (상위 20개 단어/문자 n-gram)
# ─────────────────────────────────────────────
tfidf = pipeline.named_steps['tfidf']
clf   = pipeline.named_steps['clf']
feature_names = np.array(tfidf.get_feature_names_out())

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('클래스별 TF-IDF 상위 특성 (n-gram)', fontsize=13)

for idx, (ax, label_name) in enumerate(zip(axes, LABEL_NAMES)):
    coefs = clf.coef_[idx]
    top_idx = np.argsort(coefs)[-20:][::-1]
    top_features = feature_names[top_idx]
    top_coefs    = coefs[top_idx]

    colors_bar = ['#e74c3c' if c > 0 else '#3498db' for c in top_coefs]
    ax.barh(range(len(top_features)), top_coefs[::-1], color=colors_bar[::-1])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=8)
    ax.set_title(f'{label_name} 클래스')
    ax.set_xlabel('계수 값')
    ax.axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('baseline_top_features.png', dpi=150, bbox_inches='tight')
print("특성 중요도 저장: baseline_top_features.png")

# ─────────────────────────────────────────────
# 모델 및 결과 저장
# ─────────────────────────────────────────────
joblib.dump(pipeline, 'baseline_model.pkl')
print("모델 저장: baseline_model.pkl")

results_df = pd.DataFrame({
    'model': ['TF-IDF + LR'],
    'val_acc': [val_metrics['acc']],
    'val_macro_f1': [val_metrics['macro_f1']],
    'test_acc': [test_metrics['acc']],
    'test_macro_f1': [test_metrics['macro_f1']],
    'test_weighted_f1': [test_metrics['weighted_f1']],
    'train_time_sec': [round(train_time, 1)],
})
results_df.to_csv('results.csv', index=False, encoding='utf-8-sig')
print("결과 저장: results.csv")

# ─────────────────────────────────────────────
# 오분류 예시 분석
# ─────────────────────────────────────────────
test_df = test_df.copy()
test_df['pred'] = test_preds

errors = test_df[test_df['label'] != test_df['pred']]
print(f"\n오분류 수: {len(errors):,}건 / {len(test_df):,}건 ({len(errors)/len(test_df)*100:.1f}%)")

# 긴급 → 일반 오분류 (가장 위험한 케이스) 샘플 출력
worst = errors[(errors['label'] == 0) & (errors['pred'] == 2)]
if len(worst) > 0:
    with open('baseline_errors.txt', 'w', encoding='utf-8') as f:
        f.write(f"=== 긴급 → 일반 오분류 샘플 (상위 20개) ===\n")
        for _, row in worst.head(20).iterrows():
            f.write(f"[실제:긴급 → 예측:일반]\n{row['메시지내용'][:150]}\n\n")
    print("오분류 샘플 저장: baseline_errors.txt")

print("\n=== 베이스라인 완료 ===")
print(f"Test Accuracy  : {test_metrics['acc']:.4f}")
print(f"Test Macro F1  : {test_metrics['macro_f1']:.4f}")
