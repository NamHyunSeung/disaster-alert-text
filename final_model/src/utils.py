import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix,
)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

LABEL_NAMES = {
    0: '긴급 아님 (0)',
    1: '낮은 긴급성 (1)',
    2: '중간 긴급성 (2)',
    3: '높은 긴급성 (3)',
    4: '매우 높은 긴급성 (4)',
}
SHORT_NAMES = {0: '긴급아님', 1: '낮음', 2: '중간', 3: '높음', 4: '매우높음'}


def compute_metrics(y_true, y_pred):
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall    = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1        = f1_score(y_true, y_pred, average=None, zero_division=0)
    return acc, macro_f1, precision, recall, f1


def print_evaluation(y_true, y_pred):
    acc, macro_f1, precision, recall, f1 = compute_metrics(y_true, y_pred)
    sep = '=' * 58
    print(f"\n{sep}")
    print("=== 최종 평가 결과 ===")
    print(sep)
    print(f"Accuracy:     {acc * 100:.2f}%")
    print(f"Macro F1:     {macro_f1 * 100:.2f}%")
    print(f"\n클래스별 성능:")
    for i in range(5):
        name = LABEL_NAMES[i]
        p = precision[i] * 100 if i < len(precision) else 0
        r = recall[i]    * 100 if i < len(recall)    else 0
        f = f1[i]        * 100 if i < len(f1)        else 0
        print(f"  Label {i} ({name:<20}) - P: {p:5.1f}% / R: {r:5.1f}% / F1: {f:5.1f}%")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n오분류 분석:")
    for i in range(len(cm)):
        total = cm[i].sum()
        wrong = total - cm[i, i]
        rate  = wrong / max(total, 1) * 100
        print(f"  Label {i} ({LABEL_NAMES[i]}): 전체 {total}건 중 {wrong}건 오분류 ({rate:.1f}%)")
    print(sep)
    return acc, macro_f1


def save_evaluation_report(y_true, y_pred, path: str):
    acc, macro_f1, precision, recall, f1 = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    lines = [
        "=== 최종 평가 결과 ===\n",
        f"Accuracy:     {acc * 100:.2f}%\n",
        f"Macro F1:     {macro_f1 * 100:.2f}%\n",
        "\n클래스별 성능:\n",
    ]
    for i in range(5):
        p = precision[i] * 100 if i < len(precision) else 0
        r = recall[i]    * 100 if i < len(recall)    else 0
        f_ = f1[i]       * 100 if i < len(f1)        else 0
        lines.append(
            f"  Label {i} ({LABEL_NAMES[i]:<20}) - P: {p:5.1f}% / R: {r:5.1f}% / F1: {f_:5.1f}%\n"
        )
    lines.append("\n오분류 분석:\n")
    for i in range(len(cm)):
        total = cm[i].sum()
        wrong = total - cm[i, i]
        rate  = wrong / max(total, 1) * 100
        lines.append(
            f"  Label {i} ({LABEL_NAMES[i]}): 전체 {total}건 중 {wrong}건 오분류 ({rate:.1f}%)\n"
        )
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def save_confusion_matrix(y_true, y_pred, path: str):
    cm = confusion_matrix(y_true, y_pred)
    tick_labels = [f"L{i}\n{SHORT_NAMES[i]}" for i in range(5)]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=tick_labels, yticklabels=tick_labels,
    )
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_training_curves(log_df: pd.DataFrame, path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(log_df['epoch'], log_df['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(log_df['epoch'], log_df['val_loss'],   'r-o', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.plot(log_df['epoch'], log_df['val_accuracy'] * 100, 'g-o', label='Val Accuracy')
    ax2.plot(log_df['epoch'], log_df['val_macro_f1'] * 100, 'm-o', label='Val Macro F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('%')
    ax2.set_title('Validation Accuracy & Macro F1')
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
