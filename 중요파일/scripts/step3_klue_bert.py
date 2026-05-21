"""
Step 3: KLUE-BERT 파인튜닝

사전 학습 모델: klue/bert-base (한국어 BERT)
설치: pip install transformers torch datasets

학습 설정:
- Max length  : 128 tokens
- Batch size  : 32 (GPU 메모리 부족 시 16으로 조정)
- Epochs      : 5
- LR          : 2e-5 (AdamW + Linear Warmup)
- Class weight: 불균형 보정
"""

import os, time, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
MODEL_NAME   = "klue/bert-base"
MAX_LENGTH   = 128
EPOCHS       = 5
LR           = 2e-5
WARMUP_RATIO = 0.1
NUM_LABELS   = 3
LABEL_NAMES  = ['긴급', '주의', '일반']
SAVE_DIR     = "klue_bert_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"장치: {device}")

if device.type == "cuda":
    BATCH_SIZE  = 32
    MAX_SAMPLES = None   # 전체 데이터 사용
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    # CPU 환경: 클래스 균형을 맞춰 샘플링, 배치 크기 축소
    BATCH_SIZE  = 16
    MAX_SAMPLES = 10_000  # 클래스당 균형 샘플링 후 총 ~10K
    print("[경고] GPU 없음 -> CPU 모드: 10,000개 샘플로 빠른 실험")
    print("   전체 데이터 학습은 Google Colab (GPU)에서 실행 권장")

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class DisasterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
        )
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings.get('token_type_ids',
                              torch.zeros_like(self.encodings['input_ids']))[idx],
            'labels':         self.labels[idx],
        }

# ─────────────────────────────────────────────
# 평가 함수
# ─────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss    = total_loss / len(loader)
    acc         = accuracy_score(all_labels, all_preds)
    macro_f1    = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, acc, macro_f1, weighted_f1, all_preds, all_labels

# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
print("데이터 로드 중...")
train_df = pd.read_csv('data_train.csv', encoding='utf-8-sig')
val_df   = pd.read_csv('data_val.csv',   encoding='utf-8-sig')
test_df  = pd.read_csv('data_test.csv',  encoding='utf-8-sig')

train_df['메시지내용'] = train_df['메시지내용'].fillna('')
val_df['메시지내용']   = val_df['메시지내용'].fillna('')
test_df['메시지내용']  = test_df['메시지내용'].fillna('')

print(f"원본 Train {len(train_df):,}  /  Val {len(val_df):,}  /  Test {len(test_df):,}")

def stratified_sample(df, per_class, seed=42):
    parts = [
        df[df['label'] == lbl].sample(min(len(df[df['label'] == lbl]), per_class),
                                       random_state=seed)
        for lbl in range(NUM_LABELS)
    ]
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

# CPU 환경: 클래스 균형 맞춘 stratified 샘플링
if MAX_SAMPLES is not None:
    per_class = MAX_SAMPLES // NUM_LABELS
    train_df  = stratified_sample(train_df, per_class)
    val_df    = stratified_sample(val_df,   max(200, MAX_SAMPLES // 10) // NUM_LABELS)
    print(f"샘플링 후 -> Train {len(train_df):,}  /  Val {len(val_df):,}")

print(f"클래스 분포 (train):\n{train_df['label_name'].value_counts().to_string()}\n")

# 클래스 가중치 (불균형 보정)
label_counts = train_df['label'].value_counts().sort_index()
class_weights = torch.tensor(
    [len(train_df) / (NUM_LABELS * label_counts[i]) for i in range(NUM_LABELS)],
    dtype=torch.float,
).to(device)
print(f"클래스 가중치: {class_weights.cpu().numpy().round(3)}")

# ─────────────────────────────────────────────
# 토크나이저 및 데이터셋
# ─────────────────────────────────────────────
print(f"\n토크나이저 로드: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("데이터셋 토크나이징 중... (시간 소요)")
train_dataset = DisasterDataset(train_df['메시지내용'], train_df['label'], tokenizer, MAX_LENGTH)
val_dataset   = DisasterDataset(val_df['메시지내용'],   val_df['label'],   tokenizer, MAX_LENGTH)
test_dataset  = DisasterDataset(test_df['메시지내용'],  test_df['label'],  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# 모델, 옵티마이저, 스케줄러
# ─────────────────────────────────────────────
print(f"\n모델 로드: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

print(f"총 학습 스텝: {total_steps:,}  (워밍업: {warmup_steps:,})")

# ─────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────
history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': []}
best_val_f1  = 0.0
best_epoch   = 0

print("\n학습 시작\n" + "="*60)
total_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss, epoch_steps = 0.0, 0
    epoch_start = time.time()

    for step, batch in enumerate(train_loader, 1):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels         = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        loss = criterion(outputs.logits, labels)
        loss.backward()

        # Gradient Clipping (폭발적 그라디언트 방지)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        epoch_loss  += loss.item()
        epoch_steps += 1

        if step % 200 == 0:
            elapsed = time.time() - epoch_start
            print(f"  Epoch {epoch} | Step {step:>5}/{len(train_loader)} "
                  f"| Loss {epoch_loss/epoch_steps:.4f} | {elapsed:.0f}s")

    avg_train_loss = epoch_loss / epoch_steps

    # Validation
    val_loss, val_acc, val_f1, val_wf1, _, _ = evaluate(model, val_loader, criterion)

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_macro_f1'].append(val_f1)

    epoch_time = time.time() - epoch_start
    print(f"\nEpoch {epoch}/{EPOCHS} ({epoch_time:.0f}s)")
    print(f"  Train Loss : {avg_train_loss:.4f}")
    print(f"  Val   Loss : {val_loss:.4f}  Acc {val_acc:.4f}  Macro F1 {val_f1:.4f}")

    # Best 모델 저장
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch
        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"  [Best] 모델 저장 (Val Macro F1: {best_val_f1:.4f})")
    print()

total_time = time.time() - total_start
print(f"학습 완료: {total_time/60:.1f}분  (Best epoch: {best_epoch})")

# ─────────────────────────────────────────────
# 최종 Test 평가 (Best 모델)
# ─────────────────────────────────────────────
print("\nBest 모델로 Test 평가 중...")
model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
model.to(device)

test_loss, test_acc, test_f1, test_wf1, test_preds, test_labels = \
    evaluate(model, test_loader, criterion)

print("\n=== Test 결과 ===")
print(f"  Loss            : {test_loss:.4f}")
print(f"  Accuracy        : {test_acc:.4f}")
print(f"  Macro F1        : {test_f1:.4f}")
print(f"  Weighted F1     : {test_wf1:.4f}")
print("\n" + classification_report(test_labels, test_preds,
                                   target_names=LABEL_NAMES, zero_division=0))

# ─────────────────────────────────────────────
# 학습 곡선 시각화
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('KLUE-BERT 학습 곡선', fontsize=13)

epochs_x = range(1, EPOCHS + 1)
axes[0].plot(epochs_x, history['train_loss'], 'b-o', label='Train')
axes[0].plot(epochs_x, history['val_loss'],   'r-o', label='Val')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(epochs_x, history['val_acc'], 'g-o')
axes[1].set_title('Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylim(0, 1)

axes[2].plot(epochs_x, history['val_macro_f1'], 'm-o')
axes[2].axvline(best_epoch, color='gray', linestyle='--', label=f'Best (Epoch {best_epoch})')
axes[2].set_title('Validation Macro F1')
axes[2].set_xlabel('Epoch')
axes[2].set_ylim(0, 1)
axes[2].legend()

plt.tight_layout()
plt.savefig('bert_learning_curves.png', dpi=150, bbox_inches='tight')
print("학습 곡선 저장: bert_learning_curves.png")

# ─────────────────────────────────────────────
# 혼동 행렬
# ─────────────────────────────────────────────
cm = confusion_matrix(test_labels, test_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('KLUE-BERT - 혼동 행렬 (Test)', fontsize=13)

for ax, data, fmt, subtitle in zip(
    axes, [cm, cm_norm], ['d', '.2f'], ['절대값', '비율 (행 기준)']
):
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Oranges',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel('예측')
    ax.set_ylabel('실제')
    ax.set_title(subtitle)

plt.tight_layout()
plt.savefig('bert_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("혼동 행렬 저장: bert_confusion_matrix.png")

# ─────────────────────────────────────────────
# 베이스라인과 비교
# ─────────────────────────────────────────────
try:
    results_df = pd.read_csv('results.csv', encoding='utf-8-sig')
except FileNotFoundError:
    results_df = pd.DataFrame()

bert_row = pd.DataFrame([{
    'model': 'KLUE-BERT',
    'val_acc':          max(history['val_acc']),
    'val_macro_f1':     best_val_f1,
    'test_acc':         test_acc,
    'test_macro_f1':    test_f1,
    'test_weighted_f1': test_wf1,
    'train_time_sec':   round(total_time, 1),
}])
results_df = pd.concat([results_df, bert_row], ignore_index=True)
results_df.to_csv('results.csv', index=False, encoding='utf-8-sig')
print("결과 업데이트: results.csv")

print("\n=== 모델 비교 ===")
print(results_df.to_string(index=False))

# 비교 바 차트
if len(results_df) >= 2:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results_df))
    w = 0.25
    ax.bar(x - w, results_df['test_acc'],          w, label='Accuracy',     color='#3498db')
    ax.bar(x,     results_df['test_macro_f1'],     w, label='Macro F1',     color='#e74c3c')
    ax.bar(x + w, results_df['test_weighted_f1'],  w, label='Weighted F1',  color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model'])
    ax.set_ylim(0, 1.1)
    ax.set_title('모델 성능 비교 (Test)')
    ax.set_ylabel('Score')
    ax.legend()
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("모델 비교 저장: model_comparison.png")

# 학습 이력 저장
with open('bert_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\n=== KLUE-BERT 완료 ===")
